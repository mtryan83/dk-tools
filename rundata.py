from enum import IntEnum,auto
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import collections
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from itertools import chain
from zipfile import ZipFile,ZIP_DEFLATED
from zipfile import Path as zipPath
import os,io
import logging

from runopts import RunOpts
from virial import Virial


def flatten_list(deep_list: list[list[object]]):
    return list(chain.from_iterable(deep_list))


# This function rounds numbers to the provided number of significant figures
def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def nextpow10(x,*,n:int = 0,up=True):
    pow10 = np.log10(x)
    if up:
        pow10 = np.ceil(pow10)
        pow10 += n
    else:
        pow10 = np.floor(pow10)
        pow10 -= n
    return 10**pow10


class RunDataException(Exception):
    pass


class RunTypeException(Exception):
    pass


class RunTypes(IntEnum):
    unknown = auto()
    nocollapse = auto()
    revirial = auto()
    efficient = auto()
    inefficient = auto()
    minimal = auto()
    atomic = auto()

    def color(self):
        return RunTypes.getColor(self)

    def molecular(self):
        return RunTypes.isMolecular(self)

    def cooling(self):
        return RunTypes.isCooling(self)

    def number():
        return 7

    def getColor(n):
        pc=mpl.cm.get_cmap("viridis",6)
        match(n):
            case 0:
                c=[0,0,0]
            case _:
                c=pc(n)
        return c

    def name(self):
        return RunTypes.getName(self - 1)
    
    @staticmethod
    def getName(n):
        match(n):
            case 0:
                name='Unknown'
            case 1:
                name='No Collapse'
            case 2:
                name='Revirialization'
            case 3:
                name='Efficient'
            case 4:
                name='Inefficient'
            case 5:
                name='Minimal'
            case 6:
                name='Atomic Only'
            case _:
                raise RunTypeException('Impossible enumeration value')
        return name

    @classmethod
    def getNames(cls):
        return cls.__members__.keys()

    @staticmethod
    def isMolecular(rt):
        match(rt):
            case RunTypes.efficient | RunTypes.inefficient:
                return True
            case _:
                return False

    @staticmethod
    def isCooling(rt):
        match(rt):
            case RunTypes.efficient | RunTypes.inefficient | RunTypes.minimal | RunTypes.atomic:
                return True
            case _:
                return False


@dataclass
class RunFlags:
    equil:int = 0  # 0-none, 1-possible, 2-equil & never cooled, 3-equil for >10 Gyr
    molCool:int = 0  # 0-no molecular cooling, 1-some molecular cooling
    rovibCool:int = -1  # nan-no mol cooling, 0-vib only, 1-rot, 2-eff rot
    coolType:int = float('nan')  # cooling type at min temp reached
    effCool:int = 1  # 0-min temperature reached after 10 Gyr, 1-min temp before 10 Gyr
    subsolar:int = 0  # 0-Mjmin>1 Msol, 1-subsolar mass fragment
    highn:int = 0  # if ntot>1e9, don't care if equilibrium

    hasIso:int = 0  # -1-isobaric evolution disabled, 0-no isobaric evolution, 1-isobaric evolution
    isoThresh:int = 0  # 0 - does not cross isobaric threshold, >0 - index where crosses isobaric threshold

    #constraint flags - see restrictions and constraints paper -
    #nonzero generally means the constraint is violated
    smallMass:int = 0  # m/M < 0.01
    cmb:int = 0  # 1 < 10^7 xi^3 rM
    threebody:int = 0  # ra^3 < 10^5 xi^3 rM
    radop:int = 0  # radiative transitions affect ortho/para ratio
    protMol:int = 0  # proton-H2 cooling outcompetes H-H2 cooling
    h3p:int = 0  # h3p is dominant + charge carrier
    h2opacity:int = 0  # 0-transparent to H2 line cooling, 1-rot is opaque, 2-vib is opaque, 3-both opaque
    nonhydro:int = 0  # SPH model breaks down

    @staticmethod
    def printFlagDescriptions():
        msg = [
                "equil: 0-none, 1-possible, 2-equil & never cooled, 3-equil for >10 Gyr",
                "molCool: 0-atomic cooling always dominates, 1-molecular cooling sometimes dominates",
                "rovibCool: nan-no cooling(before 1e10), 0-vib only, 1-rot, 2-eff rot",
                "coolType: cooling type at min temp reached (see comps output of computeLambda)",
                "effCool: 0-min temp reached after 10 Gyr, 1-min temp before 10 Gyr",
                "subsolar: 0-Mjmin>1 Msol, 1-Mjmin<=1 Msol",
                "highn: 0-ntot<1e9, 1-ntot>1e9 (if 1, ignore equil flag)",
                "hasIso: -1-isobaric evolution disabled, 0-no isobaric evolution, 1-isobaric evolution",
                "isoThresh: 0 - does not cross isobaric threshold, >0 - isobaric threshold crossing index",
                "smallMass: 1 - M/m < 100 (re-scaled chem breaks)",
                "cmb: 1 - xi^(-3) rM^(-1) > 10^7 (primordial breakdown)",
                "threebody: 1 - xi^(-3) rM^(-1) ra^3 > 10^5 (3-body time < 2-body)",
                "radop: X-index of first event where radiative transitions change O-P ratio",
                "protMol: X-index of first event where p-H2 cooling beats H-H2",
                "h3p: X-index of first event where H3p is dominant cation",
                "h2opacity:(1) 1-rot H2 is opaque, 2-vib H2 is opaque, 3-both opaque, (2)-index of first event",
                "nonhydro: 1-SPH model breaks down (not computed)",
        ]
        print(f'{m}\n' for m in msg)

    def long_description(self):
        def print_message(pre,flag,options):
            print(f'{pre}{options.get(flag,options["otherwise"])}')
        print_message('equil: ',self.equil,{1:'possible',2:'equil & never cooled',
                                            3:'equil for >10 Gyr','otherwise':'none'})
        print_message('molCool: ',self.molCool,{1:'mol cooling sometimes dominates',
                                                'otherwise':'atomic cooling always dominates'})
        print_message('rovibCool: ',self.rovibCool,{0:'vib only',1:'rot',2:'eff rot',
                                                    'otherwise':'no cooling before 1e10'})
        print(f'coolType: {self.coolType}')
        print_message('effCool: ',self.effCool,{1:'min temp before 10 Gyr','otherwise':'min temp after 10 Gyr'})
        print_message('subsolar: ',self.subsolar,{1:'Mjmin <= 1 Msol','otherwise':'Mjmin > 1 Msol'})
        print_message('highn: ',self.highn,{1:'ntot(end)>1e9 (ignore equil flag)','otherwise':'ntot(end)<1e9'})
        print_message('hasIso: ',self.hasIso,{1:'isobaric evolution occurs',-1:'isobaric evolution disabled',
                                              'otherwise':'no isobaric evolution'})
        print_message('isoThresh: ',(int(self.isoThresh>0)),{1:'temperature crossed isobaric threshold',
                                                             'otherwise':'temperature never crossed isobaric threshold'})
        print_message('smallMass: ',self.smallMass,{1:'M/m ratio too small, re-scaling breakdown','otherwise':'m<<M'})
        print_message('cmb:',self.cmb,{
            1:'spectral distortions violated, primodial abundances wrong','otherwise':'spectral distortions acceptable'})
        print_message('threebody: ',self.threebody,{
            1:'3-body interaction timescale dominates','otherwise':'2-body interaction timescale dominates'})
        print_message('radop: ',int(self.radop>0),{1:'radiative transition changed O-P ratio',
                                                    'otherwise':'radiative transitions are neglible'})
        print_message('protMol: ',int(self.protMol>0),{1:'p-H2 cooling dominated H-H2 cooling',
                                                       'otherwise':'p-H2 cooling neglible'})
        print_message('h3p: ',int(self.h3p>0),{1:'H3p is dominant cation','otherwise':'H3p is ignorable'})
        print_message('h2opacity: ',self.h2opacity[0][0],{1:'rot H2 line is opaque',
                                                         2:'vib H2 line is opaque',
                                                         3:'both H2 lines are opaque',
                                                         'otherwise':'transparent H2 lines'})
        print_message('nonhydro: ',int(self.nonhydro>0),{1:'SPH model breakdown','otherwise':'SPH model valid'})


@dataclass
class TempThresholds:
    Ato:float = float('nan')
    Rot:float = float('nan')
    Vib:float = float('nan')
    Dis:float = float('nan')

    def __init__(self,rE=1,rP=1,rA=1):
        # 13.6 eV=1.578e5 K
        # We'll use 16000 as a standin for the atomic collision excitation peak
        self.Ato = 1.578e4 * rE * rA**2
        self.Rot = 512 * rA**2 * rE**2 / rP
        self.Vib = 5860 * rA**2 * rE**(3 / 2) / rP**(1 / 2)
        self.Dis = 51988 * rE * rA**2


@dataclass
class DensityThresholds:
    Ror = float('nan')
    Rov = float('nan')
    H2d = float('nan')

    def __init__(self,rE=1,rP=1,rA=1):
        # We'll use 10^4 for rovib cooling, 10^3 (from Martin 96)for H2
        # diss
        self.Ror = 1e4 * rA**(8) * rE**(7) * rP**(-4);
        self.Rov = 1e4 * rA**(8) * rE**(19 / 4) * rP**(-7 / 4);
        self.H2d = 1e3 * rA**(8) * rE**(19 / 4) * rP**(-7 / 4);


@dataclass
class OpacityLimits:
    h2rot:float = float('nan')
    h2vib:float = float('nan')

    def __init__(self,rE=1,rP=1,rA=1,*,tau=10):
        self.h2rot = 1e8 * tau**2 * rP**5 / rA**2
        self.h2vib = 1e6 * tau**2 * rE**2 * rP**3 / rA**2


class RunData:
    fname:str
    data:pd.DataFrame
    cool:pd.DataFrame
    heat:pd.DataFrame
    react:pd.DataFrame
    reaction_names:dict
    opts: RunOpts
    runtype:RunTypes = RunTypes.unknown
    flags: RunFlags

    def __init__(self,rd,*,silent=False):

        # assume rd is a single string
        fname,data,cool,heat,react,opts,reaction_names = RunData.loadRunData(rd)

        self.fname = fname
        self.data = data
        self.cool = cool
        self.heat = heat
        self.react = react
        self.opts = opts
        self.reaction_names = reaction_names
        self.flags = RunFlags
        #self.classify()

    @staticmethod
    def loadRunZip(zipfile:Path):
        zipfile = Path(zipfile)
        zipname = Path(zipfile.name)
        lf = {}
        with ZipFile(zipfile) as archive:
            for suf in ['dat','cool','heat','react']:
                file = f'{zipname.with_suffix(f".{suf}")}.arrow'
                lf[suf] = pd.read_feather(io.BytesIO(archive.read(file)))
            names = lf['react'].columns[1:].values
            lf['opts'] = RunOpts.loadOpts((zipPath(archive)/zipname.with_suffix(".params")).open(),using_file=True)
            lf['reaction_names'] = RunData.readReactionNames(zipPath(archive,at='reactions_verbatim.dat').open(),names)
        return zipfile.name,lf['dat'],lf['cool'],lf['heat'],lf['react'],lf['opts'],lf['reaction_names']

    @staticmethod
    def readReactionNames(filename:Path,names):
        react_names = pd.read_csv(filename,header=None).values
        react_names = [r[0] for r in react_names]
        reaction_names = dict(zip(names[1:],react_names))
        return reaction_names

    @staticmethod
    def loadRunData(filename:Path):
        filename = Path(filename)
        zipfile = filename.with_suffix('.zip')
        if zipfile.exists():
            logging.info('Loading from zipfile')
            return RunData.loadRunZip(zipfile)
        logging.info('First load')
        dat_name = filename.with_suffix(".dat")
        cool_name = filename.with_suffix(".cool")
        heat_name = filename.with_suffix(".heat")
        react_name = filename.with_suffix(".react")
        opts_name = filename.with_suffix(".params")
        reactions_name = filename.with_name("reactions_verbatim.dat")

        dat = pd.read_csv(dat_name,delimiter=r'\s+')

        def convertDataNames(dat:pd.DataFrame):
            dat.rename(columns={"#ntot":"ntot"},inplace=True)

            def convertSigns(name:str):
                name = name.replace('+','p')
                name = name.replace('-','m')
                return name
            dat.rename(columns=convertSigns,inplace=True)
        convertDataNames(dat)
        cool = pd.read_csv(cool_name,delimiter=r"\s+")
        cool.insert(loc=0,column="ntot",value=dat.ntot)
        heat = pd.read_csv(heat_name,delimiter=r"\s+")
        heat.insert(loc=0,column="ntot",value=dat.ntot)
        react = pd.read_csv(react_name,header=0,delimiter=r"\s+",nrows=1)
        names = ["T"]
        for i in range(0,react.shape[1] - 1):
            names.append(f"f{i}")
        react = pd.read_csv(react_name,header=0,names=names,delimiter=r'\s+')
        react.insert(loc=0,column="ntot",value=dat.ntot)
        opts = RunOpts.loadOpts(opts_name)
        reaction_names = RunData.readReactionNames(reactions_name,names)

        def removeIsobaric(dat,*args):
            # Isobaric data has no change in density or anything besides temperature
            # and chemistry in repeated lines. Probably only need start and end points of
            # those
            pdv = dat.pdv.array
            # Find transition points
            ends = np.argwhere(np.diff(pdv)!=0)
            if not any(pdv>0):  # include start and end points if entire run is isobaric
                ends = [0,len(pdv)]
            elif (len(ends) % 2)>1 and pdv[-1]<1:  # if odd number of points, and ending on isobaric, include end point
                ends.append(len(pdv) - 1)
            # Set pdv value for start and end of isobaric regions to 2 so we don't remove them
            pdv[ends[0:2:-2] + 1] = 2
            pdv[ends[1:2:-1]] = 2
            nonisoinds = pdv>0
            dat = dat.loc[nonisoinds]

            for idx,df in enumerate(args):
                if len(nonisoinds)==df.shape[0] + 1:
                    df = df.loc[nonisoinds[0:-1]]
                else:
                    df = df.loc[nonisoinds]
            return dat,*args

        dat,cool,heat,react = removeIsobaric(dat,cool,heat,react)

        def convert2binary(filename:Path,df:pd.DataFrame):
            afile = Path(f'{filename}.arrow')
            df.to_feather(afile)
            return afile

        def compressRun(filedict:dict):
            with ZipFile(filedict['zip'],mode='x',compression=ZIP_DEFLATED) as zipfile:
                for k,v in filedict.items():
                    if 'zip' in k:
                        continue
                    zipfile.write(v,arcname=v.name)
                    #os.rename(v,f'{v}.old')
                    if 'verb' not in k:
                        os.remove(v)

        logging.info(f'Saving as {zipfile}')
        fd = {'zip':zipfile,'dattxt':dat_name,'cooltxt':cool_name,'heattxt':heat_name,
                     'reacttxt':react_name,
                     'dat':convert2binary(dat_name,dat), 'cool':convert2binary(cool_name,cool),
                     'heat':convert2binary(heat_name,heat), 'react':convert2binary(react_name,react),
                     'opts':opts_name,'reactverb':reactions_name,}
        if filename.with_suffix('.err').exists():
            fd['err'] = filename.with_suffix('.err')
        compressRun(fd)

        return filename.name,dat,cool,heat,react,opts,reaction_names

    @staticmethod
    def loadDirectory(dirname:Path):
        dirname = Path(dirname)
        if not dirname.is_dir():
            raise NotADirectoryError(dirname)
        rds = []
        for file in dirname.iterdir():
            if file.name[0]=='.':
                continue
            if file.is_dir():
                for rd in RunData.loadDirectory(file):
                    rds.append(rd)
            else:
                if file.suffix in ['.dat','.zip']:
                    rds.append(RunData(file))
        return rds

    def isDarkRun(self):
        if "QH" not in self.dat.columns:
            return False
        for spec in self.dat.columns:
            if "Q" not in spec:
                continue
            if any(self.dat[spec]>0.5):
                return True
        return False

    def classify(self,*,verbose=False):
        def vprint(*args,**kwargs):
            if verbose:
                print(*args,**kwargs)
            pass
        flags = RunFlags()

        ntot = self.data.ntot.to_numpy();

        try:
            # Check if any isobaric evolution
            vprint("Checking for isobaric evolution")
            if self.opts.noDynDen:
                flags.hasIso = -1
                flags.isoThresh = self.findTsoundCrossing(number=1,emptyIsZero=True)
                #if not flags.isoThresh:
                #    flags.isoThresh = 0
                #else:
                #    flags.isoThresh = flags.isoThresh[0]
            else:
                if any(self.data.pdv<1):
                    flags.hasIso = 1
                    flags.isoThresh = np.argwhere(self.data.pdv<1)[0]
                    if not flags.isoThresh:
                        flags.isoThresh = 0

            # Compute constraint flags
            # For more detail, see restrictions and constraints paper
            # Nonzero means constraint is violated
            vprint("Checking constraints...")
            ra = self.opts.rA
            rM = self.opts.rP
            rm = self.opts.rE
            # Small mass
            vprint("...small mass")
            flags.smallMass = self.opts.M * 1e6 / self.opts.m < 100
            # CMB spectral distortions unaffected
            vprint("...CMB")
            flags.cmb = 1 > 1e7 * self.opts.xi**3 * rM
            # Three-body interaction timescale is shorter than two-body timescale
            vprint("...3-body timescale")
            flags.threebody = ra**3 > 1e5 * self.opts.xi**3 * rM
            # Radiative transitions affect ortho/para ratio
            if True:
                vprint("...Ortho/Para")
                nhp = self.data.QHp.to_numpy() * ntot
                # This value should be checked. Here I just assumed all values were O(1) of Gerlich 1990 values
                gammapOPSM = 1e-10  # 1/s
                radop = np.argwhere(ra**10 * rm**(19 / 2) * rM**(-13 / 2) * nhp * gammapOPSM > 1e21)
                if radop.size:
                    # flag value is first index where constraint violated
                    flags.radop = radop[0]

            # proton-H2 cooling outcompetes H-H2 cooling
            if True:
                vprint("...proton-H2")
                #already have np from radop
                nH = self.data.QH.to_numpy() * ntot
                protMol = np.argwhere(np.sqrt(rM / rm) * nhp / nH > 1e-4)
                if protMol.size:
                    flags.protMol = protMol[0]
            # h3p is dominant + charge carrier
            if True:
                vprint("...H3p dominates")
                # No equation currently available - just check if
                # xH3p>xH2p+xp?
                nh2p = self.data.QH2p.to_numpy() * ntot
                nh3p = self.data.QH3p.to_numpy() * ntot
                h3p = np.argwhere(nh3p > nhp + nh2p)
                if h3p.size:
                    flags.h3p = h3p[0]
            # One or both of the h2 line cooling is opaque
            if True:
                vprint("...Opacity")
                # 0-transparent to H2 line cooling, 1-rot is opaque, 2-vib is opaque, 3-both opaque
                olim = self.getOpacityLimits()
                rot = np.argwhere(ntot>olim.h2rot)
                vib = np.argwhere(ntot>olim.h2vib)
                if not vib.size:
                    vib=np.infty
                else:
                    vib = vib[0]
                if not rot.size:
                    rot=np.infty
                else:
                    rot = rot[0]
                rvind = min(vib,rot)  # If using max, empty corresponds to 0
                rvind = rvind if np.isfinite(rvind) else 0
                flags.h2opacity = [(np.isfinite(rot)) + 2 * (np.isfinite(vib)),rvind]
            # SPH model breaks down
            vprint("...SPH model breakdown (Not Implemented)")
            # We don't have a conditional for this yet

            # Check if enough data to even classify
            vprint("Checking data length")
            if len(ntot)<=4:
                self.runtype = RunTypes.unknown
                self.names = self.getClassNames
                return

            # Equilibrium
            vprint("Checking for equilibrium")
            Gyrinsec = 60 * 60 * 24 * 365 * 1e9
            if 'time' in self.data.columns:
                time = self.data.time.to_numpy()
                tsound = self.data.tsound.to_numpy()
                tff = self.data.tff.to_numpy()
                time = time[np.logical_not(np.isnan(time))]
                dt = np.diff(time)
                dtstff = np.diff(tff - tsound)
                if time[-1] / Gyrinsec > 10:  # 10 Gyr
                    # Stuck in equilibrium until at least our current age. Not
                    # going to produce anything detectable by GW anytime soon
                    flags.equil=3
                elif ((len(dt)>1 and dt[-1]>10 * dt[-2]) or
                      (tsound[-1]<tff[-1] and dtstff[-1]>0)) and ntot[-1]<1e10:
                    # Possibly reentered equilibrium before our effective sim
                    # cutoff or we are in isobaric oscillations and it's getting
                    # worse
                    # Check if we're above or below the virial temperature
                    #Tv = virTempN(self.opts.dp_massG,self.opts.Mhalo,ntot[-1])
                    Tv = self.getTvir(self.data.shape[0])
                    if self.data.Tgas[-1]>Tv:
                        flags.equil=2
                    else:
                        flags.equil=1

            flags.highn = ntot[-1]>1e9

            # Molecular cooling
            vprint("Checking for molecular cooling")
            if 'DARKMOL' in self.cool.columns and 'DARK' in self.cool.columns:
                if any(self.cool.DARKMOL>self.cool.DARK):
                    flags.molCool = 1

            # Efficient Cooling
            # For efficient cooling checks, interested in temps after initial
            # heating
            vprint("Checking for efficient cooling")
            dT = np.diff(self.data.Tgas.to_numpy())
            T = self.data.Tgas.to_numpy()
            nstart=0
            if dT[1]>0:
                coolStart = np.nonzero(dT<0)
                if not np.any(coolStart):
                    T=[]
                else:
                    coolStart = coolStart[0][0]
                    nstart = coolStart
                T = T[coolStart:]

            # should use rho<1e-12 here
            #cool = find(ntot((nstart+1):)>1e10,1) # If this changes, need to change in getFlagDescription
            mu = self.getMu()
            rho = ntot * mu * self.opts.m_p
            coolEnd = np.nonzero(rho[(nstart + 1):]>1e-12)
            if coolEnd[0].size:
                T = T[:coolEnd[0][0]]

            #if not T.size:
                #try:
                    # Run never entered cooling, but it may have undergone
                    # period of reduced heating
                    #T = self.data.Tgas.to_numpy()
                    #ipt = findchangepts(np.log10(T),'MaxNumChanges',2)
                    #T = T[ipt[0]:ipt[1]]
                #except:
                #    pass

            if T.size:
                tempThresh = self.getTempThresholds()
                if any(T<tempThresh.Vib):
                    if any(T<tempThresh.Rot):
                        flags.rovibCool = 2
                    else:
                        flags.rovibCool = 1

                    # Likely molecular cooling/Pop III like
                    #mjsmin,_,mjind = self.findMJeansMinTemp()
                else:
                    flags.rovibCool = 0
                    # Likely atomic cooling/direct collapse?
                    #[mjsmin,_,mjind] = self.findMJeansElbow()
                mjsmin, mjind = self.findMJeans()

                if mjind>1:
                    mjsmin = mjsmin / self.opts.M_sun
                    flags.subsolar=mjsmin<1
                    #[~,mjind] = min(abs(self.data.ntot-mjsminxy(1)))

                    # Currently don't have computeLambda (need John's stuff maybe?)
                    # Will use index of strongest cooling rate
                    #abun = self.data[mjind,:]
                    #abun.Properties.VariableNames = convertChemNames(self.data.Properties.VariableNames,'t2k')
                    #abun{:,3:} = abun{:,3:} * repmat(ntot(mjind),1,width(self.data)-2)
                    #[~,comps] = computeLambda(self.data.Tgas(mjind),'m',self.params.dp_massG,self.params.de_massG*1e6,self.params.Dalpha,abun,self.params.xi)
                    #flags.coolType = comps(1).type
                    flags.coolType = self.cool.columns[np.argmax(self.cool.iloc[mjind,3:])+3]

                    flags.effCool = self.data.time[mjind]<10 * Gyrinsec

            self.flags = flags

            # Classification
            vprint("Computing Classification")
            runtype = RunTypes.unknown

            # Need to include a name for 0 as well, since it gets a color
            # Note that names correspond to class+1
            if not flags.highn:
                if flags.equil>=2:
                    runtype = RunTypes.nocoll
                    return
                elif flags.equil==1:
                    runtype = RunTypes.revirial
                    return
                elif flags.isoThresh and flags.hasIso:
                    runtype = RunTypes.revirial
                    return

            if not flags.molCool:
                runtype = RunTypes.atomic
            else:
                match(flags.rovibCool):
                    case 2:
                        runtype = RunTypes.efficient
                    case 1:
                        runtype = RunTypes.inefficient
                    case 0:
                        runtype = RunTypes.minimal
                    case _:
                        runtype = RunTypes.minimal

            self.runtype = runtype

        except IOError as ex:
            print("Got error",ex)
            self.runtype=RunTypes.unknown

    def getTempThresholds(self):
        rE = self.opts.rE  # self.opts.de_mass / self.opts.m_e
        rP = self.opts.rP  # self.opts.dp_mass / self.opts.m_p
        rA = self.opts.rA  # self.opts.Dalpha / self.opts.alphaN
        return TempThresholds(rE,rP,rA)

    def getDensityThresholds(self):
        rE = self.opts.rE  # self.opts.de_mass / self.opts.m_e
        rP = self.opts.rP  # self.opts.dp_mass / self.opts.m_p
        rA = self.opts.rA  # self.opts.Dalpha / self.opts.alphaN
        return DensityThresholds(rE,rP,rA)

    def getOpacityLimits(self):
        rE = self.opts.rE  # self.opts.de_mass / self.opts.m_e
        rP = self.opts.rP  # self.opts.dp_mass / self.opts.m_p
        rA = self.opts.rA  # self.opts.Dalpha / self.opts.alphaN
        return OpacityLimits(rE,rP,rA)

    def plotTempThresholds(self,ax=None):
        if ax is None:
            ax = plt.gcf().gca()
        tt = self.getTempThresholds()
        ey = [tt.Ato,tt.Rot,tt.Vib,tt.Dis]
        ex = [self.data.ntot[0]] * len(ey)
        cmap = mpl.colormaps['tab10']
        cmap = cmap(range(4,8))
        h = []
        h.append(ax.scatter(ex,ey,s=36,c=cmap,marker="o"))
        h.append(ax.text(2 * ex[0],ey[0],'Atomic',fontsize=6,clip_on=True))
        h.append(ax.text(2 * ex[1],ey[1],'Rotational',fontsize=6,clip_on=True))
        h.append(ax.text(2 * ex[2],ey[2],'Vibrational',fontsize=6,clip_on=True))
        h.append(ax.text(2 * ex[3],ey[3],'$H_2$ Diss',fontsize=6,clip_on=True))
        return h

    def plotDensityThresholds(self,ax=None):
        if ax is None:
            ax = plt.gcf().gca()
        nc = self.getDensityThresholds()
        ex = [nc.Ror,nc.Rov,nc.H2d]
        ey = [self.data.Tgas[0]] * len(ex)
        cmap = mpl.colormaps['tab10']
        cmap = cmap(range(0,3))
        h = []
        h.append(ax.scatter(ex,ey,s=36,c=cmap,marker="x"))
        h.append(ax.text(2 * ex[0],ey[0],'Rot LDL-LTE',rotation=45,fontsize=6,clip_on=True))
        h.append(ax.text(2 * ex[1],ey[1],'Vib LDL-LTE',rotation=45,fontsize=6,clip_on=True))
        h.append(ax.text(2 * ex[2],ey[2],'$H_2$ Diss',rotation=45,fontsize=6,clip_on=True))
        return h

    def plotOpacityLimits(self,ax=None):
        if ax is None:
            ax = plt.gcf().gca()
        ol = self.getOpacityLimits()
        ex = [ol.h2rot,ol.h2vib]
        ey = [self.data.Tgas[0]] * len(ex)
        cmap = mpl.colormaps['tab10']
        cmap = cmap(range(0,len(ex) - 1))
        h = []
        h.append(ax.scatter(ex,ey,s=36,c=cmap,marker="^"))
        h.append(ax.text(2 * ex[0],ey[0],'$H_{2,r}$',fontsize=8,clip_on=True))
        h.append(ax.text(2 * ex[1],ey[1],'$H_{2,v}$',fontsize=8,clip_on=True))
        return h

    def plotTrajectory(self,*,ax=None,includeIsobaric=False,initialize=False):
        eps = np.finfo('float').eps
        T = self.data.Tgas.to_numpy() + eps
        n = self.data.ntot.to_numpy() + eps
        if ax is None:
            plot = plt.figure()
            ax = plot.add_subplot()
            ax.set_xlabel('$n_{tot}$ (cm$^{-3}$)')
            ax.set_ylabel('T (K)')
            ax.grid(which='both',alpha=0.3,axis='both')

        # Set axis limits
        newxl = [nextpow10(n[0],up=False),nextpow10(n[-1])]
        newyl = [nextpow10(np.min(T),up=False),nextpow10(np.max(T))]
        if initialize:
            xl = newxl
            yl = newyl
        else:
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            #print(f'old xl:{xl}\nnew xl:{xl}\nold yl:{yl}\nnew yl:{yl}')
            xl = [np.minimum(xl[0],newxl[0]),np.maximum(xl[-1],newxl[-1])]
            yl = [np.minimum(yl[0],newyl[0]),np.maximum(yl[-1],newyl[-1])]
        ax.set_xlim(xl[0],xl[1])
        ax.set_ylim(yl[0],yl[1])

        col,sty = self.getColorAndStyle()
        h = ax.loglog(n,T,color=col,linestyle=sty)
        time = self.data.time.to_numpy()
        Gyrinsec = 60 * 60 * 24 * 365 * 1e9
        ht = []
        if time[-1] > 10 * Gyrinsec:
            ht[0] = ax.loglog(n[-2:-1],T[-2:-1],color=col,linestyle="dotted",linewidth=0.5)
            ht[1] = ax.loglog(n[-1],T[-1],color=col,linestyle="none",marker="*",)
        elif self.flags.equil:
            ht[0] = ax.loglog(n[-2:-1],T[-2:-1],color=col,linestyle="dotted",linewidth=0.5)
            ht[1] = ax.loglog(n[-1],T[-1],color=col,linestyle="none",marker="d")
        if ht:
            h = [h,ht[0],ht[1]]
        if not includeIsobaric:
            return ax,h if initialize else h
        pdv = bool(self.data.pdv)
        hiso = ax.loglog(n[pdv<1],T[pdv<1],color=col,linestyle="none",marker="s")
        h.append(hiso)
        return ax,h if initialize else h

    def getColorAndStyle(self):
        return self.runtype.color(),"solid"

    def plotTimes(self):
        Myrinsec = 60 * 60 * 24 * 365 * 1e6
        eps = np.finfo('float').eps
        n = self.data.ntot + eps
        tff = self.data.tff / Myrinsec + eps
        tsound = self.data.tsound / Myrinsec + eps
        tc = self.data.tc / Myrinsec + eps
        h = []
        h.append(plt.loglog(n,tff,label="t_{ff}"))
        h.append(plt.loglog(n,tsound,label="t_{sound}"))
        h.append(plt.loglog(n,tc,label="t_{cool}"))
        plt.xlabel("n_{tot} (cm^{-3})")
        plt.ylabel("Time Scale (Myr)")
        plt.legend()
        plt.grid()
        return h

    def processInd(self,ind):
        if ind is None:
            ind = range(0,self.data.shape[0])
        if not isinstance(ind,(collections.abc.Sequence,np.ndarray,pd.DataFrame,pd.Series)):
            return self.data.iloc[[ind],:],ind
        try:
            return self.data.iloc[ind,:],ind
        except NotImplementedError:
            return self.data.loc[ind,:],ind

    def getMu(self,ind=None):
        n, ind = self.processInd(ind)
        p = self.opts
        amu = 1.66054e-24
        mE = p.m_e / amu
        mP = p.m_p / amu
        mM = p.dp_mass / amu
        mm = p.de_mass / amu
        mu = (mE * n.E + mm * n.QE + mP * n.H + 4 * mP * n.HE + mM * n.QH +
              mP * n.Hp + mP * n.Hm + mM * n.QHp + mM * n.QHm + 2 * mP * n.H2
              + 2 * mM * n.QH2 + 2 * mM * n.QH2p + 3 * mM * n.QH3p) /\
            ((n.E + n.QE + n.H + n.HE + n.QH + n.Hp + n.Hm + n.QHp + n.QHm + n.H2 + n.QH2 + n.QH2p + n.QH3p) * mP)
        return mu.to_numpy()

    def getMu_baryon(self,ind=None):
        n, ind = self.processInd(ind)
        p = self.opts
        amu = 1.66054e-24
        mE = p.m_e / amu
        mP = p.m_p / amu
        mM = p.dp_mass / amu
        mm = p.de_mass / amu
        mu = (mP * n.H + 4 * mP * n.HE + mM * n.QH + mP * n.Hp + mP * n.Hm + mM * n.QHp
              + mM * n.QHm + 2 * mP * n.H2 + 2 * mM * n.QH2 + 2 * mM * n.QH2p + 3 * mM * n.QH3p) 
        return mu.to_numpy()

    def getGamma(self,ind=None):
        n, ind = self.processInd(ind)
        gam = (5 * (n.E + n.QE + n.H + n.HE + n.QH) + 7 * (n.H2 + n.QH2)) /\
              (3 * (n.E + n.QE + n.H + n.HE + n.QH) + 5 * (n.H2 + n.QH2))
        return gam.to_numpy()

    def getRho(self,ind=None):
        n, ind = self.processInd(ind)

        mp = self.opts.m_p;
        me = self.opts.m_e;
        mm = self.opts.de_mass;
        mM = self.opts.dp_mass;
        m={'E':me, 'H':mp, 'Hp':mp, 'Hm':mp, 'H2':2 * mp, 'H2p':2 * mp, 'H2m':2 * mp,
           'H3':3 * mp, 'H3p':3 * mp, 'H3m':3 * mp, 'HE':4 * mp, 'HEp':4 * mp, 'HEpp':4 * mp,
           'QE':mm, 'QH':mM, 'QHp':mM, 'QHm':mM, 'QH2':2 * mM, 'QH2p':2 * mM, 'QH2m':2 * mM,
           'QH3':3 * mM, 'QH3p':3 * mM, 'QH3m':3 * mM, 'QG':0,}

        rho = 0;
        names=n.columns
        for na in names[2:]:
            if 'tff' in na:
                break
            rho = rho + n[na] * m[na];

        rho = rho * n.ntot
        return rho.to_numpy()

    def findMJeans(self,mask=None):
        data,ind = self.processInd(mask)
        n = data.ntot.to_numpy()
        T = data.Tgas.to_numpy()
        lgT = np.log10(T)

        # Need to find the global minimum in range [n_0, min(n_opacity,n_LTE,n_f)]
        # where n_opacity and n_LTE are the relevant limits for the current process
        # So if we're below the vib cooling regime, for example, we should only
        # consider the rotational limits. The opacity limits are the hard limits, though,
        # meaning we can go past the LTE transitions, but not opacity
        ol = self.getOpacityLimits()
        tt = self.getTempThresholds()

        ior = np.nonzero(n>ol.h2rot)[0]
        iov = np.nonzero(n>ol.h2vib)[0]
        # Set to end of array if empty
        if not ior.size:
            ior = n.size - 1
        else:
            ior = ior[0]
        if not iov.size:
            iov = n.size - 1
        else:
            iov = iov[0]

        im = np.argmin(lgT)

        # find preferred limit
        # 4 cases: Tr ? Tv, ir ? iv,
        # 3 results: a=im,Tm (no change),  b=ior,Tor, c=iov,Tov
        # give 36 different possible outcomes
        # Case 1: Tr < Tv, ir < iv (SM case)
        #      a  |  a   |  c
        # Tv ------------------
        #      a  |  b   |  b
        # Tr ------------------
        #      a  |  b   |  b
        #         ir     iv
        # Case 2: Tv < Tr, ir < iv
        #      a  |  b   |  b
        # Tr ------------------
        #      a  |  a   |  c
        # Tv ------------------
        #      a  |  a   |  c
        #         ir     iv
        # Case 3 (Tr<Tv,iv<ir) and 4 (Tr<Tv,ir<iv) are the equivalent 
        # of 2 and 1 with b and c swapped. So I think we can simplify these
        # to 2 if statement sets. There might be a further simplification, 
        # but I couldn't find one.
        io1,io2,T1,T2 = [ior,iov,tt.Rot,tt.Vib] if ior<iov else [iov,ior,tt.Vib,tt.Rot]
        T01,T02 = T[io1],T[io2]
        Tm = T[im]
        if (ior<iov) ^ (tt.Rot<tt.Vib):  # case 2,3
            if Tm > T1:
                if im>io1:
                    im = io1
                    Tm = T01
            else:
                if im>io2:
                    im = io2
                    Tm = T02
        else:
            if Tm>T1:
                if im>io2:
                    im = io2
                    Tm = T02
            else:
                if im>io1:
                    im = io1
                    Tm = T01

        return data.Mjeans[im], im

    def getMvir(self):
        rho_ddm = self.getRho(0);
        rho_all = rho_ddm / (self.opts.epsilon * (1 - self.opts.omega_b / self.opts.omega_m))
        gamma = self.getGamma(0)
        v = Virial(z=self.opts.zred, rho=rho_all, mu=self.getMu(0) / self.opts.rP,
                   M_gev=self.opts.M, epsilon=self.opts.epsilon, OmegaM=self.opts.omega_m,
                   OmegaB=self.opts.omega_b, delta=self.opts.delta, gamma=gamma)
        Tv = v.convertTgTv(self.data.Tgas[0],gamma,True)
        mvir = v.Mv(Tv);
        return mvir, v

    def getTvir(self):
        # Only interested in the initial virial Temp
        rho_ddm = self.getRho(0);
        rho_all = rho_ddm / (self.opts.epsilon * (1 - self.opts.omega_b / self.opts.omega_m))
        gamma = self.getGamma(ind)
        v = Virial(z=self.opts.zred, rho=rho_all, mu=self.getMu(0) / self.opts.rP, M_gev=self.opts.M,
                   epsilon=self.opts.epsilon, OmegaM=self.opts.omega_m, OmegaB=self.opts.omega_b,
                   delta=self.opts.delta, gamma=gamma)
        Tv = v.Tv(self.opts.Mhalo);
        return Tv, v

    def computeTsound(self,ind=None,*,MvirScale=(3/2)**(3/2), AcoeffScale=1,Acoeff=0.198):
            #    obj (1,1) {mustBeA(obj,'RunData')}
            #    ind (1,:) {mustBeNumericOrLogical} = 1:height(obj.data)
            #    % The Mvir used in test.f90 has a built in assumption of
            #    % gamma=5/3 and thus is equivalent to Mv=Mv(Tg) instead of
            #    % Mv=Mv(Tv). So we need to multiply the correct Mv by
            #    % (3/2)^(3/2) to get the same Mv used.
            #    MvirScale = (3/2)^(3/2)
            #    % This is equivalent to t_ff = Acoeffscale*t_sound
            #    AcoeffScale = 1;
            #    %Acoeff = (32*G*m_p^(4/3))/(3*pi*kb) * (3/(4*pi))^(2/3) * M_sol^(2/3) * (1 cm^-3)^(1/3)
            #    Acoeff = 0.198; 
            Mvir,vir = self.getMvir()
            Mvir = Mvir * MvirScale
            mu = self.getMu(ind)
            gamma = self.getGamma(ind)
            epsilon = self.opts.epsilon
            om = self.opts.omega_m
            od = om - self.opts.omega_b
            ep = epsilon * od / om
            rho_od = Virial.overdensity(vir)
            nod = rho_od / self.opts.m_p
            rho = self.getRho(ind)
            nadm = rho / self.opts.m_p
            Acoeff = Acoeff * AcoeffScale**2
            Tthresh = Acoeff * (Mvir * ep)**(2 / 3) * mu / gamma * ((1 - ep) * nod + nadm) / nadm**(2 / 3);
            return Tthresh

    def findTsoundCrossing(self,*,number=None,emptyIsZero=True,**kwargs):
        Tsound = self.computeTsound(**kwargs)
        ind = np.nonzero(self.data.Tgas.to_numpy()>Tsound)[0]
        if not ind.size and emptyIsZero:
            return 0
        return ind[0:number]

    def plotTsound(self,*,ax=None,**kwargs):
        if ax is None:
            ax = plt.gcf().gca()
        Tsound = self.computeTsound(**kwargs)
        n = self.data.ntot.to_numpy()
        ax.loglog(n,Tsound,linestyle='dashdot',linewidth=2,label='T_{sound}')

    def plotFFTimes(self,ax=None):
        if ax is None:
            ax = plt.gcf().gca()
        time = self.data.time.to_numpy()
        time = time-time[0]
        tff = self.data.tff.iloc[0]
        if all(time<tff):
            return
        n = 1
        ntot = self.data.ntot.to_numpy()
        T = self.data.Tgas.to_numpy()
        # For some runs, we have acheived >7000 tffs. This is way too
        # many to display, so lets only display at most 10 at a time.
        ncrement = np.ceil(np.floor(time[-1]/tff)/10)

        def plotTff(ax,ntot,T,n):
            h = ax.plot(ntot,T,marker='*',markersize=8,linewidth=2)
            if n==1:
                txt = '$t_{ff}$'
            else:
                txt = f'${n:g}\,t_{{ff}}$'

            tt = ax.text(ntot*2,T,txt,backgroundcolor='w',fontsize=6)
            h.append(tt)
            return h

        count = 1
        h = []
        while any(time>n*tff):
            ind = np.argmin(np.abs(time - n * tff))
            h.append(plotTff(ax,ntot[ind],T[ind],n))
            n = n + ncrement
            count = count + 1

        if n * tff > time[-1] and (n - 1) * tff < time[-1]:
            return

        # ncrement skipped over largest
        n = np.floor(time[-1] / tff)
        ind = np.argmin(np.abs(time - n * tff))
        h.append(plotTff(ax,ntot(ind),T(ind),n))

        return h

    def plot(self):
        ax,_ = self.plotTrajectory(initialize=True)
        ax.set_xlim(auto=False)
        ax.set_ylim(auto=False)
        self.plotTsound(ax=ax)
        self.plotFFTimes(ax=ax)
        self.plotTempThresholds(ax=ax)
        self.plotDensityThresholds(ax=ax)
        self.plotOpacityLimits(ax=ax)

        mjeans,mjind = self.findMJeans()
        n = self.data.ntot[mjind]
        T = self.data.Tgas[mjind]
        ax.plot(n,T,marker='o')
        ax.text(2*n,T,f'$M_{{min}}$={signif(mjeans/self.opts.M_sun,2):g} $M_{{\odot}}$',fontsize=6)

        return ax

    def plotNumberDensities(self,*,ax=None,particle_type='Q'):
        initialize = False
        if ax is None:
            plot = plt.figure()
            ax = plot.add_subplot()
            ax.set_xlabel('n$_{tot}$ (cm$^{-3}$)')
            ax.set_ylabel('$n_?/n_{tot}$')
            ax.grid(which='both',alpha=0.3,axis='both')
            initialize = True
        if particle_type!='Q' and particle_type:
            raise RunDataException(f'{particle_type=} can only be "Q" or empty')
        specs = [f'{particle_type}{s}' for s in ['H','E','H2','Hm','H2p','Hp','H3p']]
        x = self.data.ntot
        h = ax.loglog(x,self.data[specs],linestyle='dashdot',linewidth=2,label=specs);
        ax.legend(ncols=2)
        #ylim([1e-11 5]);
        #xlim(xl);
        return (ax,h if initialize else h)


