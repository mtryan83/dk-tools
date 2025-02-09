import subprocess
import logging
import os
import re
import collections
from pathlib import Path
from multiprocessing import Pool
from itertools import product,repeat
import numpy as np
from dataclasses import replace

from dk_tools.runopts import RunOpts
from dk_tools.virial import Virial
from dk_tools.rundata import RunData


def doSingleRun(opts:RunOpts,Tvar,Ttype,z,epsilon):
    return DarkTestRunner.runBatchSingle(opts=opts,Tvar=Tvar,Ttype=Ttype,z=z,epsilon=epsilon)


class DarkTestRunnerException(Exception):
    pass


class DarkTestRunner:
    DKPATH = ''
    opts:RunOpts = None

    def __init__(self,*,dkpath=None,**kwargs):
        if dkpath is not None:
            DarkTestRunner.DKPATH = dkpath
        self.opts = RunOpts(skipchecks=True,**kwargs)

    def run(self,*,rE=None,rP=None,rA=None,xi=None,epsilon=None,
            T=None,Mhalo=None,Mgas=None,z=None,
            chunksize=1,
            opts:RunOpts = None,**kwargs):
        if opts is not None:
            return DarkTestRunner.runTest(opts,full=True)
        opts = replace(self.opts,skipchecks=True,**kwargs)

        def processDark(rE,rP,rA,xi,epsilon):
            if rE is None:
                rE = 1
            if not isinstance(rE,(collections.abc.Sequence,np.ndarray)):
                rE = np.array([rE])
            if rP is None:
                rP = 1
            if not isinstance(rP,(collections.abc.Sequence,np.ndarray)):
                rP = np.array([rP])
            if rA is None:
                rA = 1
            if not isinstance(rA,(collections.abc.Sequence,np.ndarray)):
                rA = np.array([rA])
            if xi is None:
                xi = 0.01
            if not isinstance(xi,(collections.abc.Sequence,np.ndarray)):
                xi = np.array([xi])
            if epsilon is None:
                epsilon = 1
            if not isinstance(epsilon,(collections.abc.Sequence,np.ndarray)):
                epsilon = np.array([epsilon])
            return rE,rP,rA,xi,epsilon
        rE,rP,rA,xi,epsilon = processDark(rE,rP,rA,xi,epsilon)
        if z is None:
            z = 15
        if not isinstance(z,(collections.abc.Sequence,np.ndarray)):
            z = np.array([z])

        def set_tempvar(T,Mhalo,Mgas):
            if (T is None and Mhalo is None and Mgas is None) or \
               not ((T is not None) ^ (Mhalo is not None) ^ (Mgas is not None)):
                raise DarkTestRunnerException('Must specify only one of T, Mgas, or Mhalo')
            if T is not None:
                if not isinstance(T,(collections.abc.Sequence,np.ndarray)):
                    T=np.array([T])
                return T,0
            elif Mhalo is not None:
                if not isinstance(Mhalo,(collections.abc.Sequence,np.ndarray)):
                    Mhalo=np.array([Mhalo])
                return Mhalo,1
            else:
                if not isinstance(Mgas,(collections.abc.Sequence,np.ndarray)):
                    Mgas=np.array([Mgas])
                return Mgas,2
        Tvar,Ttype = set_tempvar(T,Mhalo,Mgas)

        res = []
        for ra,rM,rm,x in product(rA,rP,rE,xi):
            opts.resetParticleParams(rA=ra,rP=rM,rE=rm,xi=x)
            DarkTestRunner.makeTest(opts)
            numWorkers = None
            if opts.notParallel:
                numWorkers = 1
            Tze = product(Tvar,z,epsilon)
            Tze = list(map(list,zip(*Tze)))
            arglist = zip(repeat(opts),Tze[0],repeat(Ttype),Tze[1],Tze[2])
            with Pool(numWorkers) as pool:
                r = pool.starmap(
                    doSingleRun, arglist, chunksize=chunksize)
                # lambda Tvar,z,epsilon:DarkTestRunner.runBatchSingle(opts,Tvar,Ttype,z,epsilon),
                # product(Tvar,z,epsilon),chunksize=chunksize)
                res = res + list(r)
        return res

    @classmethod
    def updateReactionsFile(cls,opts):
        dkpath = cls.DKPATH
        react_file = dkpath / Path('react_dark')
        with open(react_file,'r') as inf:
            text = inf.read()
        text = re.sub(r'qe_mass\s*=\s*\d+\.\d*d\-?\+?\d+\*e_mass',
                      f'qe_mass = {f"{opts.rE:#E}".replace("E","d")}*e_mass',text)
        text = re.sub(r'qp_mass\s*=\s*\d+\.\d*d\-?\+?\d+\*p_mass',
                      f'qp_mass = {f"{opts.rP:#E}".replace("E","d")}*p_mass',text)
        text = re.sub(r'Dalpha\s*=\s*\d+\.\d*d\-?\+?\d+\*fine_structure_constant',
                      f'Dalpha = {f"{opts.rA:#E}".replace("E","d")}*fine_structure_constant',text)
        text = re.sub(r'xi\s*=\s*\d+\.\d*d\-?\+?\d+',
                      f'xi = {f"{opts.xi:#E}".replace("E","d")}',text)
        if not opts.dryrun:
            with open(react_file,'w') as outf:
                outf.write(text)

    @classmethod
    def runSysCommand(cls,cmd,**kwargs):
        curpath = Path.cwd()
        try:
            os.chdir(cls.DKPATH)
            comproc = subprocess.run(cmd,**kwargs)
        finally:
            os.chdir(curpath)
        return comproc

    @classmethod
    def makeTest(cls,opts):
        cls.updateReactionsFile(opts)

        dkcmd = ['./darkkrome','-C','-t']
        comproc = cls.runSysCommand(dkcmd,capture_output=True,text=True)
        try:
            comproc.check_returncode()
        except subprocess.CalledProcessError as cpe:
            print(comproc.stdout)
            print(f"\x1b[31m\"{comproc.stderr}\"\x1b[0m")
            raise cpe

    @classmethod
    def runBatchSingle(cls,opts,Tvar,Ttype,z,epsilon):
        topts = replace(opts,epsilon=epsilon,zred=z)
        topts.initialize()
        v = Virial(OmegaB=opts.omega_b,OmegaM=opts.omega_m,z=z,M_gev=opts.M)
        if Ttype==0:  # temperature given
            topts.T = Tvar
            topts.Mhalo = v.Mv(Tvar)
            topts.Mgas = v.Mg(Tvar)
        elif Ttype==1:  # Mhalo given
            topts.Mhalo = Tvar
            topts.T = v.Tv(Tvar)
            topts.Mgas = Virial.convertMgMv(Tvar,v.f,False)
        else:
            topts.T = v.Mv(Tvar)
            topts.Mhalo = Virial.convertMgMv(Tvar,v.f,True)
            topts.Mgas = Tvar
        return cls.runTest(topts)

    @classmethod
    def runTest(cls,opts:RunOpts,full=False):
        if full:
            cls.makeTest(opts)
        if opts.noDynDen:
            dyndenflag = '-D'
        else:
            dyndenflag = ' '
        dkcmd = ['./darkkrome','-r','--temp',f'{opts.T}','--zred',f'{opts.zred}',
                 '--epsilon',f'{opts.epsilon}',f'{dyndenflag}']
        build = Path(f'{cls.DKPATH}/build')
        opts.dkcmd = dkcmd

        comproc = cls.runSysCommand(dkcmd,capture_output=True,text=True)
        if opts.verbose>1:
            logging.info(comproc.stdout)
        if opts.verbose>0:
            if comproc.returncode:
                logging.warn(comproc.stderr)

        def checkForStoppableError(stderr):
            if ('cooling >1d.30' in stderr or 'negative temperature' in stderr
               or 'relative abundance' in stderr or 'solver exit status'):
                return False
            return True
        if checkForStoppableError(comproc.stderr):
            return None

        def createEnhancedFileName(opts):
            return Path((f'run_rE{opts.rE}_rp{opts.rP}_ra{opts.rA}_xi{opts.xi}'
                         f'_t{int(opts.T)}_z{int(opts.zred)}_e{opts.epsilon:5.3f}.run'))

        def createFileName(opts):
            return Path((f'run_t{int(opts.T)}_z{int(opts.zred)}_e{opts.epsilon:5.3f}.run'))
        basefile = createFileName(opts)
        enhancefile = createEnhancedFileName(opts)
        if opts.verbose>0:
            print(f'Saving as file set {enhancefile}')
        # Success - move files

        def renameToEnhance(old,new):
            old = Path(old)
            new = Path(new)
            for suf in ['dat','cool','heat','react']:
                os.rename(old.with_suffix(f'.{suf}'),new.with_suffix(f'.{suf}'))
        renameToEnhance(build / basefile,build / enhancefile)
        basefile = enhancefile
        if comproc.returncode:
            with (build / basefile.with_suffix(".err",)).open('w') as out:
                out.write(comproc.stderr)
        opts.saveOpts(build / basefile.with_suffix(".params"))
        rd = RunData(build / basefile,silent=True)
        os.rename(build / basefile.with_suffix('.zip'),opts.outdir / basefile.with_suffix(".zip"))
        return rd.fname
