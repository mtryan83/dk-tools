from dataclasses import dataclass,KW_ONLY,field,fields
from configparser import ConfigParser
from pathlib import Path
import numpy as np


class RunOptsException(Exception):
    pass


@dataclass
class RunOpts:
    # Constants and functions
    g_to_KeV:float = field(init=False, default=1 / 1.782661907e-30)
    KeV_to_K:float = field(init=False, default=11604.505 * 1e3)
    k:float = field(init=False, default=1.381e-16)  # cm^2*g/(s^2*K)
    G:float = field(init=False, default=6.6743e-8)  # cm^3/(g*s^2)
    m_p:float = field(init=False, default=1.67262158e-24)
    m_e:float = field(init=False, default=9.10938188e-28)
    M_sun:float = field(init=False, default=1.9891e33)  # g
    alphaN:float = field(init=False, default=7.2973525664e-3)
    delta:float = field(init=False, default=178)
    h:float = field(init=False, default=0.6770)
    Mpcincm:float = field(init=False, default=3.085678e24)
    Myrinsec:float = field(init=False, default=60 * 60 * 24 * 365 * 1e6)
    # Definition of Mj comes from Low&Lynden-Bell 1976 (actually from
    # Rees1976?)
    #Mjeans = @(T,n,dp_mass)(pi*k/G)^(3/2)*T.^(3/2)./sqrt(n)/(1*dp_mass)^2;
    #Tjeans = @(Mj,n,dp_mass)Mj.^(2/3)*G/(pi*k).*((1*dp_mass)^4*n).^(1/3);
    # Updated to assume monoatomic gas (gamma=5/3) and mass in GeV/c^2
    # Possibly from wiki?
    #Mj = (10 k T/(3 (gamma-1) m G))^(3/2) (3/(4 pi m n))^1/2
    #Mjeans = @(T,n,m_gev)81.3*sqrt(T.^3./(m_gev.^4*n));
    #Tjeans = @(Mj,n,m_gev)(Mj/81.3*sqrt(n).*m_gev^2).^(2/3);
    # I'm not sure these are working correctly anyway. Better to error
    # here
    #Mjeans = lambda T,n,m_gev: float('nan')
    #Tjeans = lambda Mj,n,m_gev: float('nan')
    _ = KW_ONLY
    # Dark parameters
    Dalpha:float = None
    dp_mass:float = None
    de_mass:float = None
    rA:float = None
    rP:float = None
    rE:float = None
    omega_m:float = None
    omega_b:float = None
    epsilon:float = 1
    xi:float = 0.01

    # internal fields
    de_massG:float = field(init=False,default=None)
    dp_massG:float = field(init=False,default=None)
    m:float = field(init=False,default=None)
    M:float = field(init=False,default=None)

    # cosmo parameters
    krome_redshift:float = None
    zred:float = None
    start_density:float = 10
    end_density:float = 1e10

    # halo parameters
    T:float = None
    Mhalo:float = None
    Mgas: float = None

    #physics parameter
    noDynDen:bool = False

    # DK run parameters
    verbose:int = 0
    dryrun:bool = False
    notParallel:bool = False
    force:bool = False
    overwrite:bool = False
    outdir:Path = '.'
    dkcmd:str = field(init=False,default=None)

    # RunOpts flags
    skipchecks:bool = False
    initialized:bool = field(init=False,default=False)

    def __post_init__(self):
        if self.omega_m is None:
            self.omega_m = 0.1424 / self.h**2
        if self.omega_b is None:
            self.omega_b = 0.022447 / self.h**2

        if self.skipchecks:
            return

        self.checkParticleParams()

        self.krome_redshift,self.zred = RunOpts.checkMismatch(self.krome_redshift,'krome_redshift',self.zred,'zred',1,'1')
        self.initialized = True

    def initialize(self,**kwargs):
        fd = RunOpts.getInitFields()
        for k,v in self.__dict__.items():
            if fd[k] and k not in kwargs:
                kwargs[k] = v
        if 'skipchecks' in kwargs:
            kwargs['skipchecks'] = False
        self.__init__(**kwargs)

    def checkDarkCosmo(self):
        assert self.xi<=0.4, f"{self.xi=} is too high, only xi<0.4 is allowed"
        assert self.epsilon is None or self.epsilon<=1, "epsilon must be less than 1"

    def resetParticleParams(self,**kwargs):
        self.Dalpha = None
        self.de_mass = None
        self.dp_mass = None
        self.rA = None
        self.rE = None
        self.rP = None
        self.de_massG = None
        self.dp_massG = None
        self.m = None
        self.M = None
        self.xi = None
        if kwargs:
            self.setParticleParams(**kwargs)

    def setParticleParams(self,**kwargs):
        for k,v in kwargs.items():
            if k not in ['Dalpha','dp_mass','de_mass','rA','rE','rP','xi','epsilon']:
                continue
            self.__dict__[k] = v
        self.checkParticleParams()

    def checkParticleParams(self):
        self.Dalpha, self.rA = RunOpts.checkMismatch(self.Dalpha,'Dalpha',self.rA,'rA',self.alphaN,'alpha')
        self.de_mass, self.rE = RunOpts.checkMismatch(self.de_mass,'de_mass',self.rE,'rE',self.m_e,'m_e')
        self.dp_mass, self.rP = RunOpts.checkMismatch(self.dp_mass,'dp_mass',self.rP,'rP',self.m_p,'m_p')
        self.checkDarkCosmo()

        self.de_massG = self.de_mass * self.g_to_KeV / 1e6
        self.dp_massG = self.dp_mass * self.g_to_KeV / 1e6
        self.m = self.rE * 511
        self.M = self.rP * 0.938

    @staticmethod
    def getInitFields():
        fl = fields(RunOpts)
        fd = {}
        for f in fl:
            fd[f.name] = f.init
        return fd

    @staticmethod
    def checkMismatch(var1,var1n,var2,var2n,smvar,smname):
        if var1 is None and var2 is None:
            raise RunOptsException(f"Must specify either {var1n} or {var2n}")
        elif var1 is None:
            var1 = var2 * smvar
        elif var2 is None:
            var2 = var1 / smvar
        else:
            x = var1 / smvar
            y = var2
            assert np.abs(x - y) / (x + y) < 1e-4, \
                f"{var1n}/{smname}({x}) and {var2n}({y}) mismatch"
        return var1, var2

    def saveOpts(self,filename:str):
        # Saving as pickle file - NOT HUMAN READABLE
        #with open(filename, 'wb') as outp:  # Overwrites any existing file.
        #    pickle.dump(self, outp, protocol)
        # Saving as config file
        cf = ConfigParser(default_section='hidden')
        cf.optionxform = str
        cf['init'] = {}
        fd = RunOpts.getInitFields()
        for k,v in self.__dict__.items():
            if fd[k]:
                cf['init'][k] = str(v)
            else:
                cf['hidden'][k] = str(v)
        with open(filename, 'w') as configfile:
            cf.write(configfile)

    def diff(self,other):
        if not isinstance(other,RunOpts):
            raise RunOptsException("Diff can only be called on RunOpts objects")
        sd = self.__dict__
        od = other.__dict__
        diff = {}
        for k,v in sd.items():
            if k not in od:
                diff[k] = f'{v} vs [Missing]'
            elif v!=od[k]:
                diff[k] = f'{v} vs {od[k]}'
        for k,v in od.items():
            if k not in sd:
                diff[k] = f'[Missing] vs {v}'
        return diff
            
    @staticmethod
    def loadOpts(filename,*,using_file=False,using_string=False):
        # Loading as pickle file - see saveOpts
        #with open(filename,'rb') as inp:
        #    ro = pickle.load(inp)
        #    return ro
        cf = ConfigParser(default_section='hidden')
        cf.optionxform = str
        if using_file:
            cf.read_file(filename)
        elif using_string:
            cf.read_string(filename)
        else:
            cf.read(filename)
        if 'init' not in cf:
            raise FileNotFoundError(filename)
        fl = fields(RunOpts)
        fd = {}
        for f in fl:
            fd[f.name] = f.init,f.type
        init = {}
        for k in cf['init']:
            if not fd[k][0]:
                continue
            if fd[k][1]==bool:
                init[k] = cf['init'].getboolean(k)
            elif fd[k][1]==int:
                init[k] = int(cf['init'][k])
            elif fd[k][1]==float:
                init[k] = float(cf['init'][k])
            else:
                init[k] = cf['init'][k]
        ro = RunOpts(**init)
        ro.dkcmd = cf['hidden']['dkcmd']
        return ro