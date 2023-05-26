from dataclasses import dataclass,KW_ONLY,field
from numpy import pi


class VirialException(Exception):
    pass


@dataclass
class Virial:
    # Constants
    G:float = field(init=False,default=6.6743e-8)  # cm**3/(g s**2)
    KB_CGS:float = field(init=False,default=1.38065e-16)  # ergs/K
    KB_EV:float = field(init=False,default=8.61733e-5)  # eV/K
    C_CGS:float = field(init=False,default=29979245800)  # cm/s
    MSOL:float = field(init=False,default=1.988e33)  #g
    GRAM_TO_GEV:float = field(init=False,default=5.60969e23)
    SPHERICAL_PROFILE:float = field(init=False,default=3/5)

    _:KW_ONLY
    # Internal vars
    # Default cosmology
    OmegaM:float = None
    H0_base:float = field(init=False)  # in (km/s)/Mpc
    H0:float = field(init=False)  # 1/s
    OmegaB:float = None
    OmegaR:float = None
    OmegaDM:float = field(init=False)
    Omega:float = field(init=False)
    # Density profile term - default: spherical (3/5)
    dens_prof:float = None
    h:float = 0.677
    OmegaL:float = 0.6894
    epsilon:float = 1
    f:float = None
    delta:float = 178
    # Default chemistry

    # Public vars
    mu:float = 1
    gamma:float = 5 / 3
    M_cgs:float = None  # mass of heavy particle in g
    M_gev:float = None  # mass of heavy particle in GeV
    z:float = None  # redshift
    rho:float = None  # Density

    def __post_init__(self):

        self.H0_base = 100 * self.h  # in (km/s)/Mpc
        self.H0 = self.H0_base * 3.24078e-20  # 1/s

        if self.OmegaM is None:
            self.OmegaM = 0.1424 / self.h**2
        if self.OmegaB is None:
            self.OmegaB = 0.022447 / self.h**2
        if self.OmegaR is None:
            self.OmegaR = 2.488e-5 / self.h**2

        self.OmegaDM = self.OmegaM - self.OmegaB
        self.Omega = self.OmegaM

        if self.f is None:
            self.f = self.epsilon * self.OmegaDM / self.OmegaM

        if self.dens_prof is None:
            self.dens_prof = Virial.SPHERICAL_PROFILE

        assert 0<=self.f and self.f<=1,f'{self.f=} needs to be between 0 and 1!'

        if self.M_cgs is None and self.M_gev is not None:
            self.M_cgs = self.M_gev / self.GRAM_TO_GEV
        elif self.M_cgs is not None and self.M_gev is None:
            self.M_gev = self.M_cgs * self.GRAM_TO_GEV
        elif self.M_cgs is not None and self.M_gev is not None:
            assert (abs(self.M_cgs * self.GRAM_TO_GEV - self.M_gev) < 1e-6)
        else:
            raise VirialException('Must set either M_cgs or M_gev')

        if self.z is None:
            raise VirialException("Must set z!")

        if self.rho is None:
            self.setRho()

    def setRho(self,rho=None):
        if rho is None:
            self.rho = self.overdensity()
        else:
            self.rho = rho

    def Tv(self,Mv):
        return Virial.getTv(rho=self.rho,Mv=Mv,mh=self.M_cgs * self.mu,dens_prof=self.dens_prof)

    def Tg(self,Mv,gamma=None):
        if gamma is None:
            gamma = self.gamma
        return Virial.getTg(gamma=gamma,rho=self.rho,Mv=Mv,mh=self.M_cgs * self.mu,dens_prof=self.dens_prof)

    def Mv(self,Tv):
        return Virial.getMv(Tv=Tv,rho=self.rho,mh=self.M_cgs * self.mu,dens_prof=self.dens_prof)

    def Mg(self,Tv):
        return self.Mv(Tv) * self.f

    def critical_density(self):
        return (3 / (8 * pi * self.G)) * (self.H0)**2

    def overdensity(self):
        return (1 + self.delta) * (self.Omega) * (1 + self.z)**3 * self.critical_density()

    @staticmethod
    def getTv(*,rho,Mv,mh,dens_prof=None):
        if dens_prof is None:
            dens_prof = Virial.SPHERICAL_PROFILE
        Mv = Mv * Virial.MSOL
        Tv = (4 / 3 * pi * rho)**(1 / 3) * 1 / 2 * dens_prof * mh * Virial.G * Mv**(2 / 3) / Virial.KB_CGS
        return Tv

    @staticmethod
    def getTg(gamma,**kwargs):
        Tv = Virial.getTv(**kwargs)
        Tg = Virial.convertTgTv(Tv,gamma,False)
        return Tg

    @staticmethod
    def convertTgTv(Ti,gamma,forward):
        if forward:
            # Tg -> Tv
            To = Ti / (gamma - 1)
        else:
            To = (gamma - 1) * Ti
        return To

    @staticmethod
    def convertMgMv(Mi,f,forward):
        if forward:
            Mo = Mi / f
        else:
            Mo = Mi * f
        return Mo

    @staticmethod
    def getMv(*,rho,Tv,mh,dens_prof=None):
        if dens_prof is None:
            dens_prof = Virial.SPHERICAL_PROFILE
        Mv = (2 * Virial.KB_CGS * Tv / (dens_prof * mh * Virial.G))**(3 / 2) * (4 / 3 * pi * rho)**(-1 / 2)
        Mv = Mv / Virial.MSOL
        return Mv