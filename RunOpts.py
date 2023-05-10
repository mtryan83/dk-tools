from dataclasses import dataclass,KW_ONLY,field
import pickle


@dataclass
class RunOpts:
    # Constants and functions
    g_to_KeV = 1 / 1.782661907e-30;
    KeV_to_K = 11604.505 * 1e3;
    k = 1.381e-16;  # cm^2*g/(s^2*K)
    G = 6.6743e-8;  # cm^3/(g*s^2)
    m_p = 1.67262158e-24;
    m_e = 9.10938188e-28;
    M_sun = 1.9891e33;  # g
    xi = 0.01;
    alphaN=7.2973525664e-3;
    delta = 178;
    h=0.6770;
    Mpcincm = 3.085678e24;
    omega_m = 0.1424 / h**2;
    Myrinsec = 60 * 60 * 24 * 365 * 1e6;
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
    Mjeans = lambda T,n,m_gev: float('nan');
    Tjeans = lambda Mj,n,m_gev: float('nan');
    _ = KW_ONLY
    Dalpha:float = None
    dp_mass:float = None
    de_mass:float = None
    rA:float = None
    rP:float = None
    rE:float = None
    omega_b:float = 0.022447 / h**2
    epsilon:float = 1
    xi:float = 0.4

    de_massG:float = field(init=False)
    dp_massG:float = field(init=False)

    krome_redshift:float = 10
    start_density:float = 10
    end_density:float = 1e10

    T:float = None
    Mhalo:float = None
    Mgas: float = None

    verbose:int = 0
    dryrun:bool = False
    notParallel:bool = False
    force:bool = False
    overwrite:bool = False
    noDynDen:bool = False

    def __post_init__(self):
        if self.Dalpha is None and self.rA is None:
            raise Exception("Must specify either Dalpha or rA")
        elif self.Dalpha is None:
            self.Dalpha = self.rA * self.alphaN
        elif self.rA is None:
            self.rA = self.Dalpha / self.alphaN
        else:
            assert self.Dalpha == self.rA * self.alphaN, f"Dalpha and rA mismatch"
        if self.de_mass is None and self.rE is None:
            raise Exception("Must specify either de_mass or rE")
        elif self.de_mass is None:
            self.de_mass = self.rE * self.m_e
        elif self.rE is None:
            self.rE = self.de_mass / self.m_e
        else:
            assert self.de_mass == self.rE * self.m_e, f"de_mass and rE mismatch"
        if self.dp_mass is None and self.rP is None:
            raise Exception("Must specify either dp_mass or rP")
        elif self.dp_mass is None:
            self.dp_mass = self.rP * self.m_p
        elif self.rP is None:
            self.rP = self.dp_mass / self.m_p
        else:
            assert self.dp_mass == self.rP * self.m_p, f"dp_mass and rP mismatch"
        self.de_massG = self.de_mass * self.g_to_KeV / 1e6
        self.dp_massG = self.dp_mass * self.g_to_KeV / 1e6
        assert self.xi<=0.4, f"{self.xi=} is too high, only xi<0.4 is allowed"
        assert self.epsilon<=1, "epsilon must be less than 1"

    def saveOpts(self,filename:str):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadOpts(filename:str):
        with open(filename,'rb') as inp:
            ro = pickle.load(inp)
            return ro