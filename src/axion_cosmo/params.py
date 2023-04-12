"""
params
"""

def R(gstar,T):
    """
    (S46) in MeV. 
    """
    return 3.699*1e-10*(gstar**(-1/3))/T

def H(gstar,T):
    """
    (S44), and https://physics.nist.gov/cgi-bin/cuu/Value?plkmc2gev
    """
    return 1.660*(gstar**(1/2))*(T**2)/1.220890e22

def t(eta,t1):
    """
    (S6)
    """
    return (eta**2)*(t1)

def t_T(gstar,T):
    """
    (S45)
    """
    return 0.3012*(gstar**(-1/2))*1.220890e22/(T**2)

def T(gstar,t):
    """
    S(45)
    """
    return (0.3012*(gstar**(-1/2))*1.220890e22/t)**(1/2)

def init_PQ_params(fa,gstar):
    """
    these ARENT where you initialize, these are at eta = 1
    """
    T1 = (fa*1.220890e22/(1.660*(gstar**(1/2))))**(1/2)
    R1 = R(gstar,T1)
    t1 = t_T(gstar,T1)
    Tc = 1.68e-7*((400))
    return R1,T1,t1

def init_earlyQCD_params():
    #T1 = (fa*1.220890e22/(1.660*(gstar**(1/2))))**(1/2)
    pass
