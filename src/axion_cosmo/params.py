"""Calculates cosmological parameters given some input

Typical use-cases are instantiating a dictionary containing some 'time-like' variable (for example eta_PQ or eta_QCD)
and passing the dictionary to update_params, which will generate parameters such as the temperature, hubble parameter,
physical time, and values at the critical points (PQ-transition and QCD-transition). 
"""
import numpy as np

def update_params(x):
    """Calculates cosmological parameters given some input

    Args:
      x: A dictionary containing input parameters. Required keys include
               'fa' : The axion decay constant in planck units
            'gstar' : The (constant) number of relativistic degrees of freedom
         and optionally
           'eta_PQ' : The conformal time, fixed to eta = 1 at H = fa
          'eta_QCD' : The conformal time, fixed to eta = 1 at H = ma(T) (temperature dependent axion mass)
           'Lambda' : Lambda_QCD in planck units.

    Returns:
      Dictionary containing cosmological parameters. Subscripts of _PQ or _QCD refer to
      either the PQ-era simulations or early-QCD era simulations.

    """

    #Given fa scale, find all the fixed PQ scale stuff
    x['T1_PQ']   = np.sqrt(45*x['fa']/(4*np.pi**3))*(x['gstar']**(-1/4))
    x['Tc_PQ']   = np.sqrt(3)*x['fa']
    x['c']       = x['T1_PQ']**2/(3*x['fa']**2) #convenience parameter for PQ evolution
    x['etac_PQ'] = (5/(4*(np.pi**3)*81))**(1/4)/(x['fa']**(1/2))

    #Given fa scale and Lambda_QCD, find all the QCD scale stuff. I'm taking LambdaQCD = 400 MeV
    if 'Lambda' in x:
        print(r"WARNING: Asserting all QCD scales scale linearly with $\Lambda_\mathrm{QCD}$")
        # mu = 2.3 md = 4.8
        # mpi/LQCD = 135/400
        # fpi/LQCD = 130/400
        x['ma']     =  (x['Lambda']**2)*((135/400)*(130/400)/x['fa'])*np.sqrt((2.3*4.8)/((2.3+4.8)**2))
        x['T1_QCD'] =  x['Lambda']*((45/(4*np.pi**3*x['gstar']))*(x['fa']**(-2))*x['alpha'])**(1/(4+x['n']))
        x['t1_QCD'] =  np.sqrt(45/(16*np.pi**3))*(x['gstar']**(-1/2))*(x['T1_QCD']**(-2))
        x['H1_QCD'] =  np.sqrt(4*np.pi**3/45)*(x['gstar']**(1/2))*x['T1_QCD']**2
        x['Tc_QCD'] =  x['Lambda']*((x['alpha']*x['Lambda']**4)/(x['fa']**2*x['ma']**2))**(1/x['n'])
        x['tc_QCD'] =  np.sqrt(45/(16*np.pi**3))*(x['gstar']**(-1/2))*(x['Tc_QCD']**(-2))
        x['etac_QCD'] = (x['tc_QCD']/x['t1_QCD'])**(1/2)
        x['eta1_QCD_PQ'] = (45/(4*np.pi**3*x['gstar']))**(1/4)*x['fa']**(1/2)*x['T1_QCD']**(-1) #eta1_QCD in PQ units
        x['lambda_tilde'] = (x['fa']/x['H1_QCD'])**2
        print("CHECK: Normal hierarchy T1_PQ > Tc_PQ > T1_QCD > Tc_QCD:")
        print("CHECK",x['T1_PQ'],x['Tc_PQ'],x['T1_QCD'],x['Tc_QCD'])
        print(f"CHECK: PQ-oscillations start at eta_PQ = 1, and PQ breaking happens at eta_PQ = {x['etac_PQ']}")
        print(f"CHECK: Then we need to simulate until around eta_PQ = {x['eta1_QCD_PQ']} (eta_QCD = 1) which is")
        print(f"CHECK: where the QCD mass osciollations start. Then, the field oscillates until")
        print(f"CHECK: eta_QCD = {x['etac_QCD']} at which point the mass has attained")

    #Given a reference eta, or temperature, find all the other things at the reference time
    if 'eta_PQ' in x.keys():
        x['T'] = (45/(4*np.pi**3*81))**(1/4)*np.sqrt(x['fa'])/x['eta_PQ']
    elif 'eta_QCD' in x.keys():
        x['t'] = (x['eta_QCD']**2)*(x['t1_QCD'])
        x['T'] = (4*np.pi/(16*np.pi**3*x['gstar']))**(1/4)*(x['t']**(-1/2))
    elif 'T' in x.keys():
        x['eta_PQ'] = (45/(4*np.pi**3*81))**(1/4)*np.sqrt(x['fa'])/x['T']
        if 'Lambda' in x:
            #x['eta_QCD'] =
            raise NotImplemented

    x['H'] = np.sqrt(4*np.pi**3/45)*(x['gstar']**(1/2))*x['T']**2
    if 't' not in x.keys():
        x['t'] = np.sqrt(45/(16*np.pi**3))*(x['gstar']**(-1/2))
    else:
        print("CHECK: spotcheck (should be zero)",np.sqrt(45/(16*np.pi**3))*(x['gstar']**(-1/2)) - x['t'])
    return x
