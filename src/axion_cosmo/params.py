"""
params
"""

def params(x):
    """
    Expects
     x = {'fa':..., eta_PQ' : ...}
    or something like
     x = {'fa':..., 'T' : ..}
    and will spit back dictionary containing everything you would ever want. 
    """
    
    to_return = {}
    if 'eta_PQ' in x.keys():
        x['T'] = (45/(4*np.pi**3*81))**(1/4)*np.sqrt(x['fa'])/x['eta_PQ']
    elif 'T' in x.keys():
        x['eta_PQ'] = (45/(4*np.pi**3*81))**(1/4)*np.sqrt(x['fa'])/x['T']
    else:
        raise NotImplemented
    
    x['H'] = np.sqrt(4*np.pi**3/45)*(81**(1/2))*x['T']**2
    x['t'] = np.sqrt(45/(16*np.pi**3))*(81**(-1/2))
    return x
