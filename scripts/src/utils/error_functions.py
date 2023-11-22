import numpy as np
from .. import auger_data_he as pao

def err_parameter_handler(type_err, N_sim, N_pao=pao.auger_/pao.auger_[0]):
    
    if type_err == 'sqerr':
        return squared_error(N_sim, N_pao)
    
    if type_err == 'sqlgerr':
        return squared_log_error(N_sim, N_pao)
    
    if type_err == 'sqrlerr':
        return squared_relative_error(N_sim, N_pao)
    

def squared_error(N_sim, N_pao):
    diff = N_pao - N_sim
    err_tot = np.sqrt( np.sum(diff**2) )
    
    return err_tot


def squared_log_error(N_sim, N_pao):
    
    log_diff = 0.
    for s, p in zip(N_sim, N_pao):
        if p+s != 0.:
            log_diff += ( np.log10(p/s)**2 if p*s != 0. else 4.*((p-s)/(p+s))**2 )
        
    log_err_tot = np.sqrt( log_diff )
    
    return log_err_tot


def squared_relative_error(N_sim, N_pao):
    
    diff_sum = 0.
    for s, p in zip(N_sim, N_pao):
        if p+s != 0.:
            diff_sum += ( (1. - s/p)**2 if p != 0. else 4.*((p-s)/(p+s))**2 )
    
    err_tot = np.sqrt( diff_sum )
    
    return err_tot