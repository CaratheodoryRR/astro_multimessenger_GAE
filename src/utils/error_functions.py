import numpy as np

from .. import auger_data_he as pao

def err_parameter_handler(errorType, simN, paoN=pao.auger):
    if simN.sum()==0: return np.inf

    errorType = errorType.lower()

    if errorType == 'sqerr':
        return squared_error(simN, paoN)

    if errorType == 'sqlgerr':
        return squared_log_error(simN, paoN)

    if errorType == 'sqrlerr':
        return squared_relative_error(simN, paoN)

    if errorType == 'chi2':
        return chi2_error(simN, paoN)

    if errorType == 'g-test':
        nonZero = (paoN != 0)
        return g_test_error(simN[nonZero], paoN[nonZero])

def squared_error(simN, paoN):
    diff = paoN - simN
    totalError = np.sqrt( np.sum(diff**2) )

    return totalError

def squared_log_error(simN, paoN):
    log_diff = 0.
    for s, p in zip(simN, paoN):
        if p+s != 0.:
            log_diff += ( np.log10(p/s)**2 if p*s != 0. else 4.*((p-s)/(p+s))**2 )

    log_err_tot = np.sqrt( log_diff )

    return log_err_tot

def squared_relative_error(simN, paoN):
    diff_sum = 0.
    for s, p in zip(simN, paoN):
        if p+s != 0.:
            diff_sum += ( (1. - s/p)**2 if p != 0. else 4.*((p-s)/(p+s))**2 )

    err_tot = np.sqrt( diff_sum )

    return err_tot

def chi2_error(simN, paoN):
    simTotal = simN.sum()
    paoTotal = paoN.sum()

    if simTotal == 0: return np.inf

    num = (paoN/paoTotal - simN/simTotal)**2
    den = paoN/paoTotal**2 + simN/simTotal**2

    binOk = (den > 0)

    return (num[binOk]/den[binOk]).sum()

def g_test_error(simN, paoN):
    diff = paoN - simN
    logFactor = np.where(simN > 0, simN*np.log(simN/paoN), 0.)

    return 2*(diff + logFactor).sum()