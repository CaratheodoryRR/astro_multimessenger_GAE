import gc
import numpy as np

from tqdm import tqdm
from pathlib import Path

from ..UHECRs_sim_f import A_Z
from .. import auger_data_he as pao
from ..utils.error_functions import err_parameter_handler
from ..utils.general import (events_from_files, 
                             rescale_from_normal,
                             hidden_prints)

rcutRange = [19., 21.]
alphaRange = [1., 3.]
orderedNuclei = sorted(A_Z.keys(), key=lambda z: A_Z[z])

def chi2_obj_func(runPropFunc, sample, pattern, bins=pao.ebins, **kwargs):
    chi2Container = []
    parts = kwargs.get('parts') if (kwargs.get('parts') is not None) else 1
    for fracs, (rcut, alpha) in tqdm(zip(sample[:,:-2], sample[:,-2:])):
        nucleiDict = dict(zip(orderedNuclei, fracs))
        print( 'Fracs: {}'.format(dict(zip(orderedNuclei, ['%.2e' % p for p in fracs/fracs.sum()]))) )
        with hidden_prints():
            runPropFunc(yamlFile=nucleiDict,
                        rcut=rescale_from_normal(interval=rcutRange, value=rcut),
                        alpha=rescale_from_normal(interval=alphaRange, value=alpha),
                        **kwargs)
        
        
        eventFiles = Path(kwargs.get('outDir')).glob(pattern)
        simN, lnAStats, lnAErrors = events_from_files(fileNames=eventFiles, bins=bins)
        idx = np.abs(pao.ebins-bins[0]).argmin()
        print('Events per bin: {}'.format(simN))
        chi2 = err_parameter_handler(errorType='chi2', simN=simN, paoN=pao.auger[idx:idx+len(bins)])
        chi2Container.append(chi2)
        print('Associated chi2: {:.2f}'.format(chi2))
        gc.collect()
    return np.array(chi2Container)
