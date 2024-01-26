import numpy as np

from tqdm import tqdm
from pathlib import Path

from ..UHECRs_sim_f import A_Z
from ..utils.error_functions import err_parameter_handler
from ..utils.general import (events_from_files, 
                             rescale_from_normal,
                             hidden_prints)

rcutRange = [19., 21.]
alphaRange = [1., 3.]
orderedNuclei = sorted(A_Z.keys(), key=lambda z: A_Z[z])

def chi2_obj_func(runPropFunc, sample, pattern, **kwargs):
    chi2Container = []
    parts = kwargs.get('parts') if (kwargs.get('parts') is not None) else 1
    for fracs, (rcut, alpha) in tqdm(zip(sample[:,:-2], sample[:,-2:])):
        nucleiDict = dict(zip(orderedNuclei, fracs))
        with hidden_prints():
            runPropFunc(yamlFile=nucleiDict,
                        rcut=rescale_from_normal(interval=rcutRange, value=rcut),
                        alpha=rescale_from_normal(interval=alphaRange, value=alpha),
                        **kwargs)
        
        
        print parts*'\033[A'+'\r',
        eventFiles = Path(kwargs.get('outDir')).glob(pattern)
        simN, lnAStats = events_from_files(fileNames=eventFiles)
        chi2 = err_parameter_handler(errorType='chi2', simN=simN)
        chi2Container.append(chi2)
    print parts*'\n'
    return np.array(chi2Container)