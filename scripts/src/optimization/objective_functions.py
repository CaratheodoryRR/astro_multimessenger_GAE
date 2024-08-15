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

    for fracs, (rcut, alpha) in zip(tqdm(sample[:,:-2]), sample[:,-2:]):
        nucleiDict = dict(zip(orderedNuclei, fracs))
        
        print(f"Fracs: {dict(zip(orderedNuclei, [f'{p:.1%}' for p in fracs/fracs.sum()]))}")
        with hidden_prints():
            runPropFunc(yamlFile=nucleiDict,
                        rcut=rescale_from_normal(interval=rcutRange, value=rcut),
                        alpha=rescale_from_normal(interval=alphaRange, value=alpha),
                        **kwargs)
        
        eventFiles = Path(kwargs.get('outDir')).glob(pattern)
        simN, _, __ = events_from_files(fileNames=eventFiles, bins=bins)
        idx = np.abs(pao.ebins-bins[0]).argmin()
        print(f'Events per bin: {simN}')
        chi2 = err_parameter_handler(errorType='chi2', simN=simN, paoN=pao.auger[idx:idx+len(bins)])
        chi2Container.append(chi2)
        print(f'Associated chi2: {chi2:.2f}')

    return np.array(chi2Container)
