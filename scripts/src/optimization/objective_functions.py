import shutil
import pickle
from uuid import uuid4
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..UHECRs_sim_f import A_Z
from .. import auger_data_he as pao
from ..utils.error_functions import err_parameter_handler
from ..utils.general import (events_from_files, 
                            rescale_from_normal,
                            hidden_prints)

rcutRange = [19., 21.]
alphaRange = [1., 5.]
orderedNuclei = sorted(A_Z.keys(), key=lambda z: A_Z[z])

def save_output(src, dst, toPickle):
    shutil.copytree(src, dst, dirs_exist_ok=True)

    with open(Path(dst).joinpath('params_objF_list.pkl'), 'wb') as handle:
        pickle.dump(toPickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

def chi2_obj_func(runPropFunc, sample, pattern, chi2Thres, bins=pao.ebins, **kwargs):
    chi2Container = []

    outDir = Path(kwargs.get('outDir'))

    for fracs, (rcut, alpha) in zip(tqdm(sample[:,:-2]), sample[:,-2:]):
        nucleiDict = dict(zip(orderedNuclei, fracs))

        print(f"Fracs: {dict(zip(orderedNuclei, [f'{p:.1%}' for p in fracs/fracs.sum()]))}")
        with hidden_prints():
            runPropFunc(yamlFile=nucleiDict,
                        rcut=rescale_from_normal(interval=rcutRange, value=rcut),
                        alpha=rescale_from_normal(interval=alphaRange, value=alpha),
                        **kwargs)

        eventFiles = outDir.glob(pattern)
        simN, _, __ = events_from_files(fileNames=eventFiles, bins=bins)
        idx = np.abs(pao.ebins-bins[0]).argmin()
        print(f'Events per bin: {simN}')
        chi2 = err_parameter_handler(errorType='chi2', simN=simN, paoN=pao.auger[idx:idx+len(bins)])
        chi2Container.append(chi2)
        print(f'Associated chi2: {chi2:.2f}')

        if chi2 <= chi2Thres:
            save_output(src=outDir,
                        dst=outDir.parent.joinpath(f'saved_optimal_solutions/{uuid4()}/'),
                        toPickle=[*fracs, rcut, alpha, chi2])

    return np.array(chi2Container)
