import os
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "120" # Any number

import numpy as np

import src.auger_data_he as pao
import src.optimization.objective_functions as objf
from src.UHECRs_sim_f import A_Z
from one_dimensional_propagation import run as run1D
from complete_propagation_Dolag_JF12 import run as run3D
from src.utils.file_utils import (check_dir, del_by_extension)
from src.optimization.cuckoo_search import (cuckoo_search, find_ordered_checkpoints)
from src.loaders.fields import (setting_dolag_field, setting_jf12_field, setting_uf23_field)

numOfParams = len(A_Z)+2
simType = '3D'
chi2Thres = 50
##############################################################################################################
#                                   PROPAGATION SIMULATION VARIABLES
##############################################################################################################

root = '/home/caratheodory/development/astro_multimessenger_GAE/'
# sourcesFile = Path(root).joinpath('data/dolag_mass_54-186Mpc_440bins.raw')
sourcesFile = Path(root).joinpath('data/all_sources_updated.txt')
outDir = Path(f'../../results/cuckoo_search/Regular-UF23_Random-Planck/pointSources_HE_energyLim_{simType}')
numThousands = 10**4
noInteractions = False
minEnergy = 19.0
idx = np.abs(pao.ebins - minEnergy).argmin()
parts = numThousands//10**3
check_dir(outDir)
kw3d = {}
if simType=='3D':
    kw3d['JF12_field'] = setting_uf23_field() #setting_jf12_field()
    kw3d['Dolag_field'] = setting_dolag_field(pathToDolag=Path(root).joinpath('data/dolag_B_54-186Mpc_440b.raw'),
                                                bFactor=3.617e-5) # 80 Mpc/h
    kw3d['kappa'] = 1e7

obj_func = lambda r: objf.chi2_obj_func(sample=r,
                                        runPropFunc=run1D if simType=='1D' else run3D,
                                        pattern='**/*prop*.dat' if simType=='1D' else '**/*events_JF12*.txt',
                                        chi2Thres=chi2Thres,
                                        Coords='cartesian',
                                        bins=pao.ebins[idx:],
                                        srcPath=sourcesFile,
                                        outDir=outDir.joinpath('tmp'),
                                        num=numThousands,
                                        noInteractions=noInteractions,
                                        barProgress=True,
                                        parts=parts,
                                        minEnergy=minEnergy,
                                        rigLim=False,
                                        **kw3d)

##############################################################################################################
#                                   CUCKOO SEARCH VARIABLES
##############################################################################################################
nHosts = 10
pa = 0.5
ranges = np.repeat(a=[[0,1]],
                    repeats=numOfParams,
                    axis=0)
maxIter = 1
checkpointDir = outDir.joinpath('checkpoints')
check_dir(checkpointDir)
# orderedCheckpointFiles = find_ordered_checkpoints(checkpointDir)
# checkpoint = orderedCheckpointFiles[-1] if (len(orderedCheckpointFiles) > 0) else None

bfResult = cuckoo_search(
                        f=obj_func,
                        nHosts=nHosts,
                        pa=pa,
                        ranges=ranges,
                        maxIter=maxIter,
                        checkpointDir=checkpointDir,
                        loadCheckpoint=checkpointDir.joinpath('generation_state.pkl'),
                        alpha=1.25,
                        beta=0
                        )

header = (27*' ').join(objf.orderedNuclei+['Rcut','alpha','chi2'])
np.savetxt(
            fname=outDir.joinpath('best_fit_parameters.txt'),
            X=bfResult,
            header=header,
            delimiter='    '
            )

del_by_extension(parentDir=Path(''), exts=('.pyc',), recursive=True)