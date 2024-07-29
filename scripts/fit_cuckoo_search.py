import numpy as np
import src.auger_data_he as pao
import src.optimization.objective_functions as objf

from pathlib import Path
from src.UHECRs_sim_f import A_Z
from src.optimization.cuckoo_search import cuckoo_search
from src.utils.file_utils import (check_dir, del_by_extension)
from src.loaders.fields import (setting_dolag_field, setting_jf12_field)

from one_dimensional_propagation import run as run1D
from complete_propagation_Dolag_JF12 import run as run3D

numOfParams = len(A_Z)+2
simType = '3D'
##############################################################################################################
#                                   PROPAGATION SIMULATION VARIABLES
##############################################################################################################
    
root = '/home/caratheodory/development/astro_multimessenger_GAE/'
sourcesFile = Path(root).joinpath('data/EG_{}_sources.txt'.format(simType))
outDir = Path('./test_{}/'.format(simType))
numThousands = 10**3
noInteractions = False
minEnergy = 19.0
idx = np.abs(pao.ebins - minEnergy).argmin()
parts = numThousands//10**3
check_dir(outDir)
kw3d = {}
if simType=='3D':
    kw3d['JF12_field'] = setting_jf12_field()
    kw3d['Dolag_field'] = setting_dolag_field(pathToDolag=Path(root).joinpath('data/dolag_B_54-186Mpc_440b.raw'),
                                              bFactor=5e-5)
    kw3d['kappa'] = 1e7

    
obj_func = lambda r: objf.chi2_obj_func(sample=r,
                                        runPropFunc=run1D if simType=='1D' else run3D,
                                        pattern='**/*prop*.dat' if simType=='1D' else '**/*events_JF12*.txt',
                                        bins=pao.ebins[idx:],
                                        srcPath=sourcesFile,
                                        outDir=outDir,
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
maxIter = 50
checkpointDir = outDir.joinpath('checkpoints')
check_dir(checkpointDir)
bfResult = cuckoo_search(f=obj_func,
                         nHosts=nHosts,
                         pa=pa,
                         ranges=ranges,
                         maxIter=maxIter,
                         checkpointDir=checkpointDir,
                         loadCheckpoint=None,
                         alpha=1.25,
                         beta=0)

header = (27*' ').join(objf.orderedNuclei+['Rcut','alpha','chi2'])
np.savetxt(fname=outDir.joinpath('best_fit_parameters.txt'),
           X=bfResult,
           header=header,
           delimiter='    ')

del_by_extension(parentDir=Path(''), exts=('.pyc',), recursive=True)
