from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.UHECRs_sim_f import A_Z
from complete_propagation_Dolag_JF12 import run as run3D
from src.optimization.cuckoo_search import from_raw_to_real
from src.utils.file_utils import check_dir, del_by_extension
from src.optimization.objective_functions import rcutRange, alphaRange
from src.loaders.fields import setting_dolag_field, setting_jf12_field

# Which stage to simulate
prop = 'both'

root = Path('/home/caratheodory/development/results/cuckoo_search/HE_energyLim_massDensityGrid_3D/')
checkpoints = list(root.glob('**/*.checkpoint'))

bfDir = root.joinpath('best_combinations/')
check_dir(bfDir)

bfRanking = root.joinpath('bf_ranking.csv')
if bfRanking.is_file():
    ranking = pd.read_csv(bfRanking)
else:
    arr = np.genfromtxt(checkpoints[0])

    for cp in checkpoints[1:]:
        arr = np.row_stack((arr, np.genfromtxt(cp)))

    ranking = pd.DataFrame(from_raw_to_real(arr), columns=['H', 'He', 'N', 'Si', 'Fe', 'Rcut', 'alpha', 'chi2'])
    ranking.drop_duplicates(inplace=True)
    ranking.sort_values(by='chi2', inplace=True)
    ranking.reset_index(drop=True, inplace=True)

    ranking.to_csv(bfRanking, index=False)

chi2Thres = 51
ranking = ranking[ranking['chi2'] <= chi2Thres]

astroRoot = Path('/home/caratheodory/development/astro_multimessenger_GAE')

# Setting the magnetic fields
if prop != 'galactic':
    print('Setting up Dolag Extragalactic Magnetic Field...')
    dolag = setting_dolag_field(pathToDolag=astroRoot.joinpath('data/dolag_B_54-186Mpc_440b.raw'),
                                bFactor=5e-5)
    print('Done!\n')
else:
    dolag = None

if prop != 'extra-galactic':
    print('Setting up JF12 Galactic Magnetic Field...')
    jf12 = setting_jf12_field()
    print('Done!\n')
else:
    jf12 = None

numCR = 10**5
for i in tqdm(ranking.index):
    #if i <= 1: continue
    print('\n\n')
    fracDict = {nucleus:ranking.loc[i, nucleus] for nucleus in A_Z}
    rcut, alpha = ranking.loc[i, ['Rcut', 'alpha']]

    outDir = bfDir.joinpath('bf{}_100M'.format(i+1))
    check_dir(outDir)

    run3D(
        JF12_field=jf12,
        Dolag_field=dolag,
        srcPath=astroRoot.joinpath('data/dolag_mass_54-186Mpc_440bins.raw'),
        #srcPath='random',
        outDir=outDir,
        yamlFile=fracDict,
        minEnergy=19.,
        kappa=1e7,
        num=numCR,
        parts=numCR//10**3,
        rcut=rcut,
        alpha=alpha,
        prop=prop
        )
    print('\n\n')

del_by_extension(astroRoot, exts=('.pyc',), recursive=True)