import re
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import (levy_stable, uniform, norm)

from ..utils.general import rescale_from_normal
from .objective_functions import (rcutRange, alphaRange)

regChkPattern = re.compile(r'\d+')
extract_checkpoint_generation = lambda checkpointFileName: int(regChkPattern.search(checkpointFileName).group())

class checkpoint_state:
    checkpointPickled = 'generation_state.pkl'

    def __init__(self, checkpointDir, igen=0, data=None):
        self.igen = igen
        self._data = data
        self.__checkpointDir = Path(checkpointDir)
        self.pickleFile = self.__checkpointDir.joinpath(checkpoint_state.checkpointPickled)

    def update_gen_counter(self):
        self.igen += 1

    @property
    def data(self):
        if self._data is not None:
            return self._data.copy()

    @data.setter
    def data(self, value):
        if (not isinstance(value, np.ndarray)) and (value.ndim != 2):
            raise ValueError("Data must be a 2D numpy array!")
        self._data = value.copy()

    def save_state(self, fileType):
        if fileType.lower() == 'text':
            self._save_state_to_txt()
        elif fileType.lower() == 'pickle':
            self._pickle_dump_state()
        else:
            raise ValueError("Only 'text' and 'pickle' file types supported!")

    def load_state(self, checkpointFile):
        suffix = Path(checkpointFile).suffix.lower()
        if suffix == '.checkpoint':
            self._load_state_from_txt(textFile=checkpointFile)
        elif suffix in {'.pkl', '.pickle'}:
            self.pickleFile = Path(checkpointFile)
            self._pickle_load_state()
        else:
            raise ValueError("File must be in an appropriate format extension (.checkpoint, .pkl, or .pickle)")

    def _save_state_to_txt(self):
        numOfParams = self.data.shape[1]-1
        header = (22*' ').join([f'param{i+1}' for i in range(numOfParams)] + ['chi2'])
        np.savetxt(fname=self.__checkpointDir.joinpath(f'generation_{self.igen}.checkpoint'),
                    X=self.data,
                    header=header,
                    delimiter='    ')

    def _load_state_from_txt(self, textFile):
        if not Path(textFile).exists():
            raise FileNotFoundError("Checkpoint text file does not exist!")
        self.data = np.loadtxt(fname=textFile)
        self.igen = extract_checkpoint_generation(Path(textFile).name)

    def _pickle_dump_state(self):
        with open(self.pickleFile, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _pickle_load_state(self):
        if not self.pickleFile.exists():
            raise FileNotFoundError("Checkpoint pickle file does not exist!")
        with open(self.pickleFile, 'rb') as handle:
            loadedState = pickle.load(handle)
            self.__dict__.update(loadedState.__dict__)

def find_ordered_checkpoints(checkpointDir):
    checkpointFiles = Path(checkpointDir).glob('*.checkpoint')

    orderedCheckpointFiles = sorted(checkpointFiles, key=extract_checkpoint_generation)

    return orderedCheckpointFiles

def from_raw_to_real(arr):
    arrCopy = arr.copy()
    arrCopy[:,:-3] = arr[:,:-3]/np.sum(arr[:,:-3], axis=1).reshape(-1,1)
    arrCopy[:,-3] = rescale_from_normal(interval=rcutRange, value=arr[:,-3])
    arrCopy[:,-2] = rescale_from_normal(interval=alphaRange, value=arr[:,-2])

    return arrCopy

def periodic_bc(r, *args):
    wrapped = []
    for I, xi in zip(args, r.T):
        xmin, xmax = I
        dx = xmax - xmin

        xf = np.where( (xi>xmax) | (xi<xmin), xmin+(xi%dx), xi)
        wrapped.append(xf)

    return np.array(wrapped).T

def levy_advance(r0, scale, **kwargs):
    uVec = norm.rvs(size=r0.size).reshape(r0.shape)
    uVec /= np.linalg.norm(uVec, axis=1).reshape(-1,1)

    stepSize = scale*levy_stable.rvs(size=r0.shape[0], **kwargs).reshape(-1,1)

    return r0 + stepSize*uVec

def gen_population(size, *args):
    population = []
    for I in args:
        xi = I[0] + uniform.rvs(size=size)*(I[1]-I[0])
        population.append(xi)

    return np.array(population).T

def cuckoo_search(f, nHosts, pa, ranges, maxIter=10**3, checkpointDir=None, loadCheckpoint=None, **kwargs):
    drop = int(round(nHosts*pa))

    currentState = checkpoint_state(checkpointDir=checkpointDir)
    if loadCheckpoint is not None:
        print(f'Loading checkpoint {loadCheckpoint}')

        currentState.load_state(checkpointFile=loadCheckpoint)

        pts = currentState.data
        ptNum = len(pts)
        if ptNum > nHosts:
            pts = pts[:nHosts]
        elif ptNum < nHosts:
            r = gen_population(nHosts-ptNum, *ranges)
            print('Generating missing values...')
            pts0 = np.column_stack((r, f(r)))
            pts = np.row_stack((pts.copy(), pts0))
    else:
        r = gen_population(nHosts, *ranges)
        print('Generating initial population...')
        pts = np.column_stack((r, f(r)))

    finalGen = maxIter + currentState.igen
    N = nHosts - drop
    topNOld = pts[:N].copy()
    for i in range(maxIter):
        currentState.update_gen_counter()
        print(f'\n\n\nGENERATION {currentState.igen} OF {finalGen}')
        currentState.update_gen_counter()
        print(f'\n\n\nGENERATION {currentState.igen} OF {finalGen}')
        rNew = levy_advance(r0=pts[:,:-1], scale=1, **kwargs)
        rNew = periodic_bc(rNew, *ranges)
        print('\n\nLevy flight...')
        ptsNew = np.column_stack((rNew, f(rNew)))

        np.random.shuffle(ptsNew)

        pts = np.where(np.expand_dims(ptsNew[:,-1], 1) < np.expand_dims(pts[:,-1], 1), ptsNew, pts)
        pts = pts[pts[:,-1].argsort()]          

        if (checkpointDir is not None) and ((pts[:N]!=topNOld).any() or i==maxIter-1):
            topNOld = pts[:N].copy()
            currentState.data = pts.copy()
            currentState.save_state(fileType='text')

        rRand = gen_population(drop, *ranges)  
        print('\n\nRandom mutation...')
        ptsRand = np.column_stack((rRand, f(rRand)))

        pts[-drop:] = ptsRand

        best = pts[pts[:,-1].argmin()]
        print('\n\nCURRENT BEST: chi2: {0} \nFracs: {1}\nRcut: {2}\nalpha: {3}\n'.format(
                                                                                        best[-1],
                                                                                        best[:-3]/best[:-3].sum(),
                                                                                        rescale_from_normal(interval=rcutRange, 
                                                                                                            value=best[-3]),
                                                                                        rescale_from_normal(interval=alphaRange, 
                                                                                                            value=best[-2]))
                                                                                        )
        currentState.data = pts.copy()
        currentState.save_state(fileType='pickle')

    currentState.data = pts[pts[:,-1].argsort()]
    currentState.save_state(fileType='text')

    return from_raw_to_real(arr=pts[:-drop])