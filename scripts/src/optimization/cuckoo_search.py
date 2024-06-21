import numpy as np

from pathlib import Path
from scipy.stats import (levy_stable, uniform, norm)

from ..utils.general import rescale_from_normal
from .objective_functions import rcutRange, alphaRange

def save_checkpoint(checkpointArr, outDir, gen):
    numOfParams = checkpointArr.shape[1]-1
    header = (22*' ').join(['param{}'.format(i+1) for i in range(numOfParams)] + ['chi2'])
    np.savetxt(fname=outDir.joinpath('generation_{}.checkpoint'.format(gen)),
               X=checkpointArr,
               header=header,
               delimiter='    ')

def load_checkpoint(checkpointFile):
    assert Path(checkpointFile).exists, 'Checkpoint file does not exist!'
    arr = np.loadtxt(fname=checkpointFile)
    return arr

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
    
    if loadCheckpoint is not None:
        print('Loading checkpoint {}'.format(loadCheckpoint))
        pts = load_checkpoint(checkpointFile=loadCheckpoint)
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
    
    top3Old = pts[:3].copy()
    for i in range(maxIter):
        print('\n\n\nGENERATION {} OF {}'.format(i+1, maxIter))
        rNew = levy_advance(r0=pts[:,:-1], scale=1, **kwargs)
        rNew = periodic_bc(rNew, *ranges)
        print('\n\nLevy flight...')
        ptsNew = np.column_stack((rNew, f(rNew)))
        
        np.random.shuffle(ptsNew)
        
        pts = np.where(np.expand_dims(ptsNew[:,-1], 1) < np.expand_dims(pts[:,-1], 1), ptsNew, pts)
        pts = pts[pts[:,-1].argsort()]          
        
        if (checkpointDir is not None) and ((pts[:3]!=top3Old).any() or i==maxIter-1):
            top3Old = pts[:3].copy()
            save_checkpoint(checkpointArr=pts,
                            outDir=checkpointDir, 
                            gen=i+1)
        
        rRand = gen_population(drop, *ranges)  
        print('\n\nRandom mutation...')
        ptsRand = np.column_stack((rRand, f(rRand)))
        
        pts[-drop:] = ptsRand
        
        best = pts[pts[:,-1].argmin()]
        print('\n\nCURRENT BEST: chi2: {0} \nFracs: {1}\nRcut: {2}\nalpha: {3}\n'.format(best[-1],
                                                                                         best[:-3],
                                                                                         rescale_from_normal(interval=rcutRange, 
                                                                                                             value=best[-3]),
                                                                                         rescale_from_normal(interval=alphaRange, 
                                                                                                             value=best[-2])))
    
    pts = pts[pts[:,-1].argsort()]
    save_checkpoint(checkpointArr=pts, outDir=checkpointDir, gen='last')
    return from_raw_to_real(arr=pts[:-drop])
        