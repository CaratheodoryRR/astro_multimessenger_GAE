import numpy as np

from scipy.stats import (levy_stable, uniform, norm)

from ..utils.general import rescale_from_normal
from .objective_functions import rcutRange, alphaRange

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

def cuckoo_search(f, nHosts, pa, ranges, scale=1, maxIter=10**3, **kwargs):
    drop = int(round(nHosts*pa))
    r = gen_population(nHosts, *ranges)
    print('Generating initial population...')
    pts = np.column_stack((r, f(r)))
    
    
    for i in range(maxIter):
        print('\n\n\nGENERATION {} OF {}'.format(i+1, maxIter))
        rNew = levy_advance(r0=pts[:,:-1], scale=scale, **kwargs)
        rNew = periodic_bc(rNew, *ranges)
        print('\n\nLevy flight...')
        ptsNew = np.column_stack((rNew, f(rNew)))
        
        np.random.shuffle(ptsNew)
        
        pts = np.where(np.expand_dims(ptsNew[:,-1], 1) < np.expand_dims(pts[:,-1], 1), ptsNew, pts)
        pts = pts[pts[:,-1].argsort()]          
        
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
    pts[:,:-3] = pts[:,:-3]/np.sum(pts[:,:-3], axis=1).reshape(-1,1)
    pts[:,-3] = rescale_from_normal(interval=rcutRange, value=pts[:,-3])
    pts[:,-2] = rescale_from_normal(interval=alphaRange, value=pts[:,-2])
    return pts[:-drop]
        