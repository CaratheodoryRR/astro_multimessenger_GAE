
import os
import sys
import yaml
import numpy as np

from pathlib import Path
from crpropa import massNumber
from .. import auger_data_he as pao


class hidden_prints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_dict_from_yaml(pathToYAML):
    with open(str(pathToYAML), 'r') as f:
        yamlContent = f.read()
    
    return yaml.safe_load(yamlContent)

def lnA_stats(A, E, bins):
    lnA = np.log(A)
    E = np.asarray(E)
    
    container = []
    l = len(bins) - 1
    for i in range(l):
        inBin = (bins[i] <= E) & (E <= bins[i+1])
        lnAi = lnA[inBin]
        element = [lnAi.mean(), lnAi.var()] if (lnAi.size > 0) else [np.nan, np.nan]
        container.append(element)
    
    return np.array(container)

def event_counter(data, bins):
    
    energies = 18. + np.log10(data['E'])
    A = np.array([massNumber(id) for id in data['ID'].astype(int)])
    counts = np.histogram(energies[A >= 1], bins = bins)[0]
    
    return counts, energies, A

def events_from_files(fileNames, bins = pao.ebins):
    
    lnAContainer = []
    counts = np.zeros(len(bins)-1)
    for fName in fileNames:
        try:
            data = np.genfromtxt(fName, names=True, usecols=('E', 'ID'))
        except:
            continue
        cnt, engs, massn = event_counter(np.atleast_1d(data), bins)
        counts += cnt
        
        lnAContainer.append(lnA_stats(massn, engs, bins))
    
    lnA = np.array(lnAContainer)
    
    lnAStats = np.nanmean(lnA, axis=0)
    lnAErrors = np.sqrt(np.nanvar(lnA, axis=0))
    
    return counts, lnAStats, lnAErrors

def print_args(args):
    argDict = vars(args)
    
    title = "Argument values ({} items)".format(len(argDict))
    print("{0}{1}{0}{2}{0}".format("\n", title.rjust(40), ("="*len(title)).rjust(40)))
    
    for arg, val in argDict.items():
        print("{}:    {}".format(arg.rjust(25), val.resolve() if (isinstance(val, Path) and val.exists()) else val))
    print("\n")
    
def rescale_from_normal(interval, value):
    a, b = interval
    iSize = b - a
    
    return a + value*iSize