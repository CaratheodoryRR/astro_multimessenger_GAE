
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

def event_counter(fileName, bins):
    
    data = np.genfromtxt(fileName, names=True, usecols=('E', 'ID'))
    energies = 18. + np.log10(data['E'])
    A = np.array([massNumber(id) for id in data['ID'].astype(int)])
    counts = np.histogram(energies[A >= 1], bins = bins)[0]
    
    return counts


def events_from_files(fileNames, bins = pao.ebins):
    
    counts = np.zeros(len(bins)-1)
    for fName in fileNames:
        counts += event_counter(fileName=fName, bins=bins)
    
    return counts

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