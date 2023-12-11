
import numpy as np
import yaml

from .. import auger_data_he as pao
from pathlib import Path

def get_dict_from_yaml(pathToYAML):
    with open(str(pathToYAML), 'r') as f:
        yamlContent = f.read()
    
    return yaml.safe_load(yamlContent)

def event_counter_by_energy(filename, bins = pao.ebins_):
    
    data = np.genfromtxt(filename, names=['E', 'ID'])
    energies = 18. + np.log10(data['E'])
    A = np.array([massNumber(id) for id in data['ID'].astype(int)])
    counts = np.histogram(energies[A >= 1], bins = bins)[0]
    
    return counts

def print_args(args):
    argDict = vars(args)
    
    title = "Argument values ({} items)".format(len(argDict))
    print("{0}{1}{0}{2}{0}".format("\n", title.rjust(40), ("="*len(title)).rjust(40)))
    
    for arg, val in argDict.items():
        print("{}:    {}".format(arg.rjust(25), val.resolve() if (isinstance(val, Path) and val.exists()) else val))
    print("\n")