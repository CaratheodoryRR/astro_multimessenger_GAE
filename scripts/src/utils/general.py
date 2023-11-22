
import numpy as np
import yaml

from .. import auger_data_he as pao
from pathlib import Path

def get_dict_from_yaml(pathToYAML):
    with open(str(pathToYAML), 'r') as f:
        yamlContent = f.read()
    
    return yaml.safe_load(yamlContent)

def event_counter_by_energy(filename, bins = pao.ebins_, col = 2):
    
    data = np.genfromtxt(filename, usecols = col)
    data = 18. + np.log10(np.copy(data))
    counts = np.histogram(data, bins = bins)[0]
    
    return counts

def print_args(args):
    argDict = vars(args)
    
    title = "Argument values ({} items)".format(len(argDict))
    print("{0}{1}{0}{2}{0}".format("\n", title.rjust(40), ("="*len(title)).rjust(40)))
    
    for arg, val in argDict.items():
        print("{}:    {}".format(arg.rjust(25), val.resolve() if (isinstance(val, Path) and val.exists()) else val))
    print("\n")