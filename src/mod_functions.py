import numpy as np
import auger_data_he as pao

def event_counter_by_energy(filename, bins = pao.ebins_, col = 2):
    
    data = np.genfromtxt(filename, usecols = col)
    data = 18. + np.log10(np.copy(data))
    counts = np.histogram(data, bins = bins)[0]
    
    return counts
