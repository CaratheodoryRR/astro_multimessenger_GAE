import numpy as np
import auger_data_he as pao
from astropy.coordinates import SkyCoord
import astropy.units as u

def event_counter_by_energy(filename, bins = pao.ebins_, col = 2):
    
    data = np.genfromtxt(filename, usecols = col)
    data = 18. + np.log10(np.copy(data))
    counts = np.histogram(data, bins = bins)[0]
    
    return counts


def coordinate_transformation_handler(sourcePositions, coords):
    if coords == 'galactic':
        sourcePositions = from_galactic_to_cartesian(sourcePositions)
    elif coords == 'icrs':
        sourcePositions = from_icrs_to_cartesian(sourcePositions)
    elif coords == 'supergalactic':
        sourcePositions = from_supergalactic_to_cartesian(sourcePositions)
    return sourcePositions

def to_named_cartesian(cartCoords):
    cartCoords = [tuple(x) for x in cartCoords]
    
    return np.array(cartCoords, dtype=[('X', np.float64), ('Y', np.float64), ('Z', np.float64)])

def from_galactic_to_cartesian(sourcePositions):
    galCoords = SkyCoord(sourcePositions['Longitude']*u.radian, 
                         sourcePositions['Latitude']*u.radian,
                         sourcePositions['Distance']*u.Mpc,
                         frame='galactic')
    cartCoords = to_named_cartesian(galCoords.icrs.cartesian.xyz.value.T)
    
    return cartCoords
    
def from_supergalactic_to_cartesian(sourcePositions):
    sgalCoords = SkyCoord(sourcePositions['SLongitude']*u.radian, 
                          sourcePositions['SLatitude']*u.radian,
                          sourcePositions['Distance']*u.Mpc,
                          frame='supergalactic')
    cartCoords = to_named_cartesian(sgalCoords.icrs.cartesian.xyz.value.T)
    
    return cartCoords

def from_icrs_to_cartesian(sourcePositions):
    icrsCoords = SkyCoord(sourcePositions['RA']*u.radian, 
                          sourcePositions['DEC']*u.radian,
                          sourcePositions['Distance']*u.Mpc,
                          frame='icrs')
    cartCoords = to_named_cartesian(icrsCoords.cartesian.xyz.value.T)
    
    return cartCoords
