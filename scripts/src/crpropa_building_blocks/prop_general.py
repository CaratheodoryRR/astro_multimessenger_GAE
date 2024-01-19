from crpropa import *
import numpy as np

def source_energy_spectrum(alpha, rcut=np.inf):
    
    powerLaw = '(E/EeV)^{}'.format(-alpha)
    exponentialCutOff = '( (E > Z*{0}) ? exp(1 - E/(Z*{0})) : 1 )'.format(rcut)
    
    return '{}*{}'.format(powerLaw, exponentialCutOff)

def set_interactions(sim, model):
    # Simulation processes
    sim.add( PhotoPionProduction(CMB()) )
    sim.add( ElectronPairProduction(CMB()) )
    sim.add( PhotoDisintegration(CMB()) )
    sim.add( PhotoPionProduction(model) )
    sim.add( ElectronPairProduction(model) )
    sim.add( PhotoDisintegration(model) )
    sim.add( NuclearDecay() )