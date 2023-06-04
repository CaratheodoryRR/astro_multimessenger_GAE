# -*- coding: utf-8 -*-
from crpropa import *
import numpy as np
import UHECRs_sim_f as cpf

# Print iterations progress
# Copied from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    #print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd) # For python3
    print '\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), # For python2
    # Print New Line on Complete
    if iteration == total: 
        #print() # For python3
        print '\n' # For python2

def sim_settings(sim, model = IRB_Gilmore12(), minE = 1.*EeV):
    sim.add( Redshift() )

    # Simulation processes
    sim.add(PhotoPionProduction(CMB()))
    sim.add(ElectronPairProduction(CMB()))
    sim.add(PhotoDisintegration(CMB()))
    sim.add(PhotoPionProduction(model))
    sim.add(ElectronPairProduction(model))
    sim.add(PhotoDisintegration(model))
        
    sim.add(NuclearDecay())
    # Stop if particle reaches this energy 
    sim.add(MinimumEnergy(minE))


dirOutput = './'
cpf.check_dir(dirOutput)
fnameOutput = 'events'


sources = np.genfromtxt('EG_sources.txt',names=True)

# source
sourcelist = SourceList()
tau = 100. # Concentration parameter

##############################################################################################################
#                                   EXTRAGALACTIC PROPAGATION (DOLAG MODEL)
##############################################################################################################
r_galaxy = 20.*kpc


# magnetic field setup
filename_bfield = 'dolag_B_54-186Mpc_440b.raw'
gridSize = 440
size = 186*Mpc
b_factor = 1e-3
spacing = size/(gridSize)
obsPosition = Vector3d(0,0,0)                       
boxOrigin = Vector3d(-0.5*size,-0.5*size,-0.5*size) 

vgrid = Grid3f(boxOrigin, gridSize, spacing )
loadGrid(vgrid, filename_bfield, b_factor)
Dolag_field = MagneticFieldGrid(vgrid)


# simulation setup
sim = ModuleList()
sim.add(PropagationBP(Dolag_field, 1e-4, 1.*kpc, 1.*Mpc))

sim_settings(sim=sim)


# observer and output
EG_obs = Observer()
EG_obs.add(ObserverSurface( Sphere(Vector3d(0), r_galaxy) ))
#output = TextOutput('{}_Dolag.txt'.format(fnameOutput), Output.Event3D)
output = ParticleCollector()
EG_obs.onDetection( output )
sim.add(EG_obs)


# source
sourcelist = SourceList()
tau = 100. # Concentration parameter
for source in sources:
    s = Source()
    v = Vector3d()
    v.setRThetaPhi(source['Distance'], source['Longitude'], source['Latitude'])
    s.add(SourcePosition(v * Mpc))
    #s.add(SourceDirectedEmission( v.getUnitVector() * (-1.), tau )) # vMF distribution
    s.add(SourceDirection( v.getUnitVector() * (-1.))) # Emission in one direction
    s.add(SourceParticleType(nucleusId(1, 1)))
    s.add(SourcePowerLawSpectrum(1 * EeV, 1000 * EeV, -2.3))
    sourcelist.add(s, 1)

# run simulation
sim.setShowProgress(True)
sim.run(sourcelist, 10**5)

output.dump('{}{}_Dolag.txt.gz'.format(dirOutput,fnameOutput))
#output.close()


##############################################################################################################
#                                       GALACTIC PROPAGATION (JF12 MODEL)
##############################################################################################################

r_obs = 1*kpc 

# magnetic field setup
JF12_field = JF12Field()
JF12_field.randomStriated()
JF12_field.randomTurbulent()

# simulation setup
sim = ModuleList()
sim.add(PropagationBP(JF12_field, 1e-4, 0.1*pc, 1.*kpc))

sim_settings(sim=sim)


# observer and output
G_obs = Observer()
G_obs.add(ObserverSurface( Sphere(Vector3d(0), r_obs) ))
output = TextOutput('{}{}_JF12.txt'.format(dirOutput,fnameOutput), Output.Event3D)
G_obs.onDetection( output )
sim.add(G_obs)

# observer2 (for speeding things up)
test_obs = Observer()
test_obs.add(ObserverSurface( Sphere(Vector3d(0), 21.*kpc) ))
output2 = TextOutput('garbage_JF12.txt', Output.Event3D)
test_obs.onDetection( output2 )
sim.add(test_obs)


input = ParticleCollector()
input.load('{}{}_Dolag.txt.gz'.format(dirOutput,fnameOutput))
inputsize = len(input)

print('\nNumber of candidates: {}\n'.format(inputsize))

for i,c in enumerate(input):
    sim.run(c)
    
    printProgressBar(iteration=i+1, total=inputsize, prefix='Progress:', suffix='Complete')


output.close()
