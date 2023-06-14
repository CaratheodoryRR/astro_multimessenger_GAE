# -*- coding: utf-8 -*-
from crpropa import *
import numpy as np
from src import UHECRs_sim_f as cpf
from src import auger_data_he as pao
import argparse


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


parser = argparse.ArgumentParser(
                    description='Cosmic ray propagation on both extragalactic (Dolag) and galactic (JF12) magnetic fields',
                    epilog='GAE-PUCP Astroparticle Physics')

parser.add_argument('-s', '--srcPath', default='../data/EG_sources.txt',
                    help='Path to the EG sources text file (default: %(default)s)')
parser.add_argument('-D', '--dolagPath', default='../data/dolag_B_54-186Mpc_440b.raw',
                    help='Path to the Dolag EGMF raw file (default: %(default)s)')
parser.add_argument('-o', '--outDir', default='./',
                    help='Directory to store the CRs output information (default: %(default)s)')
parser.add_argument('-g', '--minEnergy', default=18, type=float, 
                    help='Minimum emission energy exponent [10^g eV] (default: %(default)s)')
parser.add_argument('-e', '--maxEnergy', default=21, type=float, 
                    help='Maximum emission energy exponent [10^e eV] (default: %(default)s)')
parser.add_argument('-x', '--stopEnergy', type=float,
                    help='CRs below this energy exponent are discarded [10^d eV] (default: minEnergy)')
parser.add_argument('-n', '--num', default=100, type=int,
                    help='Total number of emitted cosmic rays, in thousands (default: %(default)s)')
parser.add_argument('-b', '--bFactor', default=1e-4, type=float,
                    help='Scale factor for the EGMF (default: %(default)s)')

# TODO: Include an optional tau argument for vMF Emission Distribution

args = parser.parse_args()

dirOutput = args.outDir
cpf.check_dir(dirOutput)
fnameOutput = 'events'


sources = np.genfromtxt(args.srcPath, names=True)

# source
sourcelist = SourceList()
tau = 100. # Concentration parameter

if not args.stopEnergy:
    args.stopEnergy = args.minEnergy
    
minE = 10.**args.minEnergy * eV
maxE = 10.**args.maxEnergy * eV
stopE = 10.**args.stopEnergy * eV

##############################################################################################################
#                                   EXTRAGALACTIC PROPAGATION (DOLAG MODEL)
##############################################################################################################
r_galaxy = 20.*kpc


# magnetic field setup
filename_bfield = args.dolagPath
gridSize = 440
size = 186*Mpc
b_factor = args.bFactor
spacing = size/(gridSize)
obsPosition = Vector3d(0,0,0)                       
boxOrigin = Vector3d(-0.5*size, -0.5*size, -0.5*size) 

vgrid = Grid3f(boxOrigin, gridSize, spacing )
loadGrid(vgrid, filename_bfield, b_factor)
Dolag_field = MagneticFieldGrid(vgrid)


# simulation setup
sim = ModuleList()
sim.add(PropagationBP(Dolag_field, 1e-4, 1.*kpc, 1.*Mpc))

sim_settings(sim=sim, minE=stopE)

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
    s.add(SourcePowerLawSpectrum(minE, maxE, -2.3))
    sourcelist.add(s, 1)

# run simulation
sim.setShowProgress(True)
sim.run(sourcelist, 1000*args.num)

output.dump('{}/{}_Dolag.txt.gz'.format(dirOutput,fnameOutput))
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

sim_settings(sim=sim, minE=stopE)


# observer and output
G_obs = Observer()
G_obs.add(ObserverSurface( Sphere(Vector3d(-8.5*kpc, 0, 0), r_obs) ))
output = TextOutput('{}/{}_JF12.txt'.format(dirOutput,fnameOutput), Output.Event3D)
G_obs.onDetection( output )
sim.add(G_obs)

# observer2 (for speeding things up)
test_obs = Observer()
test_obs.add(ObserverSurface( Sphere(Vector3d(0), 21.*kpc) ))
output2 = TextOutput('garbage_JF12.txt', Output.Event3D)
test_obs.onDetection( output2 )
sim.add(test_obs)


input = ParticleCollector()
input.load('{}/{}_Dolag.txt.gz'.format(dirOutput,fnameOutput))
inputsize = len(input)

print('\nNumber of candidates: {}\n'.format(inputsize))

for i,c in enumerate(input):
    sim.run(c)
    
    printProgressBar(iteration=i+1, total=inputsize, prefix='Progress:', suffix='Complete')


output.close()
