from crpropa import *
import numpy as np
from src import UHECRs_sim_f as cpf
from src import auger_data_he as pao
from src import mod_functions as mf
import argparse

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
    sim.add(MinimumEnergy(cpf.minE))
    
def setting_dolag_field(pathToDolag, bFactor):
    # magnetic field setup
    gridSize = 440
    size = 186*Mpc
    spacing = size/(gridSize)
    boxOrigin = Vector3d(-0.5*size, -0.5*size, -0.5*size) 

    vgrid = Grid3f(boxOrigin, gridSize, spacing)
    loadGrid(vgrid, pathToDolag, bFactor)
    
    dolag_field = MagneticFieldGrid(vgrid)
    
    return dolag_field 

def setting_jf12_field():
    # magnetic field setup
    jf12_field = JF12Field()
    jf12_field.randomStriated()
    jf12_field.randomTurbulent()
    
    return jf12_field

    
def setting_sources(sources, source_list, emission_func, vec_pos_func, spectrumStr, nucleiFracs):
    
    source_template = SourceGenericComposition(cpf.minE, cpf.maxE, spectrumStr, 10**5)
    for z in nucleiFracs:
        nuclearCode = nucleusId(cpf.A_Z[z][0], cpf.A_Z[z][1])
        source_template.add(nuclearCode, nucleiFracs[z])
        
    for source in sources:
        s = Source()
        s.add(source_template)
        v = Vector3d()
        vec_pos_func(v, source)
        s.add(SourcePosition(v * Mpc))
        s.add(emission_func(v.getUnitVector() * (-1.)))
        source_list.add(s, 1)


parser = argparse.ArgumentParser(
                    description='Cosmic ray propagation on both extragalactic (Dolag) and galactic (JF12) magnetic fields',
                    epilog='GAE-PUCP Astroparticle Physics')

parser.add_argument('-s', '--srcPath', default='../data/EG_sources.txt',
                    help='Path to the EG sources text file (default: %(default)s)')
parser.add_argument('-D', '--dolagPath', default='../data/dolag_B_54-186Mpc_440b.raw',
                    help='Path to the Dolag EGMF raw file (default: %(default)s)')
parser.add_argument('-o', '--outDir', default='./',
                    help='Directory to store the CRs output information (default: %(default)s)')
parser.add_argument('-g', '--minEnergy', default=18., type=float, 
                    help='Minimum emission energy exponent [10^g eV] (default: %(default)s)')
parser.add_argument('-e', '--maxEnergy', default=21., type=float, 
                    help='Maximum emission energy exponent [10^e eV] (default: %(default)s)')
parser.add_argument('-x', '--stopEnergy', type=float,
                    help='CRs below this energy exponent are discarded [10^d eV] (default: minEnergy)')
parser.add_argument('-n', '--num', default=100, type=int,
                    help='Total number of emitted cosmic rays, in thousands (default: %(default)s)')
parser.add_argument('-b', '--bFactor', default=1e-4, type=float,
                    help='Scale factor for the EGMF (default: %(default)s)')
parser.add_argument('-t', '--tau', type=float,
                    help='Concentration parameter for the vMF distribution (default: %(default)s)')
parser.add_argument('-p', '--parts', type=int, default=1,
                    help='Divide the simulation into n parts (default: %(default)s)')
parser.add_argument('-C', '--Coords', default='galactic',
                    choices=['cartesian', 'spherical', 'galactic', 'icrs', 'supergalactic'],
                    help='Coordinate system of the sources (default: %(default)s)')
parser.add_argument('-y', '--yamlFile', default='./fracs.yaml',
                    help='File containing the nuclei relative abundances (default: %(default)s)')
parser.add_argument('-a', '--alpha', default=2.3, type=float, 
                    help='Power-Law exponent [dN/dE ~ E^-a] (default: %(default)s)')
parser.add_argument('-r', '--rcut', default=20.5, type=float, 
                    help='Rigidity breakpoint for the broken exponential cut-off function  [10^r V] (default: %(default)s)')

args = parser.parse_args()

dirOutput = args.outDir
cpf.check_dir(dirOutput)
fnameOutput = 'events'


sources = np.genfromtxt(args.srcPath, names=True)
sources = mf.coordinate_transformation_handler(sources, args.Coords)


if not args.tau:
    source_emission = lambda direction: SourceDirection( direction )
else:
    source_emission = lambda direction: SourceDirectedEmission( direction, args.tau )

if not args.stopEnergy:
    args.stopEnergy = args.minEnergy
    
if args.Coords == 'spherical':
    set_vector_position = lambda v, s: v.setRThetaPhi(s['R'], s['Theta'], s['Phi'])
else:
    set_vector_position = lambda v, s: v.setXYZ(s['X'], s['Y'], s['Z'])
    
fname_func = lambda ith, name, ext: '{0}/{1}_{4}_{2}_of_{3}.{5}'.format(dirOutput,
                                                                        fnameOutput, 
                                                                        ith, 
                                                                        args.parts,
                                                                        name,
                                                                        ext)
    
    
cpf.minE = 10.**args.minEnergy * eV
cpf.maxE = 10.**args.maxEnergy * eV
stopE = 10.**args.stopEnergy * eV
rcut = 10.**args.rcut * eV

energySpectrum = '(E/EeV)^-{0}*( (E > Z*{1}) ? exp(1 - E/(Z*{1})) : 1 )'.format(args.alpha, rcut)

nucleiFracs = mf.get_dict_from_yaml(args.yamlFile)

# Setting the magnetic fields
Dolag_field = setting_dolag_field(pathToDolag=args.dolagPath, bFactor=args.bFactor)
JF12_field = setting_jf12_field()
##############################################################################################################
#                                   EXTRAGALACTIC PROPAGATION (DOLAG MODEL)
##############################################################################################################
# simulation setup
sim = ModuleList()

# Sources
source_list = SourceList()
setting_sources(sources=sources, 
                source_list=source_list,
                emission_func=source_emission,
                vec_pos_func=set_vector_position,
                spectrumStr = energySpectrum,
                nucleiFracs=nucleiFracs)

# Propagator
sim.add(PropagationBP(Dolag_field, 1e-4, 1.*kpc, 1.*Mpc))

# Interactions and break condition
sim_settings(sim=sim, minE=stopE)

# Observer
rGalaxy = 20.*kpc
EG_obs = Observer()
EG_obs.add(ObserverSurface( Sphere(Vector3d(0), rGalaxy) ))

#outputs = []
# for i in range(args.parts):
#     fname = fname_func(ith=i+1, name='Dolag_part', ext='txt.gz')
#     output = TextOutput(fname, Output.Event3D)
#     output.enable(Output.SerialNumberColumn)

output = ParticleCollector()
EG_obs.onDetection( output )
sim.add(EG_obs)


print('\n\n\t\tFIRST STAGE: EXTRAGALACTIC PROPAGATION\n ')

# run simulation
sim.setShowProgress(True)
partNum = (args.num*1000) // args.parts
dolagFileNames = [fname_func(ith=i+1,name='Dolag_part',ext='txt.gz') for i in range(args.parts)]

for i, dolagFileName in enumerate(dolagFileNames):
    print('\n\tRUNNING PART {0} OF {1}\n'.format(i+1, args.parts))
    sim.run(source_list, partNum)

    output.dump(dolagFileName)
    output.clearContainer()

##############################################################################################################
#                                       GALACTIC PROPAGATION (JF12 MODEL)
##############################################################################################################

# Simulation setup
sim = ModuleList()

# Propagator
sim.add(PropagationBP(JF12_field, 1e-4, 0.1*pc, 1.*kpc))

# Interactions and break condition
sim_settings(sim=sim, minE=stopE)

# Observer 1 (Earth)
rObs = 1*kpc 
G_obs = Observer()
G_obs.add(ObserverSurface( Sphere(Vector3d(-8.5*kpc, 0, 0), rObs) ))

# Observer 2 (barely greater than the galaxy, for speeding things up)
test_obs = Observer()
test_obs.add(ObserverSurface( Sphere(Vector3d(0), 21.*kpc) ))

JF12FileNames = [fname_func(ith=i+1,name='JF12_part',ext='txt') for i in range(args.parts)]
outputs = [TextOutput(fname, Output.Event3D) for fname in JF12FileNames]

garbageFileNames = [fname_func(ith=i+1,name='garbage_JF12_part',ext='txt') for i in range(args.parts)]
outputs2 = [TextOutput(fname, Output.Event3D) for fname in garbageFileNames]

print('\n\n\t\tSECOND STAGE: GALACTIC PROPAGATION\n ')

input = ParticleCollector()
sim.setShowProgress(True)
for i, (output, output2, dolagFileName) in enumerate(zip(outputs, outputs2, dolagFileNames)):
    print('\n\tRUNNING PART {0} OF {1}\n'.format(i+1, args.parts))
    input.load(dolagFileName)
    
    inputsize = len(input)
    print('Number of candidates: {}\n'.format(inputsize))
    
    G_obs.onDetection( output )
    sim.add(G_obs)

    test_obs.onDetection( output2 )
    sim.add(test_obs)
    
    sim.run(input.getContainer())
    
    output.close()
    output2.close()
    
    sim.remove(sim.size()-1)
    sim.remove(sim.size()-1)
    input.clearContainer()

