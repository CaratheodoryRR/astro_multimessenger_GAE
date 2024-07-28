import argparse
import numpy as np

from crpropa import *
from pathlib import Path

from src import UHECRs_sim_f as cpf
from src.crpropa_building_blocks import prop_3D
from src.utils import file_utils as flu
from src.utils.general import get_dict_from_yaml, print_args
from src.utils.coords import coordinate_transformation_handler
from src.loaders.fields import setting_dolag_field, setting_jf12_field
from src.crpropa_building_blocks.prop_general import source_energy_spectrum

# We model the galaxy as a sphere of radius 20 kpc
rGalaxy = 20.*kpc

def run(
    JF12_field, # JF12 field object
    Dolag_field, # Dolag field grid
    srcPath = Path('../data/EG_3D_sources.txt'), # Path to a text file containing source positions
    Coords = 'galactic', # Coordinate system used in `srcPath`
    outDir = Path('./'), # Path to the output directory
    yamlFile = Path('./fracs.yaml'), # Path to a YAML file containing relative CRs abundances
    maxEnergy = 21.0, # Maximum source energy (10^maxEnergy eV)
    minEnergy = 18.0, # Minimum source energy (10^minEnergy eV)
    stopEnergy = None, # Simulation's stopping energy (10^stopEnergy eV)
    tau = None, # Concentration parameter for the vMF distribution (default: directed emission)
    rcut = 20.5, # Rigidity breakpoint for the broken exponential cut-off function (10^rcut V)
    alpha = 2.3, # Power-Law exponent (dN/dE ~ E^-alpha)
    num = 100, # Number of emitted CRs (in thousands)
    parts = 1, # Divide the simulation into n parts (avoids segmentation faults due to insufficient memory)
    rigLim = False, # Whether to use rigidity limits for source energy emissions
    noInteractions = False, # Deactivate all CRs interactions
    barProgress = True, # Show the crpropa built-in bar progress
    prop = 'both' # Propagation stages to consider in the simulation ['both', 'extra-galactic', 'galactic']
):
    flu.check_dir(outDir)
    if prop != 'galactic': flu.del_by_extension(outDir, exts=('.gz', '.txt', '.dat'), recursive=True)
    fnameOutput = 'events'

    rnd = (str(srcPath) == 'random')
    if rnd:
        srcType = 'random'
    elif str(srcPath).endswith('.raw'):
        srcType = 'grid'
    else:
        srcType = 'point-like'

    if stopEnergy is None:
        stopEnergy = minEnergy
        
    fname_func = lambda outPath, ith, name, ext: '{0}/{1}_{4}_{2}_of_{3}.{5}'.format(outPath,
                                                                                     fnameOutput, 
                                                                                     ith, 
                                                                                     parts,
                                                                                     name,
                                                                                     ext)
        
        
    cpf.minE = 10.**minEnergy * eV
    cpf.maxE = 10.**maxEnergy * eV
    cpf.stopE = 10.**stopEnergy * eV
    rcut = 10.**rcut * eV

    kwargsProp = {}
    if not rnd: kwargsProp['srcPath'] = srcPath
    kwargsProp['spectrumStr'] = source_energy_spectrum(alpha, rcut)
    kwargsProp['nucleiFracs'] = yamlFile if isinstance(yamlFile, dict) else get_dict_from_yaml(yamlFile)
    kwargsProp['rigidityLimits'] = rigLim
    if srcType == 'point-like':
        kwargsProp['Coords'] = Coords
        kwargsProp['tau'] = tau
    
    filesDict = {}
    ##############################################################################################################
    #                                   EXTRAGALACTIC PROPAGATION (DOLAG MODEL)
    ##############################################################################################################
    
    outDirEG = outDir.joinpath('Extra-Galactic-part')
    flu.check_dir(outDirEG)
    garbageDir = outDir.joinpath('Garbage')
    flu.check_dir(garbageDir)
    
    filesDict['extra-galactic'] = [fname_func(outPath=outDirEG,ith=i+1,name='Dolag_part',ext='txt.gz') for i in range(parts)]
    
    fileIntegrity = [(Path(f).exists() and flu.not_empty_file(f)) for f in filesDict['extra-galactic']]
    
    if prop != 'galactic' or not all(fileIntegrity):
        partNum = (num*1000) // parts
        
        run_extra_galactic_part(filesDict={'extra-galactic':[f for (f, integrity) in zip(filesDict['extra-galactic'], fileIntegrity) if not integrity],
                                           'garbage':[fname_func(outPath=garbageDir,
                                                      ith=i+1,
                                                      name='garbage_Dolag_part',
                                                      ext='txt') for (i, integrity) in enumerate(fileIntegrity) if not integrity]},
                                kwargsProp=kwargsProp,
                                srcType=srcType,
                                partNum=partNum,
                                tau=tau,
                                field=Dolag_field,
                                interactions=(not noInteractions),
                                barProgress=barProgress,
                                tolerance=1e-5,
                                minStep=10.*kpc,
                                maxStep=100.*kpc)
    
    ##############################################################################################################
    #                                       GALACTIC PROPAGATION (JF12 MODEL)
    ##############################################################################################################
    
    outDirG = outDir.joinpath('Galactic-part')
    flu.check_dir(outDirG)
    
    if prop != 'extra-galactic':
        filesDict['galactic'] = [fname_func(outPath=outDirG,ith=i+1,name='JF12_part',ext='txt') for i in range(parts)]
        filesDict['garbage'] = [fname_func(outPath=garbageDir,ith=i+1,name='garbage_JF12_part',ext='txt') for i in range(parts)]
        
        run_galactic_part(filesDict=filesDict,
                          rObs=1*kpc,
                          field=JF12_field,
                          interactions=(not noInteractions),
                          barProgress=barProgress,
                          tolerance=1e-5,
                          minStep=10.*pc,
                          maxStep=100*pc)


def run_extra_galactic_part(filesDict, kwargsProp, srcType, partNum, tau, **kwargsSim):

    # simulation setup
    sim = ModuleList()

    # Sources
    sources = prop_3D.set_sources_handler(srcType, **kwargsProp)
    if srcType != 'point-like':
        fixedPoint = Vector3d(0)
        if tau is None:
            customSourceFeature = prop_3D.SourceDirectionTowardsPoint(fixedPoint)
        else: 
            customSourceFeature = prop_3D.SourceDirectedvMFTowardsPoint(fixedPoint, tau)
        sources.add( customSourceFeature )
    # Propagator, interactions and break condition
    cpf.maxTrajectoryL = 200. * Mpc
    prop_3D.set_simulation(sim=sim, **kwargsSim)

    # Observer
    EG_obs = Observer()
    EG_obs.add(ObserverSurface( Sphere(Vector3d(0), rGalaxy) ))
    
    # Observer 2 (barely farther than the farthest, for speeding things up)
    test_obs = Observer()
    test_obs.add(ObserverSurface( Sphere(Vector3d(0), prop_3D.farthestSourceDistance.get(srcType)) ))

    print('\n\n\t\tFIRST STAGE: EXTRAGALACTIC PROPAGATION\n ')

    # run simulation
    outputs = [TextOutput(fname, Output.Everything) for fname in filesDict['extra-galactic']]
    outputs2 = [TextOutput(fname, Output.Event3D) for fname in filesDict['garbage']]
    
    parts = len(outputs)
    for i, (output, output2, dolagFileName) in enumerate(zip(outputs, outputs2, filesDict['extra-galactic'])):
        print('\n\tRUNNING PART {0} OF {1}\n'.format(i+1, parts))
        
        EG_obs.onDetection( output )
        EG_obs.setDeactivateOnDetection(True)
        sim.add(EG_obs)
        
        test_obs.onDetection( output2 )
        test_obs.setDeactivateOnDetection(True)
        sim.add(test_obs)
        
        sim.run(sources, partNum)
        
        print('Results successfully saved at {}'.format(dolagFileName))
        output.close()
        sim.remove(sim.size()-1)
        sim.remove(sim.size()-1)


def run_galactic_part(filesDict, rObs, **kwargsSim):

    # Simulation setup
    sim = ModuleList()

    # Propagator, interactions and break condition
    #cpf.maxTrajectoryL = 200. * kpc
    prop_3D.set_simulation(sim=sim, **kwargsSim)

    # Observer 1 (Earth)
    G_obs = Observer()
    G_obs.add(ObserverSurface( Sphere(Vector3d(-8.5*kpc, 0, 0), rObs) ))

    # Observer 2 (barely greater than the galaxy, for speeding things up)
    test_obs = Observer()
    test_obs.add(ObserverSurface( Sphere(Vector3d(0), 1.01*rGalaxy) ))

    outputs = [TextOutput(fname, Output.Event3D) for fname in filesDict['galactic']]
    outputs2 = [TextOutput(fname, Output.Event3D) for fname in filesDict['garbage']]

    print('\n\n\t\tSECOND STAGE: GALACTIC PROPAGATION\n ')

    input = ParticleCollector()
    parts = len(outputs)
    for i, (output, output2, EGFiles) in enumerate(zip(outputs, outputs2, filesDict['extra-galactic'])):
        print('\n\tRUNNING PART {0} OF {1}\n'.format(i+1, parts))
        input.load(EGFiles)
        
        inputsize = len(input)
        print('Number of candidates: {}\n'.format(inputsize))
        
        G_obs.onDetection( output )
        G_obs.setDeactivateOnDetection(True)
        sim.add(G_obs)

        test_obs.onDetection( output2 )
        test_obs.setDeactivateOnDetection(True)
        sim.add(test_obs)
        
        sim.run(input.getContainer())
        
        output.close()
        output2.close()
        
        sim.remove(sim.size()-1)
        sim.remove(sim.size()-1)
        input.clearContainer()
    
    input.shrinkToFit()

def args_parser_function():
    parser = argparse.ArgumentParser(
                    description='Cosmic ray propagation on both extragalactic (Dolag) and galactic (JF12) magnetic fields',
                    epilog='GAE-PUCP Astroparticle Physics')
    parser.add_argument('-s', '--srcPath', default='../data/EG_3D_sources.txt', type=Path,
                        help='Path to the EG sources text file (default: %(default)s)')
    parser.add_argument('-D', '--dolagPath', default='../data/dolag_B_54-186Mpc_440b.raw', type=Path,
                        help='Path to the Dolag EGMF raw file (default: %(default)s)')
    parser.add_argument('-o', '--outDir', default='./', type=Path,
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
    parser.add_argument('-y', '--yamlFile', default='./fracs.yaml', type=Path,
                        help='File containing the nuclei relative abundances (default: %(default)s)')
    parser.add_argument('-a', '--alpha', default=2.3, type=float, 
                        help='Power-Law exponent [dN/dE ~ E^-a] (default: %(default)s)')
    parser.add_argument('-r', '--rcut', default=20.5, type=float, 
                        help='Rigidity breakpoint for the broken exponential cut-off function  [10^r V] (default: %(default)s)')
    parser.add_argument('--rigLim', action='store_true', 
                        help='Whether to use rigidity limits for source energy emissions (default: %(default)s)')
    parser.add_argument('--prop', default='both',
                        choices=['both', 'extra-galactic', 'galactic'],
                        help='Which stages to consider in the simulation (default: %(default)s)')
    
    args = parser.parse_args()
    print_args(args)
    
    return args

def main(args):
    # Setting the magnetic fields
    if args.prop != 'galactic': 
        print('Setting up Dolag Extragalactic Magnetic Field...')
        Dolag_field = setting_dolag_field(pathToDolag=args.dolagPath, bFactor=args.bFactor)
        print('Done!\n')
    else:
        Dolag_field = None
    
    if args.prop != 'extra-galactic': 
        print('Setting up JF12 Galactic Magnetic Field...')
        JF12_field = setting_jf12_field()
        print('Done!\n')
    else:
        JF12_field = None
        
    delattr(args, 'dolagPath')
    delattr(args, 'bFactor')
    run(JF12_field=JF12_field, Dolag_field=Dolag_field, **vars(args))
    
    flu.del_by_extension(parentDir=Path(''), exts=('.pyc',), recursive=True)
    
if __name__ == '__main__':
    args = args_parser_function()
    main(args)


