import argparse
import numpy as np

from crpropa import *
from pathlib import Path

from src import UHECRs_sim_f as cpf
from src import auger_data_he as pao
from src.utils.general import get_dict_from_yaml, print_args
from src.utils.coords import coordinate_transformation_handler
from src.utils.file_utils import check_dir, del_by_extension
from src.loaders.fields import setting_dolag_field, setting_jf12_field



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
    barProgress = True # Show the crpropa built-in bar progress 
):
    check_dir(outDir)
    del_by_extension(outDir, exts=('.gz', '.txt', '.dat'), recursive=True)
    fnameOutput = 'events'


    sources = np.genfromtxt(srcPath, names=True)
    redshiftPresent = 'Redshift' in sources.dtype.names    
    redshifts = sources['Redshift'] if redshiftPresent else None
    sources = coordinate_transformation_handler(sources, Coords)


    if tau is None:
        source_emission = lambda direction: SourceDirection( direction )
    else:
        source_emission = lambda direction: SourceDirectedEmission( direction, tau )

    if stopEnergy is None:
        stopEnergy = minEnergy
        
    if Coords == 'spherical':
        set_vector_position = lambda v, s: v.setRThetaPhi(s['R'], s['Theta'], s['Phi'])
    else:
        set_vector_position = lambda v, s: v.setXYZ(s['X'], s['Y'], s['Z'])
        
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

    energySpectrum = '(E/EeV)^-{0}*( (E > Z*{1}) ? exp(1 - E/(Z*{1})) : 1 )'.format(alpha, rcut)

    nucleiFracs = yamlFile if isinstance(yamlFile, dict) else get_dict_from_yaml(yamlFile)

    ##############################################################################################################
    #                                   EXTRAGALACTIC PROPAGATION (DOLAG MODEL)
    ##############################################################################################################
    
    outDirEG = outDir.joinpath('Extra-Galactic-part')
    check_dir(outDirEG)
    
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
    sim_settings(sim=sim)

    # Observer
    rGalaxy = 20.*kpc
    EG_obs = Observer()
    EG_obs.add(ObserverSurface( Sphere(Vector3d(0), rGalaxy) ))

    #outputs = []
    # for i in range(parts):
    #     fname = fname_func(ith=i+1, name='Dolag_part', ext='txt.gz')
    #     output = TextOutput(fname, Output.Event3D)
    #     output.enable(Output.SerialNumberColumn)

    # output = ParticleCollector()
    # EG_obs.onDetection( output )
    # sim.add(EG_obs)


    print('\n\n\t\tFIRST STAGE: EXTRAGALACTIC PROPAGATION\n ')

    # run simulation
    sim.setShowProgress(barProgress)
    partNum = (num*1000) // parts
    dolagFileNames = [fname_func(outPath=outDirEG,ith=i+1,name='Dolag_part',ext='txt.gz') for i in range(parts)]
    outputs = [TextOutput(fname, Output.Everything) for fname in dolagFileNames]

    for i, (output, dolagFileName) in enumerate(zip(outputs, dolagFileNames)):
        print('\n\tRUNNING PART {0} OF {1}\n'.format(i+1, parts))
        EG_obs.onDetection( output )
        sim.add(EG_obs)
        sim.run(source_list, partNum)
        
        print('Results successfully saved at {}'.format(dolagFileName))
        output.close()
        sim.remove(sim.size()-1)
        # output.dump(dolagFileName)
        # output.clearContainer()

    ##############################################################################################################
    #                                       GALACTIC PROPAGATION (JF12 MODEL)
    ##############################################################################################################
    
    outDirG = outDir.joinpath('Galactic-part')
    check_dir(outDirG)
    garbageDir = outDir.joinpath('Garbage')
    check_dir(garbageDir)
    
    # Simulation setup
    sim = ModuleList()

    # Propagator
    sim.add(PropagationBP(JF12_field, 1e-4, 0.1*pc, 1.*kpc))

    # Interactions and break condition
    sim_settings(sim=sim)

    # Observer 1 (Earth)
    rObs = 1*kpc 
    G_obs = Observer()
    G_obs.add(ObserverSurface( Sphere(Vector3d(-8.5*kpc, 0, 0), rObs) ))

    # Observer 2 (barely greater than the galaxy, for speeding things up)
    test_obs = Observer()
    test_obs.add(ObserverSurface( Sphere(Vector3d(0), 21.*kpc) ))

    JF12FileNames = [fname_func(outPath=outDirG,ith=i+1,name='JF12_part',ext='txt') for i in range(parts)]
    outputs = [TextOutput(fname, Output.Event3D) for fname in JF12FileNames]

    garbageFileNames = [fname_func(outPath=garbageDir,ith=i+1,name='garbage_JF12_part',ext='txt') for i in range(parts)]
    outputs2 = [TextOutput(fname, Output.Event3D) for fname in garbageFileNames]

    print('\n\n\t\tSECOND STAGE: GALACTIC PROPAGATION\n ')

    input = ParticleCollector()
    sim.setShowProgress(barProgress)
    for i, (output, output2, dolagFileName) in enumerate(zip(outputs, outputs2, dolagFileNames)):
        print('\n\tRUNNING PART {0} OF {1}\n'.format(i+1, parts))
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
    
    args = parser.parse_args()
    print_args(args)
    
    return args


def sim_settings(sim, model = IRB_Gilmore12()):
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
    sim.add(MinimumEnergy(cpf.stopE))

    
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

def main(args):
    # Setting the magnetic fields
    Dolag_field = setting_dolag_field(pathToDolag=args.dolagPath, bFactor=args.bFactor)
    JF12_field = setting_jf12_field()
    
    delattr(args, 'dolagPath')
    delattr(args, 'bFactor')
    run(JF12_field=JF12_field, Dolag_field=Dolag_field, **vars(args))
    
    del_by_extension(parentDir=Path(''), exts=('.pyc',), recursive=True)
    
if __name__ == '__main__':
    args = args_parser_function()
    main(args)


