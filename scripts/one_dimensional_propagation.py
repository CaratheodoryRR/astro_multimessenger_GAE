import argparse
import numpy as np

from crpropa import *
from pathlib import Path

from src import UHECRs_sim_f as cpf
from src.crpropa_building_blocks import prop_1D
from src.utils.general import get_dict_from_yaml, print_args
from src.utils.file_utils import check_dir, del_by_extension
from src.crpropa_building_blocks.prop_general import source_energy_spectrum


def run(
    srcPath = Path('../data/EG_1D_sources.txt'), # Path to a text file containing source positions
    outDir = Path('./'), # Path to the output directory
    yamlFile = Path('./fracs.yaml'), # Path to a YAML file containing relative CRs abundances
    maxEnergy = 21.0, # Maximum source energy (10^maxEnergy eV)
    minEnergy = 18.0, # Minimum source energy (10^minEnergy eV)
    stopEnergy = None, # Simulation's stopping energy (10^stopEnergy eV)
    rcut = 20.5, # Rigidity breakpoint for the broken exponential cut-off function (10^rcut V)
    alpha = 2.3, # Power-Law exponent (dN/dE ~ E^-alpha)
    num = 100, # Number of emitted CRs (in thousands)
    parts = 1, # Divide the simulation into n parts (avoids segmentation faults due to insufficient memory)
    rigLim = False, # Whether to use rigidity limits for source energy emissions
    noInteractions = False, # Deactivate all CRs interactions
    barProgress = True # Show the crpropa built-in bar progress 
):
    
    check_dir(outDir)
    del_by_extension(outDir, exts=('.gz', '.txt', '.dat'), recursive=True)
    fnameOutput = 'events'

    sources = np.genfromtxt(srcPath, names=True)

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

    energySpectrum = source_energy_spectrum(alpha, rcut)

    nucleiFracs = yamlFile if isinstance(yamlFile, dict) else get_dict_from_yaml(yamlFile)

    ##############################################################################################################
    #                                   CR 1D Propagation
    #############################################################################################################
    
    # simulation setup
    sim = ModuleList()

    # Sources
    source_list = SourceList()
    prop_1D.set_sources_handler(rigidityLimits=rigLim,
                                sources=sources,
                                source_list=source_list,
                                spectrumStr=energySpectrum,
                                nucleiFracs=nucleiFracs)

    # Propagator, interactions and break condition
    prop_1D.set_simulation(sim=sim,
                           interactions=(not noInteractions),
                           minStep=1.*kpc,
                           maxStep=1.*Mpc)

    # Observer
    obs = Observer()
    # Observer at x=0
    obs.add(Observer1D())

    print('\n\n\t\tONE DIMENSIONAL PROPAGATION\n ')

    # run simulation
    sim.setShowProgress(barProgress)
    partNum = (num*1000) // parts
    propFileNames = [fname_func(outPath=outDir,ith=i+1,name='prop_1D',ext='dat') for i in range(parts)]
    outputs = [TextOutput(fname, Output.Event1D) for fname in propFileNames]

    for i, (output, propFileName) in enumerate(zip(outputs, propFileNames)):
        print('\n\tRUNNING PART {0} OF {1}\n'.format(i+1, parts))
        obs.onDetection(output)
        sim.add(obs)
        sim.run(source_list, partNum)
        
        print('Results successfully saved at {}'.format(propFileName))
        output.close()
        sim.remove(sim.size()-1)

def args_parser_function():
    parser = argparse.ArgumentParser(
                    description='Cosmic ray propagation on one dimension',
                    epilog='GAE-PUCP Astroparticle Physics')
    parser.add_argument('-s', '--srcPath', default='../data/EG_1D_sources.txt', type=Path,
                        help='Path to the EG sources text file (default: %(default)s)')
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
    parser.add_argument('-p', '--parts', type=int, default=1,
                        help='Divide the simulation into n parts (default: %(default)s)')
    parser.add_argument('-y', '--yamlFile', default='./fracs.yaml', type=Path,
                        help='File containing the nuclei relative abundances (default: %(default)s)')
    parser.add_argument('-a', '--alpha', default=2.3, type=float, 
                        help='Power-Law exponent [dN/dE ~ E^-a] (default: %(default)s)')
    parser.add_argument('-r', '--rcut', default=20.5, type=float, 
                        help='Rigidity breakpoint for the broken exponential cut-off function  [10^r V] (default: %(default)s)')
    parser.add_argument('--rigLim', action='store_true', 
                        help='Whether to use rigidity limits for source energy emissions (default: %(default)s)')
    parser.add_argument('--noInteractions', action='store_true', 
                        help='Deactivate all CRs interactions (default: %(default)s)')

    args = parser.parse_args()
    print_args(args)
    
    return args

def main(args):
    run(**vars(args))
    
    del_by_extension(parentDir=Path(''), exts=('.pyc',), recursive=True)
    
if __name__ == '__main__':
    args = args_parser_function()
    main(args)


