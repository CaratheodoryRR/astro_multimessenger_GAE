from crpropa import *
from pathlib import Path

import auger_data_he as pao
import matplotlib.pyplot as plt
import sim_functions_he as sf
import utils.file_utils as flu
import numpy as np
import time

from .utils.error_functions import *


# Global variables
# =================================================================================

# Mass number and atomic number for different nuclei
A_Z = {
    'H':  (1,  1), 
    'He': (4,  2), 
    'N':  (14, 7), 
    'Si': (28, 14), 
    'Fe': (56, 26)
}

# Default values
# --------------
stopE = 1. # In EeV, used in MinimumEnergy()

# Minimum and maximum energies (in EeV) for the source energy distribution (see variable energy at ...)
minE = 1.
maxE = 1000.

E_bins = 10**4 # Number of bins for the energy distribution
n_E = 10**4 # Number of energy values taken from the energy distribution
# --------------

pao_nbins = len(pao.ecens)
# =================================================================================




# "Atomic" functions
# =================================================================================

def fracs_dict_complete(nuclei):
    for z in A_Z:
        if z not in nuclei:
            nuclei[z] = 0.
    return nuclei
    

def energy_distribution(Z, rcut_g, frac, E_dist):
    """
        Z      = Atomic number
        rcut_g = List or tuple of log10(rcut/EV) and gamma values
        frac   = Proportion of this nucleus in the mass composition
        E_dist = Energy distribution
    """
    probs = np.array([sf.delta_n(e = x, frac = frac, zeta = Z, Rcut = 10.**rcut_g[0], gamma = rcut_g[1]) for x in E_dist])
    probs /= np.sum(probs) # Normalization of propabilities
    return np.random.choice(E_dist, n_E, p = probs)


# Simulation interactions for 1D propagation
def sim_steps(sim, model):
    sim.add(SimplePropagation(1*kpc, 10*Mpc))
    # Simulation processes
    sim.add(PhotoPionProduction(CMB()))
    sim.add(ElectronPairProduction(CMB()))
    sim.add(PhotoDisintegration(CMB()))
    sim.add(PhotoPionProduction(model))
    sim.add(ElectronPairProduction(model))
    sim.add(PhotoDisintegration(model))
        
    sim.add(NuclearDecay())
    # Stop if particle reaches this energy 
    sim.add(MinimumEnergy(stopE*EeV))
    

def CR_sources(sourcelist, distance, element, energies, weight):
    """
        sourcelist = Container of sources. One source for every energy value, distance, and nucleus type
        distance   = distance of the source, in Mpc
        element    = Nucleus type
        energies   = Energy distribution for this nucleus
        weight     = weight of the source. We use the individual weights from the distance catalogue and 
                     the fractional composition for this particular nucleus (see "element" description)
    """
    for energy in energies:
        source = Source()
        source.add( SourcePosition(distance * Mpc) )
        source.add( SourceIsotropicEmission() )
        source.add( element ) 
        source.add( SourceEnergy(energy * EeV) )
        source.add( SourceRedshift1D() )
        sourcelist.add(source, weight)


def event_counter_by_energy(filenames, bins = pao.ebins, col = 3):
    
    counts = np.zeros(len(bins)-1)
    for filename in filenames:
        data = np.genfromtxt(filename, usecols = col)
        data = 18. + np.log10(np.copy(data))
        counts += np.histogram(data, bins = bins)[0]
    
    return counts

# =================================================================================




# "Simulation" functions
# =================================================================================

def many_sources_1D_parts(rcut_g, nuclei_f, distances_and_weights, num, output_dir, title = "new_simulation", model = "Gilmore12", parts = 1, seed = 0, sim_seed = 0):

    flu.check_dir(output_dir)

    # Energy distributions for each particle: H, He, N, Si, Fe
    energies = np.linspace(minE, maxE, E_bins)
    energy_Z = {} # Energy distribution per element
    #nuclei = map(lambda x: x.capitalize(), nuclei) #> BE CAREFUL HERE

    # We set the simulation seeds if their values are not 0   
    if seed:     np.random.seed(seed)
    if sim_seed: Random_seedThreads(sim_seed)
        
    # Cosmic infrared background model
    if (model == "Gilmore12"):   model = IRB_Gilmore12()
    if (model == "Dominguez11"): model = IRB_Dominguez11()

    sourcelist = SourceList()

    composition = {}
    # We create the energy distribution for each element and for each source
    for z in nuclei_f:
        energy_Z[z] = energy_distribution(Z = A_Z[z][1], rcut_g = rcut_g, frac = 1., E_dist = energies)
        composition[z] = SourceMultipleParticleTypes()
        composition[z].add(nucleusId(A_Z[z][0], A_Z[z][1]), 1)

        for d_w in distances_and_weights:
            # Cosmic ray source
            # ===========================================================
            CR_sources(sourcelist, distance=d_w[0], element=composition[z], energies=energy_Z[z], weight=nuclei_f[z]*d_w[1])
            # ===========================================================

    # We divide the model
    num_parts = int(np.ceil( num*1./parts )) 

    for i_parts in range(parts):
        
        # It could be improved to avoid fixed format
        filename = "{0}/{1}{2}.dat".format(output_dir, title, i_parts)
        # If the file already exists, we avoid to calculate it again
        if (Path(filename).exists): continue           

        print('\n\tPart {} of {} \n'.format(i_parts+1, parts))

        sim = ModuleList()

        # Simulation steps
        # =======================================
        sim_steps(sim, model)
        # =======================================
            
            
        # Define observer
        # ======================
        obs = Observer()
        # Observer at x=0
        obs.add(ObserverPoint()) 
        # ======================


        # Write output on detection of event
        # ===========================================
        output = TextOutput(filename, Output.Event1D)
        #output.disableAll()
        #output.enable(Output.CurrentEnergyColumn)
        #output.enable(Output.SourceEnergyColumn)
        #output.enable(Output.CreatedPositionColumn)
        #output.enable(Output.SourcePositionColumn)
        #output.enable(Output.CurrentPositionColumn)
        output.enable(Output.SerialNumberColumn)
        #output.enable(Output.SourceIdColumn)
        #output.enable(Output.CurrentIdColumn)
        # ===========================================
            
            
        obs.onDetection(output)
        sim.add(obs)
            
        # Run simulation
        sim.setShowProgress(True)

        sim.run(sourcelist, num_parts)

        # Load events
        output.close()



def simulate_1D_1Source_1Element(rcut_g, distances, nuclei, num, title = "new_simulation", model = "Gilmore12", seed = 0, sim_seed = 0):

    output_dir = './output'
    flu.check_dir(output_dir)
    
    # Energy distributions for each particle: H, He, N, Si, Fe
    energies = np.linspace(minE, maxE, E_bins)
    energy_Z = {} # Energy distribution per element
    Ntot = len(nuclei) * len(distances)
    nuclei = map(lambda x: x.capitalize(), nuclei)

    # We set the simulation seeds if their values are not 0   
    if seed:     np.random.seed(seed)
    if sim_seed: Random_seedThreads(sim_seed)
        
    # Cosmic infrared background model
    if (model == "Gilmore12"):   model = IRB_Gilmore12()
    if (model == "Dominguez11"): model = IRB_Dominguez11()
        
    # We create the energy distribution for each element
    for z in nuclei:
        energy_Z[z] = energy_distribution(Z = A_Z[z][1], rcut_g = rcut_g, frac = 1., E_dist = energies)

    #>------
    for i_d, distance in enumerate(distances):
        for j_z, z in enumerate(nuclei):

            # Name of the file
            filename = '{3}/{0}_{1}_{2}.dat'.format(title, i_d, z, output_dir) 
            if (Path(filename).exists): break           

            print('\n RUN {} of {} \n'.format(i_d*len(nuclei) + j_z + 1, Ntot))
            time_start = time.time()
            
            
            sim = ModuleList()

            # Simulation steps
            # =======================================
            sim_steps(sim, model)
            # =======================================
            
            
            # Define observer
            # ======================
            obs = Observer()
            # Observer at x=0
            obs.add(ObserverPoint()) 
            # ======================


            # Write output on detection of event
            # ===========================================
            output = TextOutput(filename, Output.Event1D)
            #output.disableAll()
            #output.enable(Output.CurrentEnergyColumn)
            #output.enable(Output.SourceEnergyColumn)
            #output.enable(Output.CreatedPositionColumn)
            #output.enable(Output.SourcePositionColumn)
            #output.enable(Output.CurrentPositionColumn)
            output.enable(Output.SerialNumberColumn)
            #output.enable(Output.SourceIdColumn)
            #output.enable(Output.CurrentIdColumn)
            # ===========================================
            
            
            obs.onDetection(output)
            sim.add(obs)

            #############################################################
            # Cosmic ray source
            # ===========================================================
            element = SourceParticleType(nucleusId(A_Z[z][0], A_Z[z][1]))
            sourcelist  = SourceList()
            CR_sources(sourcelist, distance, element, energies=energy_Z[z], weight=1.)
            # ===========================================================
            
             # Run simulation
            sim.setShowProgress(True)

            sim.run(sourcelist, num[j_z])

            # Load events
            output.close()
            
            minutes = (time.time()-time_start)/60.0
            print('\n TIME: {:.4f} minutes \n'.format(minutes))
            
# =================================================================================




# Plotting functions
# =================================================================================

def plot_counts_vs_distance(distances, nuclei, title, num, last_n_bins = 3):
    
    fileout = './counts_vs_distance'
    flu.check_dir(fileout)
    lEbins = pao.ebins[-3-last_n_bins:-2]  # logarithmic bins
    lEcens = (lEbins[1:] + lEbins[:-1]) / 2  # logarithmic bin centers
    it_d = range(len(distances)) # iterator for distance indexes
    
    for i_z, z in enumerate(nuclei):
        
        print('\n\nCreating plot for {}'.format(z))
        Nf = []
        for j_d in it_d:
            # Name of the file
            filename = 'output/{0}_{1}_{2}.dat'.format(title, j_d, z)
            data = np.genfromtxt(filename, names = True, usecols=3)
            lEf = np.log10(data['E']) + 18
            # calculate distribution: N(E)
            Nf.append( np.histogram(lEf, bins = lEbins)[0] )
        
        Nf = np.array(Nf).T
        
        fig, ax = plt.subplots(figsize=(10,7))
        
        np.random.seed(10)
        col = np.random.rand(pao_nbins,3) # Random RGB colors
        
        #> Different markers for each plot 
        #> See the matplotlib marker documentation for further reference: 
        # https://matplotlib.org/stable/api/markers_api.html
        # -----------
        sides = np.random.randint(3, 8, size=pao_nbins)
        mark = np.random.randint(3, size=pao_nbins)
        marker = np.array( [(i, j, 0) for i,j in  zip(sides, mark)] )
        # -----------
        
        for i in range(last_n_bins):
            ax.plot(distances, Nf[i], color=col[i], marker = marker[i], \
            label='eCent = {0:.4f} (PAO = {1:,})'.format(lEcens[i], pao.auger[-last_n_bins-2+i]))
        
        plt.legend(fontsize=12, frameon=True)  
        plt.xscale('symlog')
        plt.yscale('symlog')
        plt.title( 'Element: {0}      Number of particles = {1:,}'.format(z,num[i_z]), fontsize=20 )
        plt.grid()
        plt.ylabel('Counts', fontsize = 18, x = 0.8)
        plt.xlabel('Distances $(Mpc)$', fontsize = 18, y = 0.8)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.savefig('{2}/{0}_{1}_nlines_{3}.jpg'\
        .format(title, z, fileout, last_n_bins), dpi=300, bbox_inches='tight')
        plt.close()

        
    print('\n\nPlotting finished!\n')
        

def best_fit_plot_parts(parts, num, nplots = 5, plot_types = ('chi2', 'sqerr', 'sqlgerr', 'sqrlerr'), plots_dir = './plots'):
    
    flu.check_dir(plots_dir)
    
    mydata = "../distances_test.txt"
    distances = np.genfromtxt(mydata)
    outdir = './output'

    elements = ('H', 'He', 'N', 'Si', 'Fe')
    
    lEbins = pao.ebins  # logarithmic bins
    lEcens = pao.ecens  # logarithmic bin centers
    dE = pao.dE  # bin widths
    
    for plot_type in plot_types:
        
        data = np.genfromtxt('grid_bf_search_sorted_by_{}.dat'.format(plot_type), names=True, max_rows=nplots)
        g_rcut = data[['gamma','rcut']] 
        fracs = data[['fh','fhe','fn','fsi','ffe']]
        
        for i in range(nplots):
            
            print('\n Plot {0} of {1} for {2}'.format(i+1, nplots, plot_type))
            time_start = time.time()
            plotfile = '{0}_bf{1}.png'.format(plot_type, i+1)
            title = '{0}_bf{1}_part_'.format(plot_type,i+1)
            nuclei = {elements[k]:fracs[i][k] for k in range(len(elements)) if fracs[i][k] != 0.}
            many_sources_1D_parts(rcut_g=(g_rcut[i][1],g_rcut[i][0]),nuclei_f=nuclei,output_dir=outdir,distances_and_weights=distances,title=title,parts=parts,num=num)
            files = ["{0}/{1}{2}.dat".format(outdir, title, i_parts) for i_parts  in range(parts)]
            
            if plot_type == 'chi2':
                tst = sf.chi2_global_augerparts(title = title, parts = parts)
            else:
                J_sim = event_counter_by_energy(filenames=files)/dE
                J_sim /= J_sim[0] # Normalization
                tst = err_parameter_handler(N_sim=J_sim, type_err=plot_type)
            
            print('\nPlotting and saving in {}...\n'.format(plotfile))
            count = 0
            for i_parts in range(parts):
                
                if flu.not_empty_file(files[i_parts]):
                    d = np.genfromtxt(files[i_parts], names=True)
                    d2 = np.genfromtxt(files[i_parts])
                        
                    if (d2.ndim == 2):
                        # observed quantities
                        Z = np.array([chargeNumber(int(id)) for id in d['ID'].astype(int)])  # element
                        A = np.array([massNumber(int(id)) for id in d['ID'].astype(int)])  # atomic mass number
                        lE = np.log10(d['E']) + 18  # energy in log10(E/eV))
                        
                        # identify mass groups
                        idx1 = A == 1
                        idx2 = (A > 1) * (A <= 4)
                        idx3 = (A > 4) * (A <= 22)
                        idx4 = (A > 22) * (A <= 38)
                        idx5 = (A > 38)

                        if (count == 0):
                            # calculate spectrum: J(E) = dN/dE
                            J  = np.histogram(lE, bins=lEbins)[0].astype(float)
                            J1 = np.histogram(lE[idx1], bins=lEbins)[0].astype(float)
                            J2 = np.histogram(lE[idx2], bins=lEbins)[0].astype(float)
                            J3 = np.histogram(lE[idx3], bins=lEbins)[0].astype(float)
                            J4 = np.histogram(lE[idx4], bins=lEbins)[0].astype(float)
                            J5 = np.histogram(lE[idx5], bins=lEbins)[0].astype(float)
                        else:
                            J  += np.histogram(lE, bins=lEbins)[0].astype(float)
                            J1 += np.histogram(lE[idx1], bins=lEbins)[0].astype(float)
                            J2 += np.histogram(lE[idx2], bins=lEbins)[0].astype(float)
                            J3 += np.histogram(lE[idx3], bins=lEbins)[0].astype(float)
                            J4 += np.histogram(lE[idx4], bins=lEbins)[0].astype(float)
                            J5 += np.histogram(lE[idx5], bins=lEbins)[0].astype(float)
                        
                        count += 1

            #simulation errors
            Jextra = J*(pao.auger.sum())/(float(J.sum()))

            Jsimerror = np.sqrt(Jextra) / dE
            J1simerror = np.sqrt(J1) / dE
            J2simerror = np.sqrt(J2) / dE
            J3simerror = np.sqrt(J3) / dE
            J4simerror = np.sqrt(J4) / dE
            J5simerror = np.sqrt(J5) / dE
            
            J /= dE
            
            # normalize
            Jsimerror /= J[0]
            J1simerror /= J[0]
            J2simerror /= J[0]
            J3simerror /= J[0]
            J4simerror /= J[0]
            J5simerror /= J[0]

            J1 /= J[0]*dE
            J2 /= J[0]*dE
            J3 /= J[0]*dE
            J4 /= J[0]*dE
            J5 /= J[0]*dE
            J /= J[0]

            msize = 3

            fig, ax = plt.subplots(figsize=(10,7))

            #J_Auger
            ax.plot(lEcens, pao.Jaugerscaled, "ok", label = 'PAO data')
            ax.errorbar(lEcens, pao.Jaugerscaled, yerr=pao.Jaugererr_scaled, fmt = "ok", markersize = msize)
            #Total
            ax.plot(lEcens, J,  color='red', label='Fitted model')
            ax.errorbar(lEcens, J, yerr=Jsimerror, fmt = "none", ecolor = 'black', color = 'none')

            #J1
            ax.plot(lEcens, J1, color='blue', label='A = 1 ({0:.2f})'.format(fracs[i][0]))
            ax.errorbar(lEcens, J1, yerr=J1simerror, fmt = "none", ecolor = 'black', color = 'blue')
            #J2
            ax.plot(lEcens, J2, color='green', label='A = 2-4 ({0:.2f})'.format(fracs[i][1]))
            ax.errorbar(lEcens, J2, yerr=J3simerror, fmt = "ok", markersize = msize, color = 'green')
            #J3
            ax.plot(lEcens, J3, color='brown', label='A = 5-22 ({0:.2f})'.format(fracs[i][2]))
            ax.errorbar(lEcens, J3, yerr=J3simerror, fmt = "ok", markersize = msize, color = 'brown')
            #J4
            ax.plot(lEcens, J4, color='purple', label='A = 23-38 ({0:.2f})'.format(fracs[i][3]))
            ax.errorbar(lEcens, J4, yerr=J4simerror, fmt = "ok", markersize = msize, color = 'purple')
            #J5
            ax.plot(lEcens, J5, color='gray', label='A $>$ 38 ({0:.2f})'.format(fracs[i][4]))
            ax.errorbar(lEcens, J5, yerr=J5simerror, fmt = "ok", markersize = msize, color = 'gray')


            plt.legend(fontsize=12, frameon=True)
            plt.semilogy()
            plt.ylim(bottom = 1.3e-6, top = 1.5)
            plt.title('$\gamma$ = {0}, $\log_{{10}}$(Rcut) = {1}, {3} = {2:.2f}'.format(g_rcut[i][0],g_rcut[i][1],tst,plot_type))
            plt.grid()
            plt.ylabel('$J(E)$ [a.u.] ({:,})'.format(num), fontsize = 18, x = 0.8)
            plt.xlabel('$\log_{10}$(E/eV)', fontsize = 18, y = 0.8)
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.savefig('{}/{}'.format(plots_dir,plotfile), dpi=300, bbox_inches='tight')
            plt.close()
            
            minutes = (time.time() - time_start)/60.
            print('\n TIME: {:4f} minutes\n'.format(minutes))
            
            flu.del_by_extension(parent_dir=outdir, exts='.dat')

