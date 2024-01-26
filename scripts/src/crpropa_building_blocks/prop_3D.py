from crpropa import *
from .prop_general import set_interactions
from .. import UHECRs_sim_f as cpf


def set_simulation(sim, model=IRB_Gilmore12(), interactions=True, **kwargs):
    sim.add( PropagationBP(**kwargs) )
    sim.add( Redshift() )
    
    if interactions: set_interactions(sim, model)
    
    sim.add( MinimumEnergy(cpf.stopE) )
    
def set_sources_handler(rigidityLimits=False, **kwargs):
    if rigidityLimits:
        set_sources_rigidity_limits(**kwargs)
    else:
        set_sources_energy_limits(**kwargs)

def set_sources_energy_limits(sources, source_list, emission_func, vec_pos_func, spectrumStr, nucleiFracs, redshifts=None):
    
    source_template = SourceGenericComposition(cpf.minE, cpf.maxE, spectrumStr, 10**5)
    for z in nucleiFracs:
        nuclearCode = nucleusId(cpf.A_Z[z][0], cpf.A_Z[z][1])
        source_template.add(nuclearCode, nucleiFracs[z])
    
    #sourcesWrapper = zip(sources, redshifts) if (redshifts is not None) else sources
    for i,source in enumerate(sources):
        s = Source()
        s.add(source_template)
        v = Vector3d()
        vec_pos_func(v, source)
        s.add(SourcePosition(v * Mpc))
        s.add(emission_func(v.getUnitVector() * (-1.)))
        if redshifts is not None: s.add( SourceRedshift(redshifts[i]) )
        source_list.add(s, 1)

def set_sources_rigidity_limits(sources, source_list, emission_func, vec_pos_func, spectrumStr, nucleiFracs, redshifts=None):
    
    for nucleus in nucleiFracs:
        A, Z = cpf.A_Z[nucleus]
        source_template = SourceGenericComposition(Z*cpf.minE, Z*cpf.maxE, spectrumStr, 10**5)
        nuclearCode = nucleusId(A, Z)
        source_template.add(nuclearCode, 1)
        
        for i,source in enumerate(sources):
            s = Source()
            s.add(source_template)
            v = Vector3d()
            vec_pos_func(v, source)
            s.add(SourcePosition(v * Mpc))
            s.add(emission_func(v.getUnitVector() * (-1.)))
            if redshifts is not None: s.add( SourceRedshift(redshifts[i]) )
            source_list.add(s, nucleiFracs[nucleus])