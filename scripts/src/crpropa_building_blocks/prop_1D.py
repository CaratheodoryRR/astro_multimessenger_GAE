from crpropa import *
from .. import UHECRs_sim_f as cpf
from .prop_general import set_interactions

def set_simulation(sim, model=IRB_Gilmore12(), interactions=True, **kwargs):
    
    sim.add( SimplePropagation(kwargs['minStep'], kwargs['maxStep']) )
    sim.add( Redshift() )
    
    if interactions: set_interactions(sim, model)
    
    sim.add( MinimumEnergy(cpf.stopE) )

def set_sources_handler(rigidityLimits=False, **kwargs):
    if rigidityLimits:
        set_sources_rigidity_limits(**kwargs)
    else:
        set_sources_energy_limits(**kwargs)


def set_sources_energy_limits(sources, source_list, spectrumStr, nucleiFracs):
    
    source_template = SourceGenericComposition(cpf.minE, cpf.maxE, spectrumStr, 10**5)
    for z in nucleiFracs:
        nuclearCode = nucleusId(cpf.A_Z[z][0], cpf.A_Z[z][1])
        source_template.add(nuclearCode, nucleiFracs[z])
        
    for distance, weight in sources:
        s = Source()
        s.add( source_template )
        s.add( SourcePosition(distance * Mpc) )
        s.add( SourceDirection() )
        s.add( SourceRedshift1D() )
        source_list.add(s, weight)

def set_sources_rigidity_limits(sources, source_list, spectrumStr, nucleiFracs):
    
    for nucleus in nucleiFracs:
        A, Z = cpf.A_Z[nucleus]
        source_template = SourceGenericComposition(Z*cpf.minE, Z*cpf.maxE, spectrumStr, 10**5)
        nuclearCode = nucleusId(A, Z)
        source_template.add(nuclearCode, 1)
        
        for distance, weight in sources:
            s = Source()
            s.add( source_template )
            s.add( SourcePosition(distance * Mpc) )
            s.add( SourceDirection() )
            s.add( SourceRedshift1D() )
            source_list.add(s, weight*nucleiFracs[nucleus])