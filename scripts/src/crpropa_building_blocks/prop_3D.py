import numpy as np
from crpropa import *

from .. import UHECRs_sim_f as cpf
from ..loaders.fields import dolag_grid
from .prop_general import set_interactions
from ..utils.coords import coordinate_transformation_handler

farthestSourceDistance = {
                        'grid': np.sqrt(3)*dolag_grid().size/2,
                        'point-like': 100.*Mpc,
                        'random': 100.*Mpc
                        }

randomCRPropa = Random.instance()

class SourceDirectionTowardsPoint(SourceFeature):
    def __init__(self, point):
        SourceFeature.__init__(self)
        self.point = point

    def __str__(self):
        return 'Direction towards {}'.format(self.point)

    def prepareCandidate(self, candidate):
        direction = self.point - candidate.source.getPosition()  # Vector.Head - Vector.Tail
        candidate.source.setDirection(direction.getUnitVector())
        SourceFeature.prepareCandidate(self, candidate)

class SourceDirectedvMFTowardsPoint(SourceFeature):
    def __init__(self, point, kappa):
        SourceFeature.__init__(self)
        self.point = point
        self.kappa = kappa

    def __str__(self):
        return 'Direction towards {} using vMF distribution (kappa = {:.2e})'.format(self.point, self.kappa)    

    def prepareCandidate(self, candidate):
        mu = self.point - candidate.source.getPosition()  # Vector.Head - Vector.Tail
        mu = mu.getUnitVector()

        direction = randomCRPropa.randFisherVector(mu, self.kappa)
        direction = direction.getUnitVector()
        candidate.source.setDirection(direction)

        pdfVonMises = self.kappa / (2. * M_PI * (1. - np.exp(-2. * self.kappa))) * np.exp(-self.kappa * (1. - direction.dot(mu)))
        weight = 1. / (4. * M_PI * pdfVonMises)
        candidate.setWeight(weight)

        SourceFeature.prepareCandidate(self, candidate)

def set_simulation(sim, model=IRB_Gilmore12(), interactions=True, barProgress=True, **kwargs):
    sim.add( PropagationBP(kwargs['field'], kwargs['tolerance'], kwargs['minStep'], kwargs['maxStep']) )
    sim.add( Redshift() )

    if interactions: set_interactions(sim, model)

    # sim.add( MinimumEnergy(cpf.stopE) )
    # sim.add( MaximumTrajectoryLength(cpf.maxTrajectoryL) )
    sim.setShowProgress(barProgress)

def source_template(spectrumStr, nucleiFracs):
    sourceTemplate = SourceGenericComposition(cpf.minE, cpf.maxE, spectrumStr, 10**5)
    for z in nucleiFracs:
        nuclearCode = nucleusId(cpf.A_Z[z][0], cpf.A_Z[z][1])
        sourceTemplate.add(nuclearCode, nucleiFracs[z])

    return sourceTemplate

def set_sources_handler(srcType, **kwargs):
    if srcType == 'point-like':
        return point_like_sources(**kwargs)
    elif srcType == 'grid':
        return grid_like_sources(**kwargs)
    elif srcType == 'random':
        return random_sources(**kwargs)

def random_sources(spectrumStr, nucleiFracs, rigidityLimits=False):
    sourceTemplate = source_template(spectrumStr=spectrumStr, nucleiFracs=nucleiFracs)

    source_module = Source()
    source_module.add( sourceTemplate )
    source_module.add( SourceUniformHollowSphere(Vector3d(0), 10.*Mpc, 100.*Mpc) )

    return source_module 

def grid_like_sources(srcPath, spectrumStr, nucleiFracs, rigidityLimits=False):
    sourceTemplate = source_template(spectrumStr=spectrumStr, nucleiFracs=nucleiFracs)

    # Source Density Grid
    dolagConfig = dolag_grid()
    mgrid = Grid1f( dolagConfig.boxOrigin, dolagConfig.gridSize, dolagConfig.spacing )
    loadGrid(mgrid, str(srcPath))

    source_module = Source()
    source_module.add( sourceTemplate )
    source_module.add( SourceDensityGrid( mgrid ) )

    return source_module

def point_like_sources(srcPath, spectrumStr, nucleiFracs, Coords, kappa=None, rigidityLimits=False):
    sourcesData = np.genfromtxt(srcPath, names=True)
    farthestSourceDistance['point-like'] = 1.01 * sourcesData['Distance'].max() * Mpc
    redshiftPresent = 'Redshift' in sourcesData.dtype.names    
    redshifts = sourcesData['Redshift'] if redshiftPresent else None
    sourcesPos = coordinate_transformation_handler(sourcesData, Coords)

    if kappa is None:
        source_emission = lambda direction: SourceDirection( direction )
    else:
        source_emission = lambda direction: SourceDirectedEmission( direction, kappa )

    if Coords == 'spherical':
        set_vector_position = lambda v, s: v.setRThetaPhi(s['R'], s['Theta'], s['Phi'])
    else:
        set_vector_position = lambda v, s: v.setXYZ(s['X'], s['Y'], s['Z'])


    source_module = SourceList()
    set_sources_by_limits(
                        rigidityLimits=rigidityLimits,
                        sourcesPos=sourcesPos,
                        source_module=source_module,
                        emission_func=source_emission,
                        vec_pos_func=set_vector_position,
                        spectrumStr=spectrumStr,
                        nucleiFracs=nucleiFracs,
                        redshifts=redshifts
                        )

    return source_module 

def set_sources_by_limits(rigidityLimits=False, **kwargs):
    if rigidityLimits:
        set_sources_rigidity_limits(**kwargs)
    else:
        set_sources_energy_limits(**kwargs)

def set_sources_energy_limits(sourcesPos, source_module, emission_func, vec_pos_func, spectrumStr, nucleiFracs, redshifts=None):
    sourceTemplate = source_template(spectrumStr=spectrumStr, nucleiFracs=nucleiFracs)

    for i,sourcePos in enumerate(sourcesPos):
        s = Source()
        s.add(sourceTemplate)
        v = Vector3d()
        vec_pos_func(v, sourcePos)
        s.add(SourcePosition(v * Mpc))
        s.add(emission_func(v.getUnitVector() * (-1.)))
        if redshifts is not None: s.add( SourceRedshift(redshifts[i]) )
        source_module.add(s, 1)

def set_sources_rigidity_limits(sourcesPos, source_module, emission_func, vec_pos_func, spectrumStr, nucleiFracs, redshifts=None):
    for nucleus in nucleiFracs:
        A, Z = cpf.A_Z[nucleus]
        sourceTemplate = SourceGenericComposition(Z*cpf.minE, Z*cpf.maxE, spectrumStr, 10**5)
        nuclearCode = nucleusId(A, Z)
        sourceTemplate.add(nuclearCode, 1)

        for i,sourcePos in enumerate(sourcesPos):
            s = Source()
            s.add(sourceTemplate)
            v = Vector3d()
            vec_pos_func(v, sourcePos)
            s.add(SourcePosition(v * Mpc))
            s.add(emission_func(v.getUnitVector() * (-1.)))
            if redshifts is not None: s.add( SourceRedshift(redshifts[i]) )
            source_module.add(s, nucleiFracs[nucleus])