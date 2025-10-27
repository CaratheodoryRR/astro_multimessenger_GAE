import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from crpropa import Vector3d

state = coord.galactocentric_frame_defaults.get_from_registry("latest")
state["parameters"]["z_sun"] = 0
state["parameters"]["galcen_v_sun"] = (0, 0, 0)
state["parameters"]["galcen_distance"] = 8.5*u.kpc

coord.galactocentric_frame_defaults.register(name="JF12", **state)

_ = coord.galactocentric_frame_defaults.set("JF12")

# Galactic center in Supergalactic (SG) coordinates
galCenGC = coord.Galactocentric(
                                x=0*u.m,
                                y=0*u.m,
                                z=0*u.m,
                                representation_type=coord.CartesianRepresentation
                                )
galCenSG = galCenGC.transform_to(coord.Supergalactic())
galCenSGVec = Vector3d(*galCenSG.cartesian.xyz.value)

def collector_coord_transformation(collector):
    positions = [candidate.current.getPosition() for candidate in collector]
    velocities = [candidate.current.getVelocity() for candidate in collector]
    currentCoords = coord.Supergalactic(
                                        sgx=[pos.x for pos in positions]*u.m,
                                        sgy=[pos.y for pos in positions]*u.m,
                                        sgz=[pos.z for pos in positions]*u.m,
                                        v_x=[vel.x for vel in velocities]*u.m/u.s,
                                        v_y=[vel.y for vel in velocities]*u.m/u.s,
                                        v_z=[vel.z for vel in velocities]*u.m/u.s,
                                        representation_type=coord.CartesianRepresentation,
                                        differential_type=coord.CartesianDifferential
                                    )

    newCoords = currentCoords.transform_to(coord.Galactocentric())

    newPos = newCoords.cartesian
    newDir = newCoords.velocity/newCoords.velocity.norm()

    posGen = (Vector3d(*xyz) for xyz in newPos.xyz.value.T)
    dirGen = (Vector3d(*d_xyz) for d_xyz in newDir.d_xyz.value.T)
    for (candidate, posVec, dirVec) in zip(collector, posGen, dirGen):
        candidate.current.setPosition(posVec)
        candidate.current.setDirection(dirVec)

def coordinate_transformation_handler(sourcePositions, coords):
    if coords == 'galactic':
        sourcePositions = from_galactic_to_cartesian(sourcePositions)
    elif coords == 'icrs':
        sourcePositions = from_icrs_to_cartesian(sourcePositions)
    elif coords == 'supergalactic':
        sourcePositions = from_supergalactic_to_cartesian(sourcePositions)
    return sourcePositions

def to_named_cartesian(cartCoords):
    cartCoords = [tuple(x) for x in cartCoords]

    return np.array(cartCoords, dtype=[('X', np.float64), ('Y', np.float64), ('Z', np.float64)])

def from_galactic_to_cartesian(sourcePositions):
    galCoords = coord.SkyCoord(sourcePositions['Longitude']*u.radian,
                                sourcePositions['Latitude']*u.radian,
                                sourcePositions['Distance']*u.Mpc,
                                frame='galactic')
    cartCoords = to_named_cartesian(galCoords.icrs.cartesian.xyz.value.T)

    return cartCoords

def from_supergalactic_to_cartesian(sourcePositions):
    sgalCoords = coord.SkyCoord(sourcePositions['SLongitude']*u.radian,
                                sourcePositions['SLatitude']*u.radian,
                                sourcePositions['Distance']*u.Mpc,
                                frame='supergalactic')
    cartCoords = to_named_cartesian(sgalCoords.icrs.cartesian.xyz.value.T)

    return cartCoords

def from_icrs_to_cartesian(sourcePositions):
    icrsCoords = coord.SkyCoord(sourcePositions['RA']*u.radian,
                                sourcePositions['DEC']*u.radian,
                                sourcePositions['Distance']*u.Mpc,
                                frame='icrs')
    cartCoords = to_named_cartesian(icrsCoords.cartesian.xyz.value.T)

    return cartCoords