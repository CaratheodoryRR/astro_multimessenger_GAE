from crpropa import *
from pydataclasses import DataClass

class dolag_grid(DataClass):
    # magnetic field setup
    gridSize = 440
    size = 186*Mpc
    spacing = size/(gridSize)
    boxOrigin = Vector3d(-0.5*size)


def setting_dolag_field(pathToDolag, bFactor):
    
    dolagConfig = dolag_grid()
    
    vgrid = Grid3f(dolagConfig.boxOrigin, dolagConfig.gridSize, dolagConfig.spacing)
    loadGrid(vgrid, str(pathToDolag), bFactor)
    
    dolag_field = MagneticFieldGrid(vgrid)
    
    return dolag_field 

def setting_jf12_field(striated=True, turbulent=True):
    # magnetic field setup
    jf12_field = JF12Field()
    if striated:
        jf12_field.randomStriated()
    if turbulent:
        jf12_field.randomTurbulent()
    
    return jf12_field