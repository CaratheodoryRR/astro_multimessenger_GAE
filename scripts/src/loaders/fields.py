from crpropa import *

def setting_dolag_field(pathToDolag, bFactor):
    # magnetic field setup
    gridSize = 440
    size = 186*Mpc
    spacing = size/(gridSize)
    boxOrigin = Vector3d(-0.5*size, -0.5*size, -0.5*size) 

    vgrid = Grid3f(boxOrigin, gridSize, spacing)
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