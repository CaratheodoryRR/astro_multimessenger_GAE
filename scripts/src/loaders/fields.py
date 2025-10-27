from crpropa import *
from pydataclasses import DataClass

class DolagGrid(DataClass):
    # magnetic field setup
    h = 0.677
    gridSize = 440
    size = 80*Mpc/h
    spacing = size/(gridSize)
    boxOrigin = Vector3d(-0.5*size)

class HacksteinGrid(DataClass):
    # magnetic field setup
    gridSize = 1024                          # Size of uniform grid in data points
    h = 0.677                                # Dimensionless Hubble parameter
    size = 249.827/h *Mpc                    # Physical edgelength of volume in Mpc
    spacing = size/(gridSize)                # Resolution, physical size of single cell
    boxOrigin = Vector3d(-0.5*size)          # Origin of the full box of the simulation

def setting_dolag_field(pathToDolag, bFactor):
    dolagConfig = DolagGrid()

    vgrid = Grid3f(dolagConfig.boxOrigin, dolagConfig.gridSize, dolagConfig.spacing)
    loadGrid(vgrid, str(pathToDolag), bFactor)

    dolag_field = MagneticFieldGrid(vgrid)

    return dolag_field 

def setting_hackstein_field(pathToHackstein, bFactor):
    hacksteinConfig = HacksteinGrid()

    vgrid = Grid3f(hacksteinConfig.boxOrigin, hacksteinConfig.gridSize, hacksteinConfig.spacing)
    loadGrid(vgrid, str(pathToHackstein), bFactor)

    hackstein_field = MagneticFieldGrid(vgrid)

    return hackstein_field 

def setting_jf12_field(striated=True, turbulent=True):
    # magnetic field setup
    jf12_field = JF12Field()
    if striated:
        jf12_field.randomStriated()
    if turbulent:
        jf12_field.randomTurbulent()

    return jf12_field

def setting_uf23_field(modelIdx=0, striated=True, turbulent=True):
    # magnetic field setup
    planck_jf12_random = PlanckJF12bField()
    planck_jf12_random.setUseRegularField(False) # Deactivate regular/coherent field

    uf23_coherent = UF23Field(modelIdx) # UF23 regular modelIdx model (default: base)

    if striated:
        planck_jf12_random.randomTurbulent() # Activate random turbulent field
    if turbulent:
        planck_jf12_random.randomStriated() # Activate random striated field

    # Superposition of fields
    gmf = MagneticFieldList()
    gmf.addField(uf23_coherent)
    if striated or turbulent:
        gmf.addField(planck_jf12_random)

    return gmf