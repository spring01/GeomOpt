import numpy as np
import SCF
from GeomOpt import GeomOpt

def EnergyFunc(xyz, guess, info):
    hf = SCF.HF(xyz, info)
    hf.pypsi.Settings_SetMaxNumCPUCores(4)
    hf.pypsi.JK_Initialize('dfjk', 'cc-pvdz-jkfit')
    return hf.RunSCF(guess)[0:2]

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

#~ xyz = np.array([[8, 0.0,  0.000000,  0.110200],
                #~ [1, 0.0,  0.711600, -0.440800],
                #~ [1, 0.0, -0.711600, -0.440800]])

xyz = np.array([[7, -0.10339734, -0.22895125,  0.00000000],
                [1,  0.22992455, -1.17176434,  0.00000000],
                [1,  0.22994176,  0.24244893,  0.81649673],
                [1,  0.22994176,  0.24244893, -0.81649673]])

info = {'basis': '6-31g',
        'charge': 0,
        '2s+1': 1,
        'EnergyFunc': EnergyFunc}

hf = SCF.HF(xyz, info)

geomOpt = GeomOpt(xyz, info, True)
print(geomOpt.RunGeomOpt()[0])
geomOpt = GeomOpt(xyz, info, True)
print(geomOpt.RunGeomOptSciPy()[0])


