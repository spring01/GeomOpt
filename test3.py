import numpy as np
import SCF
from GeomOpt import GeomOpt

def EnergyFunc(xyz, guess, info):
    newInfo = info.copy()
    newInfo['xyz'] = xyz
    hf = SCF.HF(newInfo)
    hf.pypsi.Settings_SetMaxNumCPUCores(4)
    hf.pypsi.JK_Initialize('dfjk', 'cc-pvdz-jkfit')
    return hf.RunSCF(guess)[0:2]

np.set_printoptions(precision=3)

xyz = np.array([[8, 0.0,  0.000000,  0.110200], 
                [1, 0.0,  0.711600, -0.440800],
                [1, 0.0, -0.711600, -0.440800]])

info = {'xyz': xyz,
        'basis': 'cc-pvdz',
        'charge': 0,
        '2s+1': 1,
        'EnergyFunc': EnergyFunc}

hf = SCF.HF(info)

geomOpt = GeomOpt(info)
print geomOpt.RunGeomOpt(info)


