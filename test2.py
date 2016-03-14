import numpy as np
import SCF

#~ xyz = np.array([[6,    2.8846,   -0.0848,    0.0000],
                #~ [6,    1.4422,   -0.0157,    0.0000],
                #~ [6,    0.2534,    0.0412,    0.0000],
                #~ [6,   -1.1833,    0.0784,    0.0000],
                #~ [7,   -1.8685,   -1.0757,    0.0000],
                #~ [7,   -1.6977,    1.2519,    0.0000],
                #~ [1,    3.2812,   -0.0412,    1.0190],
                #~ [1,    3.3189,    0.7466,   -0.5637],
                #~ [1,    3.2345,   -1.0164,   -0.4553],
                #~ [1,   -1.4110,   -1.9746,    0.0000],
                #~ [1,   -2.8773,   -1.0298,    0.0000],
                #~ [1,   -0.9648,    1.9684,    0.0000]])

xyz = np.array([[8, 0.0,  0.000000,  0.110200],
                [1, 0.0,  0.711600, -0.440800],
                [1, 0.0, -0.711600, -0.440800]])

#~ xyz = np.array([[7, -0.10339734, -0.22895125,  0.00000000],
                #~ [1,  0.22992455, -1.17176434,  0.00000000],
                #~ [1,  0.22994176,  0.24244893,  0.81649673],
                #~ [1,  0.22994176,  0.24244893, -0.81649673]])

info = {'basis': '6-31g',
        'charge': 0,
        '2s+1': 1}

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})

hf = SCF.HF(xyz, info)
(energy, occOrbList, fockList) = hf.RunSCF()
hf.pypsi.JK_Initialize('directjk', 'cc-pvdz-jkfit')
print('energy:', energy)
print(hf.SolveFock(fockList[0])[0])


info['dft'] = 'b3lyp'
info['hfExcMix'] = 0.2
ks = SCF.KS(xyz, info, verbose=True)
ks.pypsi.Settings_SetMaxNumCPUCores(4)
ks.pypsi.JK_Initialize('dfjk', 'cc-pvdz-jkfit')
(energy, occOrbList, fockList) = ks.RunSCF()
print('energy:', energy)
print(ks.SolveFock(fockList[0])[0])

(energy, occOrbList, fockList) = ks.RunSCF(occOrbList)
print('energy:', energy)
print(ks.SolveFock(fockList[0])[0])

