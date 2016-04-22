import numpy as np
import SCF


xyz = np.array([[8, 0.0,  0.000000,  0.110200],
                [1, 0.0,  0.711600, -0.440800],
                [1, 0.0, -0.711600, -0.440800]])

info = {'basis': 'sto-3g',
        'charge': 0,
        '2s+1': 3}

info['dft'] = 'b3lyp'
info['hfExcMix'] = 0.2
ks = SCF.KSplay(xyz, info, verbose=True)
(energy, occOrbList, fockList) = ks.RunSCF()
print('energy:', energy)
print(ks.SolveFock(fockList[0])[0])


