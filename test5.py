import numpy as np
import SCF

xyz = np.array([[24,    0.000000,   5.488000,   1.911000],
                [ 8,    0.000000,   4.137000,   2.851000],
                [ 8,    0.000000,   6.839000,   2.851000],
                [ 8,    1.351000,   5.488000,   0.971000],
                [ 8,   -1.351000,   5.488000,   0.971000]])

info = {'dft': 'b3lyp',
        'basis': 'lanl2dz',
        'charge': 0,
        '2s+1': 1}

hf = SCF.G09SCF(xyz, info, verbose=True)
(energy, occOrbList, fockList) = hf.RunSCF()
print('energy:', energy)
print(hf.SolveFock(fockList[0])[0])


