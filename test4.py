import numpy as np
import SCF

xyz = np.array([[8, 0.0,  0.000000,  0.110200],
                [1, 0.0,  0.711600, -0.440800],
                [1, 0.0, -0.711600, -0.440800]])


#~ xyz = np.array([[24,  0.00000000,   0.00000000,   0.40000000],
                #~ [6 ,  0.00000000,   0.00000000,  -1.60000000]])

info = {'basis': '6-31g*',
        'charge': 0,
        '2s+1': 1}

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})



info['dft'] = 'b3lyp'
info['hfExcMix'] = 0.2
ks = SCF.KS(xyz, info, verbose=True)
ks.pypsi.Settings_SetMaxNumCPUCores(4)

occOrbListGuess = ks._HF__GuessSAD()
densGuess = ks._HF__OccOrbToDens(occOrbListGuess)

(energy, occOrbList, fockList) = ks.RunSCF(guess=occOrbListGuess)
dens = ks._HF__OccOrbToDens(occOrbList)

print(np.abs(dens[0].max()), np.abs(densGuess[0].max()))

diff = (densGuess[0] - dens[0]).ravel()
print(np.abs(diff.min()), np.abs(diff.max()), diff.std())
print(np.linalg.norm(diff))


