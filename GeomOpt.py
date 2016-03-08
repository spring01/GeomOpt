import numpy as np

class GeomOpt(object):
    
    #--------------------------------------------------------------------------
    # Either info['GradFunc'] or info['EnergyFunc'] has to be provided;
    # info['GradFunc'] is preferred over info['EnergyFunc']
    # with info['EnergyFunc'] the (expensive) finite difference is invoked
    #
    # Function interfaces:
    #   'EnergyFunc':   (energy, occOrbList) = EnergyFunc(xyz, guess, info)
    #
    #   'GradFunc'  :   (energy, occOrbList, grad) = GradFunc(xyz, guess, info)
    #
    # Input:
    #   xyz:        2d np.array with shape (numAtom, 4);
    #               1st column is atomic number, other 3's are xyz coordinates
    #
    #   guess:      'core', 'sad', or list of occupied orbitals (occOrbList);
    #               see Output: occOrbList for details
    #
    #   info:       Dictionary with at least 'GradFunc' or 'EnergyFunc';
    #               can be used to pass other arguments to GradFunc/EnergyFunc
    #
    #   verbose:    True or False for printing
    #
    # Output:
    #   energy:     Double precision scalar; total scf energy
    #
    #   occOrbList: List of 1 or 2 occupied orbital matrices (wavefunction);
    #               [occOrb] for RHF/RKS, [occOrbA, occOrbB] for UHF/UKS
    #
    #   grad:       2d np.array with shape (numAtom, 3)
    #--------------------------------------------------------------------------
    def __init__(self, xyz, info, verbose=False):
        self.__xyz = xyz.copy()
        self.__info = info
        self.__verbose = verbose
        if 'GradFunc' in info:
            self.__gradFunc = info['GradFunc']
        else:
            self.__gradFunc = self.__FiniteDiff
            self.__energyFunc = info['EnergyFunc']
        self.__maxNumIter = 200
        self.__stepSize = 1.0
        self.__thresMaxGrad = 0.000450
        self.__thresRmsGrad = 0.000300
        self.__thresMaxDisp = 0.001800
        self.__thresRmsDisp = 0.001200
    
    # Homemade BFGS optimizer
    def RunGeomOpt(self, guess='sad'):
        xyz = self.__xyz
        info = self.__info
        (energy, occOrbList, grad) = self.__gradFunc(xyz, guess, info)
        hess = np.eye(grad.size)
        for numIter in range(1, self.__maxNumIter):
            if self.__verbose:
                print('geom opt iter {}; energy: {}'.format(numIter, energy))
            step = -self.__stepSize * np.linalg.solve(hess, grad.ravel())
            xyz[:, 1:] += step.reshape(xyz[:, 1:].shape)
            if self.__Converged(grad, step):
                break
            gradOld = grad
            (energy, occOrbList, grad) = self.__gradFunc(xyz, occOrbList, info)
            diffGrad = (grad - gradOld).ravel()
            hessDotStep = hess.dot(step)
            hess += np.outer(diffGrad, diffGrad) / diffGrad.dot(step)
            hess -= np.outer(hessDotStep, hessDotStep) / step.dot(hessDotStep)
        # end for
        if self.__verbose:
            print('geom opt done; energy: {}'.format(energy))
        return (xyz, energy)
    
    # SciPy's optimizer wrapper
    def RunGeomOptSciPy(self, guess='sad'):
        def Fun(coords, *args):
            xyz = args[0].copy()
            info = args[1]
            xyz[:, 1:] = coords.reshape(xyz[:, 1:].shape)
            guess = info['guess']
            (energy, info['guess'], grad) = self.__gradFunc(xyz, guess, info)
            return (energy, grad.ravel())
        class NumIter:
            numIter = 0
        def Callback(coords):
            NumIter.numIter += 1
            step = np.abs(coords - xyz[:, 1:].ravel())
            xyz[:, 1:] = coords.reshape(xyz[:, 1:].shape)
            maxDisp = np.max(np.abs(step))
            rmsDisp = np.sqrt(np.mean(step**2))
            if self.__verbose:
                form = '  {:s}: {:0.6f}'.format
                print('geom opt iter {}'.format(NumIter.numIter))
                print(form('maxDisp', maxDisp))
                print(form('rmsDisp', rmsDisp))
        # end Callback
        import scipy.optimize as opt
        xyz = self.__xyz.copy()
        info = self.__info.copy()
        info['guess'] = guess
        coordsIni = xyz[:, 1:].ravel().copy()
        optResult = opt.minimize(fun=Fun, x0=coordsIni, args=(xyz, info),
                                 method='BFGS', jac=True, callback=Callback)
        xyz[:, 1:] = optResult.x.reshape(xyz[:, 1:].shape)
        return (xyz, optResult.fun)
    
    # Finite (forward) difference approximation of gradient
    def __FiniteDiff(self, xyz, guess, info):
        (energy, occOrbList) = self.__energyFunc(xyz, guess, info)
        diff = 1.0e-6
        grad = np.zeros((xyz.shape[0], xyz.shape[1] - 1))
        for atom in range(xyz.shape[0]):
            for coord in range(1, xyz.shape[1]):
                xyzNew = xyz.copy()
                xyzNew[atom, coord] += diff
                energyNew = self.__energyFunc(xyzNew, occOrbList, info)[0]
                grad[atom, coord - 1] = (energyNew - energy) / diff
            # end For
        # end For
        return (energy, occOrbList, grad)
    
    # Test geometry optimization convergence
    def __Converged(self, gradNew, step):
        maxGrad = np.max(np.abs(gradNew))
        rmsGrad = np.sqrt(np.mean(gradNew**2))
        maxDisp = np.max(np.abs(step))
        rmsDisp = np.sqrt(np.mean(step**2))
        if self.__verbose:
            form = '  {:s}: {:0.6f}'.format
            print(form('maxGrad', maxGrad) + form('thres', self.__thresMaxGrad))
            print(form('rmsGrad', rmsGrad) + form('thres', self.__thresRmsGrad))
            print(form('maxDisp', maxDisp) + form('thres', self.__thresMaxDisp))
            print(form('rmsDisp', rmsDisp) + form('thres', self.__thresRmsDisp))
        return (maxGrad < self.__thresMaxGrad and
                rmsGrad < self.__thresRmsGrad and
                maxDisp < self.__thresMaxDisp and
                rmsDisp < self.__thresRmsDisp)
    
