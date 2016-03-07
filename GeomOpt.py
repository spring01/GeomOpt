import numpy as np

class GeomOpt(object):
    
    #--------------------------------------------------------------------------
    # Either info['GradFunc'] or info['EnergyFunc'] has to be provided
    #   info['GradFunc'] is preferred over info['EnergyFunc']
    #   because with info['EnergyFunc'] we can only do finite difference
    #
    # Function interfaces:
    #   'EnergyFunc':   (energy, occOrbList) = EnergyFunc(xyz, guess, info)
    #   'GradFunc'  :   (energy, occOrbList, grad) = GradFunc(xyz, guess, info)
    #
    # Variables:
    # Input:
    #   xyz:        2d np.array with shape (numAtom, 4);
    #               1st column is atomic number, other 3's are xyz coordinates
    #   guess:      String 'core', 'sad', or list of occupied orbitals;
    #               see occOrbList for details
    #   info:       Dictionary with at least 'GradFunc' or 'EnergyFunc';
    #               can be used to pass other arguments to GradFunc/EnergyFunc
    # Output:
    #   energy:     Double precision scalar; total scf energy
    #   occOrbList: List of 1 or 2 occupied orbital matrices (wavefunction);
    #               [occOrb] for RHF/RKS, [occOrbA, occOrbB] for UHF/UKS
    #   grad:       2d np.array with shape (numAtom, 3)
    #--------------------------------------------------------------------------
    def __init__(self, info):
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
    def RunGeomOpt(self, xyz, info):
        xyz = xyz.copy()
        (energy, occOrbList, grad) = self.__gradFunc(xyz, 'sad', info)
        hess = np.eye(grad.size)
        for numIter in xrange(self.__maxNumIter):
            print 'GeomOpt current energy:', energy
            step = -self.__stepSize * np.linalg.solve(hess, grad.ravel())
            xyz[:, 1:] += step.reshape(xyz[:, 1:].shape)
            if self.__HasConverged(grad, step):
                break
            gradOld = grad
            (energy, occOrbList, grad) = self.__gradFunc(xyz, occOrbList, info)
            diffGrad = (grad - gradOld).ravel()
            hessDotStep = hess.dot(step)
            hess += np.outer(diffGrad, diffGrad) / diffGrad.dot(step)
            hess -= np.outer(hessDotStep, hessDotStep) / step.dot(hessDotStep)
        # end for
        print 'GeomOpt done in', numIter + 1, 'iterations'
        return (xyz, energy, occOrbList)
    
    # SciPy's BFGS optimizer wrapper
    def RunGeomOptScipy(self, xyz, info):
        import scipy.optimize as opt
        def Func(coords, *args):
            xyz = args[0].copy()
            guess = args[1]
            xyz[:, 1:] = coords.reshape(xyz[:, 1:].shape)
            return self.__energyFunc(xyz, guess, info)[0]
        def Print(xk):
            print xk
        xyz = xyz.copy()
        coords0 = xyz[:, 1:].ravel().copy()
        coords = opt.fmin_bfgs(Func, coords0, args=(xyz, 'sad'), callback=Print)
        xyz[:, 1:] = coords.reshape(xyz[:, 1:].shape)
        return xyz
    
    # Finite difference approximation of gradient
    def __FiniteDiff(self, xyz, guess, info):
        (energy, occOrbList) = self.__energyFunc(xyz, guess, info)
        diff = 1.0e-6
        grad = np.zeros((xyz.shape[0], xyz.shape[1] - 1))
        for atom in xrange(xyz.shape[0]):
            for coord in xrange(1, xyz.shape[1]):
                xyzNew = xyz.copy()
                xyzNew[atom, coord] += diff
                energyNew = self.__energyFunc(xyzNew, occOrbList, info)[0]
                grad[atom, coord - 1] = (energyNew - energy) / diff
        return (energy, occOrbList, grad)
    
    # Test geometry optimization convergence
    def __HasConverged(self, gradNew, step):
        maxGrad = np.max(np.abs(gradNew))
        rmsGrad = np.sqrt(np.mean(gradNew**2))
        maxDisp = np.max(np.abs(step))
        rmsDisp = np.sqrt(np.mean(step**2))
        #~ print (maxGrad, rmsGrad, maxDisp, rmsDisp)
        return (maxGrad < self.__thresMaxGrad and
                rmsGrad < self.__thresRmsGrad and
                maxDisp < self.__thresMaxDisp and
                rmsDisp < self.__thresRmsDisp)
    
