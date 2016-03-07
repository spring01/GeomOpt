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
    #   xyz:        2d np.array with shape (numAtom, 4);
    #               1st column has atomic numbers other 3 are xyz coordinates
    #   occOrbList: list of 1 or 2 occupied orbital matrices
    #               [occOrb] for RHF/RKS, [occOrbA, occOrbB] for UHF/UKS
    #   grad:       2d np.array with shape (numAtom, 3)
    #   guess:      'core', 'sad', or occOrbList
    #   info:       dictionary with at least {'energy': Energy}
    #               can be used to pass other arguments to GradFunc/EnergyFunc
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
    
    # BFGS optimizer
    def RunGeomOpt(self, info):
        xyz = info['xyz'].copy()
        (energy, occOrbList, grad) = self.__gradFunc(xyz, 'sad', info)
        hess = np.eye(grad.size)
        for numIter in xrange(self.__maxNumIter):
            print 'geometry optimization:', numIter
            step = -self.__stepSize * np.linalg.solve(hess, grad.ravel())
            xyz[:, 1:] += step.reshape(xyz[:, 1:].shape)
            (energy, occOrbList, gradNew) =\
                self.__gradFunc(xyz, occOrbList, info)
            diffGrad = (gradNew - grad).ravel()
            hessDotStep = hess.dot(step)
            hess += np.outer(diffGrad, diffGrad) / diffGrad.dot(step)
            hess -= np.outer(hessDotStep, hessDotStep) / step.dot(hessDotStep)
            if self.__HasConverged(gradNew, step):
                break
            grad = gradNew
        # end for
        print 'geometry optimization done in', numIter + 1, 'iterations'
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
    
