import numpy as np
from PyPsi import PyPsi
from LCIIS import LCIIS

class HF(object):
    
    def __init__(self, xyz, info):
        pypsi = PyPsi(xyz, info['basis'], info['charge'], info['2s+1'])
        self.pypsi = pypsi
        numElecTotal = pypsi.Molecule_NumElectrons()
        self.__numElecAB = self.__NumElecAB(numElecTotal, info['2s+1'])
        pypsi.SCF_SetSCFType({1: 'rhf', 2: 'uhf'}[len(set(self.__numElecAB))])
        
        self.__overlap = pypsi.Integrals_Overlap()
        self.__toOr = self.__ToOrtho(self.__overlap)
        self.__coreH = pypsi.Integrals_Kinetic() + pypsi.Integrals_Potential()
        self.__nucRepEnergy = pypsi.Molecule_NucRepEnergy()
        
        self.__maxSCFIter = 200
        self.__thresRmsDiffDens = 1.0e-8
        self.__thresMaxDiffDens = 1.0e-6
        self.__thresDiffEnergy = 1.0e-6
    
    # Perform SCF calculation
    #   guessType: 'core' or 'sad', case insensitive
    def RunSCF(self, guess='sad'):
        if type(guess) is str:
            guessDict = {'core': self.__GuessCore, 'sad': self.__GuessSAD}
            occOrbList = guessDict[guess.lower()]()
        elif type(guess) is list:
            occOrbList = guess
        densList = self.__OccOrbToDens(occOrbList)
        lciis = LCIIS(self.__overlap, self.__toOr)
        energy = 0.0
        for numIter in xrange(self.__maxSCFIter):
            #~ print 'scf:', numIter
            oldEnergy = energy
            fockList = self._OccOrbToFock(occOrbList)
            energy = self._SCFEnergy(fockList, densList)
            lciis.Enqueue(fockList, densList)
            fockList = lciis.NewFock()
            oldDensList = [dens.copy() for dens in densList]
            occOrbList = self.__FockToOccOrb(fockList)
            densList = self.__OccOrbToDens(occOrbList)
            if self.__HasConverged(densList, oldDensList, energy, oldEnergy):
                break
        # end for
        #~ print 'done scf', numIter
        return (energy + self.__nucRepEnergy, occOrbList, fockList)
    
    # Solve a Fock matrix by transforming to a orthogonalized basis then eigh
    def SolveFock(self, fock):
        orFock = self.__toOr.T.dot(fock).dot(self.__toOr)
        (orbEigVal, orOrb) = np.linalg.eigh(orFock)
        argsort = np.argsort(orbEigVal)
        return (orbEigVal[argsort], self.__toOr.dot(orOrb[:, argsort]))
    
    
    # These protected functions are overloaded in class KS
    # Construct a Fock matrix from occOrbList
    def _OccOrbToFock(self, occOrbList):
        self.pypsi.JK_CalcAllFromOccOrb(occOrbList)
        couList = self.pypsi.JK_GetJ()
        excList = self.pypsi.JK_GetK()
        return [self._FockNoExc(couList) - exc for exc in excList]
    
    # Calculate SCF energy; core repelling energy is not included
    def _SCFEnergy(self, fockList, densList):
        hamList = [(fock + self.__coreH) for fock in fockList]
        return np.mean([ham.ravel().dot(dens.ravel())
                    for (ham, dens) in zip(hamList, densList)])
    
    # This helper function is not overloaded but is used in KS
    def _FockNoExc(self, couList):
        return self.__coreH + sum(couList) * (2.0 / len(couList))
    
    # Find number of alpha/beta electrons
    def __NumElecAB(self, numElecTotal, multiplicity):
        numElecA = (numElecTotal + multiplicity - 1) / 2.0
        if numElecA % 1 != 0.0:
            raise Exception('numElecTotal and multiplicity do not agree')
        return [int(numElecA), int(numElecTotal - numElecA)]
    
    # Find a transform from atomic orbital to orthogonal orbital
    def __ToOrtho(self, overlap):
        (eigVal, eigVec) = np.linalg.eigh(overlap)
        keep = eigVal >= 1.0e-6
        return eigVec[:, keep].dot(np.diag(1.0 / np.sqrt(eigVal[keep])))
    
    # These guess functions return a tuple (occOrbList, densList)
    def __GuessCore(self):
        return self.__FockToOccOrb([self.__coreH] * len(set(self.__numElecAB)))
    
    def __GuessSAD(self):
        def DensToOccOrb(dens):
            (eigVal, eigVec) = np.linalg.eigh(dens)
            eps = np.finfo(float).eps
            return eigVec[:, eigVal > eps] * np.sqrt(eigVal[eigVal > eps])
        self.pypsi.SCF_SetGuessType('sad')
        sadGuessDens = self.pypsi.SCF_GuessDensity()
        return [DensToOccOrb(sadGuessDens)] * len(set(self.__numElecAB))
    
    # Solve Fock matrix to get (occOrbList, densList)
    def __FockToOccOrb(self, fockList):
        orbList = [self.SolveFock(fock)[1] for fock in fockList]
        return [orb[:, :ne] for (orb, ne) in zip(orbList, self.__numElecAB)]
    
    # Calculate list of density matrices from list of occupied orbital sets
    def __OccOrbToDens(self, occOrbList):
        return [occOrb.dot(occOrb.T) for occOrb in occOrbList]
    
    # Test scf convergence
    def __HasConverged(self, densList, oldDensList, energy, oldEnergy):
        diffDens = np.concatenate(
            [dens - oldDens for (dens, oldDens) in zip(densList, oldDensList)])
        diffEnergy = energy - oldEnergy
        rmsDiffDens = np.sqrt(np.mean(diffDens**2))
        maxDiffDens = np.max(np.abs(diffDens))
        #~ print (rmsDiffDens, maxDiffDens, diffEnergy)
        return (rmsDiffDens < self.__thresRmsDiffDens and
                maxDiffDens < self.__thresMaxDiffDens and
                diffEnergy < self.__thresDiffEnergy)
    
    
    
    
    
    
    
    
    
    
    
    

