import numpy as np
from PyPsi import PyPsi
from .LCIIS import LCIIS

class HF(object):
    
    def __init__(self, xyz, info, verbose=False):
        self.__verbose = verbose
        pypsi = PyPsi(xyz, info['basis'], info['charge'], info['2s+1'])
        self.pypsi = pypsi
        numElecTotal = pypsi.Molecule_NumElectrons()
        self.__numElecAB = self.__NumElecAB(numElecTotal, info['2s+1'])
        pypsi.SCF_SetSCFType({1: 'rhf', 2: 'uhf'}[len(set(self.__numElecAB))])
        
        self.__nucRepEnergy = pypsi.Molecule_NucRepEnergy()
        self.__overlap = pypsi.Integrals_Overlap()
        self.__toOr = self.__ToOrtho(self.__overlap)
        self.__coreH = pypsi.Integrals_Kinetic() + pypsi.Integrals_Potential()
        
        self.__maxSCFIter = 200
        self.__thresRmsDiffDens = 1.0e-8
        self.__thresMaxDiffDens = 1.0e-6
        self.__thresDiffEnergy = 1.0e-6
    
    # Perform SCF calculation
    #   guess: 'core', 'sad', or occOrbList
    def RunSCF(self, guess='sad'):
        if type(guess) is str:
            guessDict = {'core': self.__GuessCore, 'sad': self.__GuessSAD}
            occOrbList = guessDict[guess.lower()]()
        elif type(guess) is list:
            occOrbList = guess
        densList = self.__OccOrbToDens(occOrbList)
        lciis = LCIIS(self.__overlap, self.__toOr, self.__verbose)
        energy = 0.0
        for numIter in range(1, self.__maxSCFIter):
            oldEnergy = energy
            (fockList, energy) = self._FockEnergy(occOrbList, densList)
            fockList = lciis.NewFock(fockList, densList)
            oldDensList = [dens.copy() for dens in densList]
            occOrbList = self.__FockToOccOrb(fockList)
            densList = self.__OccOrbToDens(occOrbList)
            if self.__verbose:
                print('scf iter {}; energy: {}'.format(numIter, energy))
            if self.__Converged(densList, oldDensList, energy, oldEnergy):
                break
        # end for
        if self.__verbose:
            print('scf done; energy: {}'.format(energy))
        return (energy, occOrbList, fockList)
    
    # Solve a Fock matrix by transforming to a orthogonalized basis then eigh
    def SolveFock(self, fock):
        orFock = self.__toOr.T.dot(fock).dot(self.__toOr)
        (orbEigVal, orOrb) = np.linalg.eigh(orFock)
        argsort = np.argsort(orbEigVal)
        return (orbEigVal[argsort], self.__toOr.dot(orOrb[:, argsort]))
    
    # This protected function is overloaded in class KS
    # Construct a list of Fock matrix and calculate energy
    def _FockEnergy(self, occOrbList, densList):
        self.pypsi.JK_CalcAllFromOccOrb(occOrbList)
        couList = self.pypsi.JK_GetJ()
        excList = self.pypsi.JK_GetK()
        fockList = [self._FockNoExc(couList) - exc for exc in excList]
        return (fockList, self._HFEnergy(fockList, densList))
    
    # These helper functions are not overloaded but is used in KS
    def _HFEnergy(self, fockList, densList):
        return (self.__nucRepEnergy +
                np.mean([(fock + self.__coreH).ravel().dot(dens.ravel())
                          for (fock, dens) in zip(fockList, densList)]))
    
    def _FockNoExc(self, couList):
        return self.__coreH + sum(couList) * (2.0 / len(couList))
    
    # Find number of alpha/beta electrons
    def __NumElecAB(self, numElecTotal, multiplicity):
        numElecA = (numElecTotal + multiplicity - 1) / 2.0
        if numElecA % 1 != 0.0:
            raise Exception('numElecTotal and multiplicity do not agree')
        return [int(numElecA), int(numElecTotal - numElecA)]
    
    # Find a transform from atomic orbitals to orthogonal orbitals
    def __ToOrtho(self, overlap):
        (eigVal, eigVec) = np.linalg.eigh(overlap)
        keep = eigVal > 1.0e-6
        return eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]
    
    # These guess functions return occOrbList
    def __GuessCore(self):
        return self.__FockToOccOrb([self.__coreH] * len(set(self.__numElecAB)))
    
    def __GuessSAD(self):
        self.pypsi.SCF_SetGuessType('sad')
        sadDens = self.pypsi.SCF_GuessDensity()
        return [self.__DensToFakeOccOrb(sadDens)] * len(set(self.__numElecAB))
    
    # Convert a density matrix to a (fake) occupied orbital matrix (not a list)
    def __DensToFakeOccOrb(self, dens):
        (eigVal, eigVec) = np.linalg.eigh(dens)
        keep = eigVal > np.finfo(float).eps
        return eigVec[:, keep] * np.sqrt(eigVal[keep])[None, :]
    
    # Solve Fock matrix to get occOrbList
    def __FockToOccOrb(self, fockList):
        orbList = [self.SolveFock(fock)[1] for fock in fockList]
        return [orb[:, :ne] for (orb, ne) in zip(orbList, self.__numElecAB)]
    
    # Calculate list of density matrices from list of occupied orbital sets
    def __OccOrbToDens(self, occOrbList):
        return [occOrb.dot(occOrb.T) for occOrb in occOrbList]
    
    # Test scf convergence
    def __Converged(self, densList, oldDensList, energy, oldEnergy):
        zipList = zip(densList, oldDensList)
        diffDens = np.concatenate([dens - old for (dens, old) in zipList])
        diffEnergy = np.abs(energy - oldEnergy)
        rmsDiffDens = np.sqrt(np.mean(diffDens**2))
        maxDiffDens = np.max(np.abs(diffDens))
        if self.__verbose:
            form = '  {:s}: {:.3e}'.format
            print (form('rmsDiffDens', rmsDiffDens) +
                   form('thres', self.__thresRmsDiffDens))
            print (form('maxDiffDens', maxDiffDens) +
                   form('thres', self.__thresMaxDiffDens))
            print (form('diffEnergy ', diffEnergy) +
                   form('thres', self.__thresDiffEnergy))
        return (rmsDiffDens < self.__thresRmsDiffDens and
                maxDiffDens < self.__thresMaxDiffDens and
                diffEnergy < self.__thresDiffEnergy)
    



