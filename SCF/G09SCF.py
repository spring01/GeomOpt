import os
import re
import struct
import tempfile
import numpy as np
from .LCIIS import LCIIS

class G09SCF(object):
    
    def __init__(self, xyz, info, verbose=False):
        self.__verbose = verbose
        self.__info = info.copy()
        self.__info['xyz'] = xyz.copy()
        
        self.__workPath = tempfile.mkdtemp()
        self.__workGjf = self.__workPath + '/run.gjf'
        self.__workLog = self.__workPath + '/run.log'
        self.__workDat = self.__workPath + '/run.dat'
        self.__RunG09('iop(4/199=1) guess=harris')
        with open(self.__workLog, 'r') as fLog:
            for line in fLog:
                if 'alpha electrons' in line:
                    numElecABStr = re.findall(r'[\d]+', line)
                    break
        self.__numElecAB = [int(num) for num in numElecABStr]
        with open(self.__workDat, 'rb') as fDat:
            nbf = int(np.sqrt(struct.unpack('i', fDat.read(4))[0] / 8.0))
            shape = (nbf, nbf)
            self.__overlap = np.fromfile(fDat, float, nbf**2).reshape(shape)
            fDat.read(4)
            fDat.read(4)
            self.__harrisMO = np.fromfile(fDat, float, nbf**2).reshape(shape).T
            fDat.read(4)
        self.__toOr = self.__ToOrtho(self.__overlap)
        
        self.__maxSCFIter = 200
        self.__thresRmsDiffDens = 1.0e-8
        self.__thresMaxDiffDens = 1.0e-6
        self.__thresDiffEnergy = 1.0e-6
    
    def __del__(self):
        os.system('rm -r ' + self.__workPath)
    
    def __RunG09(self, keyword):
        info = self.__info
        numCPUCore = info['numcores'] if 'numcores' in info else 1
        memory = info['memory'] if 'memory' in info else '1gb'
        method = info['dft'] if 'dft' in info else 'hf'
        gjf = ['%nprocshared={:d}'.format(numCPUCore)]
        gjf += ['%mem={:s}'.format(memory)]
        gjf += ['#p ' + method + ' ' + info['basis'] + ' ' +
                'iop(5/13=1, 5/18=-2) scf(maxcycle=1, vshift=-1) ' +
                keyword]
        gjf += ['', 'dummy title', '']
        gjf += ['{:3d} {:3d}'.format(info['charge'], info['2s+1'])]
        gjf += [('{:3d}' + ' {:15.10f}' * 3)
                .format(int(line[0]), *tuple(line[1:])) for line in info['xyz']]
        gjf += ['', '', '']
        gjf = '\n'.join(gjf)
        with open(self.__workGjf, 'w') as fGjf:
            fGjf.write(gjf)
        os.system('g09binfile=' + self.__workDat + ' '
                  'g09 ' + self.__workGjf + ' ' + self.__workLog)
    
    # Perform SCF calculation
    #   guess: 'core', 'sad', or occOrbList
    def RunSCF(self, guess='harris'):
        if type(guess) is str:
            guessDict = {'core': self.__GuessCore, 'harris': self.__GuessHarris}
            occOrbList = guessDict[guess.lower()]()
        elif type(guess) is list:
            occOrbList = guess
        densList = self.__OccOrbToDens(occOrbList)
        lciis = LCIIS(self.__overlap, self.__toOr, self.__verbose)
        energy = 0.0
        for numIter in range(1, self.__maxSCFIter):
            if self.__verbose:
                print('scf iter {}; energy: {}'.format(numIter, energy))
            oldEnergy = energy
            (fockList, energy) = self._FockEnergy(occOrbList, densList)
            fockList = lciis.NewFock(fockList, densList)
            oldDensList = [dens.copy() for dens in densList]
            occOrbList = self.__FockToOccOrb(fockList)
            densList = self.__OccOrbToDens(occOrbList)
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
    
    # Construct a list of Fock matrix and calculate energy
    def _FockEnergy(self, occOrbList, densList):
        nbf = self.__overlap.shape[0]
        with open(self.__workDat, 'wb') as fDat:
            for dens in densList:
                fDat.write(struct.pack('i', 8 * nbf**2))
                dens.tofile(fDat)
                fDat.write(struct.pack('i', 8 * nbf**2))
        self.__RunG09('iop(5/199=1) guess=core')
        fockList = []
        with open(self.__workDat, 'rb') as fDat:
            fDat.read(4)
            energy = struct.unpack('d', fDat.read(8))[0]
            fDat.read(4)
            for _ in range(len(set(self.__numElecAB))):
                fDat.read(4)
                fock = np.fromfile(fDat, float, nbf**2).reshape(nbf, nbf).T
                fDat.read(4)
                fockList += [fock]
        return (fockList, energy)
    
    # Find a transform from atomic orbitals to orthogonal orbitals
    def __ToOrtho(self, overlap):
        (eigVal, eigVec) = np.linalg.eigh(overlap)
        keep = eigVal > 1.0e-6
        return eigVec[:, keep] / np.sqrt(eigVal[keep])[None, :]
    
    # These guess functions return occOrbList
    def __GuessCore(self):
        return self.__FockToOccOrb([self.__coreH] * len(set(self.__numElecAB)))
    
    def __GuessHarris(self):
        return [self.__harrisMO[:, :ne] for ne in self.__numElecAB]
    
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
    



