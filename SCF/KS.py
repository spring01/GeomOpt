import numpy as np
from .HF import HF

class KS(HF):
    
    def __init__(self, xyz, info, verbose=False):
        HF.__init__(self, xyz, info, verbose)
        self.pypsi.DFT_Initialize(info['dft'])
        self.__hfExcMix = info['hfExcMix']
    
    def _FockEnergy(self, occOrbList, densList):
        self.__dftVList = self.pypsi.DFT_OccOrbToV(occOrbList)
        if self.__hfExcMix == 0.0:
            couList = self.pypsi.JK_OccOrbToJ(occOrbList)
            fockList = [self._FockNoExc(couList) + dftV
                        for dftV in self.__dftVList]
        else:
            self.pypsi.JK_CalcAllFromOccOrb(occOrbList)
            couList = self.pypsi.JK_GetJ()
            excList = self.pypsi.JK_GetK()
            fockList = [self._FockNoExc(couList) + dftV - self.__hfExcMix * exc
                        for (dftV, exc) in zip(self.__dftVList, excList)]
        energy = self.pypsi.DFT_EnergyXC()
        energy -= np.mean([dftV.ravel().dot(dens.ravel())
                           for (dftV, dens) in zip(self.__dftVList, densList)])
        energy += HF._HFEnergy(self, fockList, densList)
        return (fockList, energy)

