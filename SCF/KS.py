import numpy as np
from HF import HF

class KS(HF):
    
    def __init__(self, xyz, info):
        HF.__init__(self, xyz, info)
        self.pypsi.DFT_Initialize(info['dft'])
        self.__hfExcMix = info['hfExcMix']
    
    def _OccOrbToFock(self, occOrbList):
        self.__dftVList = self.pypsi.DFT_OccOrbToV(occOrbList)
        if self.__hfExcMix == 0.0:
            couList = self.pypsi.JK_OccOrbToJ(occOrbList)
            return [self._FockNoExc(couList) + dftV for dftV in self.__dftVList]
        else:
            self.pypsi.JK_CalcAllFromOccOrb(occOrbList)
            couList = self.pypsi.JK_GetJ()
            excList = self.pypsi.JK_GetK()
            return [self._FockNoExc(couList) + dftV - self.__hfExcMix * exc
                    for (dftV, exc) in zip(self.__dftVList, excList)]
    
    def _SCFEnergy(self, fockList, densList):
        zipList = zip(self.__dftVList, densList)
        ksEDiff = self.pypsi.DFT_EnergyXC() - np.mean(
            [dftV.ravel().dot(dens.ravel()) for (dftV, dens) in zipList])
        return HF._SCFEnergy(self, fockList, densList) + ksEDiff
