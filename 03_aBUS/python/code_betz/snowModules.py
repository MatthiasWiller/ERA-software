#!/usr/bin/env python
#module containing necessary fuctionality to simulate snow accumulation and
#processes

import pandas as pd
import numpy as np

#Snow accumulation according to section 2.9.1 of WaSiM Documentation
def snowAccumulation(temp, prec, tempTrans, tempThreshA):
    if (transTemp(temp, tempTrans, tempThreshA)):
       pSnow=snowFraction(temp, tempTrans, tempThreshA)
       swe = prec * pSnow
    else:
        swe = prec
    return swe

def snowFraction(temp, tempTrans, tempThreshA):
    pSnow=(tempThreshA + tempTrans - temp)/(2 * tempTrans)
    return pSnow

def transTemp(temp, tempTrans, tempThreshA):
    bol =((temp>(tempThreshA-tempTrans)) and (temp<tempThreshA+tempTrans))
    return bol

#snow ablation according to section 2.9.2 of WaSiM Documentation
#temperature-index-approach
def snowMeltTIndex(c_0, temp, tempThreshM):
       if(temp>tempThreshM):
           M=c_0*(temp-tempThreshM)*(-1/24)
       else:
           M=0
       return M

def TIndex(temp, prec, tempThreshM, tempThreshA, tempTrans, c_0):
    deltaSWE = 0
    if (prec>0):
       deltaSWE += snowAccumulation(temp, prec, tempTrans, tempThreshA)
       deltaSWE += snowMeltTIndex(c_0, temp, tempThreshM)
    return deltaSWE


#snowModule
def snowModule(data, par, module):
    data['swe_sim'] = np.nan
    #initial condition:there connot be snow before the module is started!
    fallback=0
    #if TIndex method is specified
    if(module=='TIndex'):
        #first time step
       index = data.index.values 
       data.set_value(index[0], 'swe_sim', fallback + TIndex(data['temperature'].iloc[0],
                                               data['precipitation'].iloc[0],
                                               par[0], par[1], par[2], par[3]))
       if(data['swe_sim'].iloc[0]<0):
           data.set_value(index[0],'swe_sim',0)
       for timestep in range(1,data.shape[0]):
           deltaSWE=TIndex(data['temperature'].iloc[timestep],
                       data['precipitation'].iloc[timestep], par[0], par[1],
                       par[2], par[3])

           if (np.isnan(data['swe_sim'].iloc[timestep-1])):
                newSWE = fallback + deltaSWE
           else:
                newSWE = data['swe_sim'].iloc[timestep-1] + deltaSWE
           if (newSWE < 0):
                data.set_value(index[timestep], 'swe_sim', 0)
           else:
               data.set_value(index[timestep], 'swe_sim', newSWE)
               fallback = data['swe_sim'].iloc[timestep]
    return data
