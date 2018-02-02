import numpy as np
"""
---------------------------------------------------------------------------
Basic resampling algorithm
---------------------------------------------------------------------------
Created by:
Felipe Uribe (felipe.uribe@tum.de)
Iason Papaioannou (iason.papaioannou@tum.de)
implemented in Python by:
Matthias Willer (matthias.willer@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-02
---------------------------------------------------------------------------
Input:
* w : unnormalized plausability weights
---------------------------------------------------------------------------
Output:
* idx : index with the highest probability
---------------------------------------------------------------------------
"""
def resampling_index(w):
    N    = len(w)
    csum = np.cumsum(w)
    ssum = sum(w)
    lim  = ssum*np.rand.rand()
    #
    i = 1
    while csum[i] <= lim:
        i = i+1
        if i > N:
            i   = 1
            lim = ssum*np.rand.rand()
    idx = i

    return idx
## END