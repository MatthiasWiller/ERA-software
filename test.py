import numpy as np
import scipy.stats

Ca = 140
d = 100
lam = 1
tmp = 1-scipy.stats.gamma.cdf(Ca,a=d, scale=lam)

print('done!')