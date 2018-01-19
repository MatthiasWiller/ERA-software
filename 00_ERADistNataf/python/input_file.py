import numpy as np
import matplotlib.pylab as plt
from ERANataf import ERANataf
from ERADist import ERADist
'''
---------------------------------------------------------------------------
Example file: Use of ERADist and ERANataf
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer (s.geyer@tum.de), 
Iason Papaioannou and Felipe Uribe
implemented in Python by:
Alexander von Ramm (alexander.ramm@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2017-10
---------------------------------------------------------------------------
References:
1. ERADist-ERANataf documentation
---------------------------------------------------------------------------
'''
plt.close('all')
np.random.seed(2017)
example = 2

M = list()
if(example == 1):
    M.append(ERADist('normal', 'PAR', [4, 2]))
    M.append(ERADist('gumbel', 'MOM', [1, 2]))
    M.append(ERADist('exponential', 'PAR', [4]))
    Rho = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]])
elif(example == 2):
    M.append(ERADist('rayleigh', 'PAR', 1))
    M.append(ERADist('gumbel', 'MOM', [0, 1]))
    Rho = np.array([[1.0, 0.6], [0.6, 1.0]])
elif(example == 3):
    M.append(ERADist('gamma', 'PAR', [2, 1]))
    M.append(ERADist('chisquare', 'PAR', 5))
    Rho = np.array([[1.0, 0.5], [0.5, 1.0]])
elif(example == 4):
    M.append(ERADist('gamma', 'PAR', [1, 2]))
    M.append(ERADist('weibull', 'PAR', [4, 5]))
    Rho = np.array([[1.0, 0.1], [0.1, 1.0]])

# applying Nataf transformation
T_Nataf = ERANataf(M, Rho)

# generation of of random samples from the joint distribution
N = 1000
X = T_Nataf.random(N)

# samples in standard space
U = T_Nataf.X2U(X)

# samples in physical space
X = T_Nataf.U2X(U)

n = len(M)
xx = np.zeros(shape=[n, N])
f_X = np.zeros(shape=[n, N])
F_X = np.zeros(shape=[n, N])

for i in range(n):
    xx[i, :] = X[i, :]
    # marginal pdfs in pyhsical space
    f_X[i, :] = T_Nataf.Marginals[i].pdf(xx[i, :])
    # marginal cdfs in pyhsical space
    F_X[i, :] = T_Nataf.Marginals[i].cdf(xx[i, :])

# plot the marginal pdfs
fig_marginalPDFs    = plt.figure(figsize=[16, 9])
fig_marginalPDFs_ax = fig_marginalPDFs.add_subplot(111)
for i in range(n):
    fig_marginalPDFs_ax.scatter(xx[i, :], f_X[i, :],
                                s=3,
                                marker='.',
                                label=r'$f_{{X{i}}}$'.format(i=i+1))
plt.legend()

# plot the marginal cdfs
fig_marginalCDFs = plt.figure(figsize=[16, 9])
fig_marginalCDFs_ax = fig_marginalCDFs.add_subplot(111)
for i in range(n):
    fig_marginalCDFs_ax.scatter(xx[i, :], F_X[i, :],
                                s=3,
                                marker='.',
                                label=r'$f_{{X{i}}}$'.format(i=i+1))
plt.legend()

# plot samples in physical and standard space
fig_Samples = plt.figure(figsize=[16, 9])
if(n == 2):
    fig_SamplesAx1 = fig_Samples.add_subplot(121)
    fig_SamplesAx1.scatter(X[0, :], X[1, :])
    fig_SamplesAx1.set_title('Physical Space')
    fig_SamplesAx1.set_xlabel(r'$X_{1}$')
    fig_SamplesAx1.set_ylabel(r'$X_{2}$')
    fig_SamplesAx2 = fig_Samples.add_subplot(122)
    fig_SamplesAx2.scatter(U[0, :], U[1, :])
    fig_SamplesAx2.set_title('Standard space')
    fig_SamplesAx2.set_xlabel(r'$U_{1}$')
    fig_SamplesAx2.set_ylabel(r'$U_{2}$')

if(n == 3):
    fig_SamplesAx1 = fig_Samples.add_subplot(121, projection='3d')
    fig_SamplesAx1.scatter(X[0, :], X[1, :], X[2, :])
    fig_SamplesAx1.set_title('Physical Space')
    fig_SamplesAx1.set_title('Physical Space')
    fig_SamplesAx1.set_xlabel(r'$X_{1}$')
    fig_SamplesAx1.set_ylabel(r'$X_{2}$')
    fig_SamplesAx1.set_zlabel(r'$X_{3}$')

    fig_SamplesAx2 = fig_Samples.add_subplot(122, projection='3d')
    fig_SamplesAx2.scatter(U[0, :], U[1, :], U[2, :])
    fig_SamplesAx2.set_title('Standard space')
    fig_SamplesAx2.set_xlabel(r'$U_{1}$')
    fig_SamplesAx2.set_ylabel(r'$U_{2}$')
    fig_SamplesAx2.set_zlabel(r'$U_{3}$')