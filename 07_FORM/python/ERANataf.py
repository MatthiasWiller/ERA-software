# import of modules
import numpy as np
from scipy import stats
from scipy import optimize
from scipy import linalg
'''
---------------------------------------------------------------------------
Nataf Transformation of random variables
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
Version 2017-10:
---------------------------------------------------------------------------
* This software performs the Nataf transformation of random variables.
* It is possible to generate random numbers according to their Nataf
joint pdf and to evaluate this pdf.
* The inverse Nataf transformation is also defined as a function.
* It requires the use of Objects of the class ERADist which is also
published on the homepage of the ERA Group of TUM.
---------------------------------------------------------------------------
References:
1. Liu, Pei-Ling and Armen Der Kiureghian (1986) - Multivariate distribution
models with prescribed marginals and covariances.
Probabilistic Engineering Mechanics 1(2), 105-112
---------------------------------------------------------------------------
'''

class ERANataf(object):
    def __init__(self, M, Correlation):
        self.Marginals = np.array(M, ndmin=1)
        self.Rho_X = Correlation
        n_dist = len(M)

        #  check if all distributions have finite moments
        for i in range(n_dist):
            if (not((np.isfinite(self.Marginals[i].mean()) and
                     np.isfinite(self.Marginals[i].std())))):
                raise RuntimeError("The marginal distributions need to have "
                                   "finite mean and variance")

        '''
        Calculation of the transformed correlation matrix. This is achieved
        by a quadratic two-dimensional Gauss-Legendre integration
        '''

        n = 1024
        zmax = 8
        zmin = -zmax
        points, weights = np.polynomial.legendre.leggauss(n)
        points = - (0.5 * (points + 1) * (zmax - zmin) + zmin)
        weights = weights * (0.5 * (zmax - zmin))

        xi = np.tile(points, [n, 1])
        xi = xi.flatten(order='F')
        eta = np.tile(points, n)

        first = np.tile(weights, n)
        first = np.reshape(first, [n, n])
        second = np.transpose(first)

        weights2d = first * second
        w2d = weights2d.flatten()

        #  check is X the identiy
        self.Rho_Z = np.identity(n=n_dist)
        if (np.linalg.norm(self.Rho_X - np.identity(n=n_dist)) > 10**(-5)):
            for i in range(n_dist):
                for j in range(i+1, n_dist):
                    if (self.Rho_X[i, j] == 0):
                        continue

                    elif ((self.Marginals[i].Name == 'standardnormal') and
                          (self.Marginals[j].Name == 'standardnormal')):
                        self.Rho_Z[i, j] = self.Rho_X[i, j]
                        self.Rho_Z[j, i] = self.Rho_X[j, i]
                        continue

                    elif ((self.Marginals[i].Name == 'normal') and
                          (self.Marginals[j].Name == 'normal')):
                        self.Rho_Z[i, j] = self.Rho_X[i, j]
                        self.Rho_Z[j, i] = self.Rho_X[j, i]
                        continue

                    elif ((self.Marginals[i].Name == 'normal') and
                          (self.Marginals[j].Name == 'lognormal')):
                        Vj = self.Marginals[j].std()/self.Marginals[j].mean()
                        self.Rho_Z[i, j] = (self.Rho_X[i, j] *
                                            Vj/np.sqrt(np.log(1 + Vj**2)))
                        self.Rho_Z[j, i] = self.Rho_Z[i, j]
                        continue

                    elif ((self.Marginals[i].Name == 'lognormal') and
                          (self.Marginals[j].Name == 'normal')):
                        Vi = self.Marginals[i].std()/self.Marginals[i].mean()
                        self.Rho_Z[i, j] = (self.Rho_X[i, j] *
                                            Vi/np.sqrt(np.log(1 + Vi**2)))
                        self.Rho_Z[j, i] = self.Rho_Z[i, j]
                        continue

                    elif ((self.Marginals[i].Name == 'lognormal') and
                          (self.Marginals[j].Name == 'lognormal')):
                        Vi = self.Marginals[i].std()/self.Marginals[i].mean()
                        Vj = self.Marginals[j].std()/self.Marginals[j].mean()
                        self.Rho_Z[i, j] = (np.log(1 + self.Rho_X[i, j]*Vi*Vj)
                                            / np.sqrt(np.log(1 + Vi**2) *
                                                      np.log(1+Vj**2)))
                        self.Rho_Z[j, i] = self.Rho_Z[i, j]
                        continue

                    #  solving Nataf
                    tmp_f_xi = ((self.Marginals[j].icdf(stats.norm.cdf(eta)) -
                                self.Marginals[j].mean()) /
                                self.Marginals[j].std())
                    tmp_f_eta = ((self.Marginals[i].icdf(stats.norm.cdf(xi)) -
                                  self.Marginals[i].mean()) /
                                 self.Marginals[i].std())
                    coef = tmp_f_xi * tmp_f_eta * w2d

                    def fun(rho0):
                        return ((coef *
                                 self.bivariateNormalPdf(xi, eta, rho0)).sum()
                                - self.Rho_X[i, j])

                    x0, r = optimize.brentq(f=fun,
                                            a=-1 + np.finfo(float).eps,
                                            b=1 - np.finfo(float).eps,
                                            full_output=True)
                    if (r.converged == 1):
                        self.Rho_Z[i, j] = x0
                        self.Rho_Z[j, i] = self.Rho_Z[i, j]
                    else:
                        sol = optimize.fsolve(func=fun,
                                              x0=self.Rho_X[i, j],
                                              full_output=True)
                        if (sol[2] == 1):
                            self.Rho_Z[i, j] = sol[0]
                            self.Rho_Z[j, i] = self.Rho_Z[i, j]
                        else:
                            sol = optimize.fsolve(func=fun,
                                                  x0=-self.Rho_X[i, j],
                                                  full_output=True)
                            if (sol[2] == 1):
                                self.Rho_Z[i, j] = sol[0]
                                self.Rho_Z[j, i] = self.Rho_Z[i, j]
                            else:
                                for i in range(10):
                                    init = 2 * np.random.rand() - 1
                                    sol = optimize.fsolve(func=fun,
                                                          x0=init,
                                                          full_output=True)
                                    if (sol[2] == 1):
                                        break
                                if (sol[2] == 1):
                                    self.Rho_Z[i, j] = sol[0]
                                    self.Rho_Z[j, i] = self.Rho_Z[i, j]
                                else:
                                    raise RuntimeError("brentq and fsolve coul"
                                                       "d not converge to a "
                                                       "solution of the Nataf "
                                                       "integral equation")
        try:
            self.A = linalg.cholesky(self.Rho_Z, lower=True)
        except linalg.LinAlgError:
            raise RuntimeError("Transformed correlation matrix is not positive"
                               " definite --> Nataf transformation is not "
                               "applicable")

    # %% ----------------------------------------------------------------------------
    '''
    This function performs the transformation from X to U by taking
    the inverse standard normal cdf of the cdf of every value. Then it
    performs the transformation from Z to U. A is the lower triangular
    matrix of the cholesky decomposition of Rho_Z and U is the resulting
    independent standard normal vector Afterwards it calculates the
    Jacobian of this Transformation if it is needed
    '''
    def X2U(self, X, Jacobian=False):
        if (not(np.shape(self.Marginals)[0] == np.shape(X)[0])):
            X = X.T
        m, n = np.shape(X)
        if (np.size(self.Marginals) == 1):
            Z = np.zeros([n, m])
            for i in range(m):
                Z[i, :] = stats.norm.ppf(self.Marginals[i].cdf(X[i, :]))
            diag = np.zeros([m, m])
        else:
            Z = np.zeros([m, n])
            for i in range(m):
                Z[i, :] = stats.norm.ppf(self.Marginals[i].cdf(X[i, :]))
            diag = np.zeros([m, m])
        U = np.linalg.solve(self.A, Z)
        if (not Jacobian):
            return U
        else:
            for i in range(m):
                diag[i, i] = stats.norm.pdf(Z[i])/self.Marginals[i].pdf(X[i])
            Jac = np.dot(diag, self.A)
            return U, Jac
    
    # %% ----------------------------------------------------------------------------
    def U2X(self, U, Jacobian=False):
        if (not(np.shape(self.Marginals)[0] == np.shape(U)[0])):
            U = U.T
        Z = np.dot(self.A, U)
        m, n = np.shape(U)
        X = np.zeros([m, n])
        for i in range(m):
            X[i, :] = self.Marginals[i].icdf(stats.norm.cdf(Z[i, :]))
        diag = np.zeros([m, m])
#        U = np.linalg.solve(self.A, Z)
        if (not Jacobian):
            return X
        else:
            for i in range(m):
                diag[i, i] = self.Marginals[i].pdf(X[i])/stats.norm.pdf(Z[i])
            Jac = np.linalg.solve(self.A, diag)
            return X, Jac

    # %% ----------------------------------------------------------------------------
    def random(self, N):
        N = int(N)
        si = np.size(self.Marginals)
        U = np.random.randn(si, N)
        Z = np.dot(self.A, U)
        jr = np.zeros([si, N])
        for i in range(si):
            jr[i, :] = self.Marginals[i].icdf(stats.norm.cdf(Z[i, :]))
        return jr

    # %% ----------------------------------------------------------------------------
    def jointpdf(self, x):
        x = np.array(x, ndmin=2)
        if(np.shape(x)[1] == 1):
            x = x.T
        try:
            m, n = np.shape(self.Marginals)
        except ValueError:
            m = np.shape(self.Marginals)[0]
            n = 1
        if(m > n):
            s = m
        else:
            s = n
        n = np.shape(x)[0]
        U = np.zeros([s, n])
        mu = np.zeros(s)
        f = np.zeros([s, n])
        phi = np.zeros([s, n])
        for i in range(s):
            U[i, :] = stats.norm.ppf(self.Marginals[i].cdf(x[:, i]))
            phi[i, :] = stats.norm.pdf(U[i, :])
            f[i, :] = self.Marginals[i].pdf(x[:, i])
        phi_n = stats.multivariate_normal.pdf(U.T,
                                              mu,
                                              self.Rho_Z)
        print(phi_n)
        jointpdf = np.zeros(n)
        for i in range(n):
            try:
                jointpdf[i] = ((np.prod(f[:, i])/np.prod(phi[:, i])) * phi_n)
            except ZeroDivisionError:
                jointpdf[i] = 0
        return jointpdf

    # %% ----------------------------------------------------------------------------
    def jointcdf(self, x):
        x = np.array(x, ndmin=2)
        if(np.shape(x)[1] == 1):
            x = x.T
        try:
            m, n = np.shape(self.Marginals)
        except ValueError:
            m = np.shape(self.Marginals)[0]
            n = 1
        if(m > n):
            s = m
        else:
            s = n
        n = np.shape(x)[0]
        U = np.zeros([n, s])
        mu = np.zeros([1, s])
        for i in range(s):
            U[:, i] = stats.norm.ppf(self.Marginals[i].cdf(x[:, i]),
                                     loc=0,
                                     scale=1)
        low = np.array(-np.inf * np.ones(s), ndmin=2)
        return stats.mvn.mvnun(low, U, mu.T, np.matrix(self.Rho_Z))

    # %% ----------------------------------------------------------------------------
    @staticmethod
    def bivariateNormalPdf(x1, x2, rho):
        return (1 / (2 * np.pi * np.sqrt(1-rho**2)) *
                np.exp(-1/(2*(1-rho**2)) *
                       (x1**2 - 2 * rho * x1 * x2 + x2**2)))
