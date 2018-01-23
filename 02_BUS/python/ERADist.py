# import of modules
import numpy as np
import scipy.stats
'''
---------------------------------------------------------------------------
Generation of distribution objects
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
Version 2018-01:
* Fixing of bugs in the gumbel,gumbelmin and gamma distribution
---------------------------------------------------------------------------
This software generates distribution objects according to the parameters
and definitions used in the distribution table of the ERA Group of TUM.
They can be defined either by their parameters, the first and second
moment or by data, given as a vector.
---------------------------------------------------------------------------
'''

class ERADist(object):
    def __init__(self, name, opt, val):
        # constructor
        self.Name = name.lower()
        self.Par = [np.nan, np.nan, np.nan, np.nan]
        val = np.asarray(val, dtype=float)
        # definition of the distribution by its parameters
        if (opt.upper() == 'PAR'):
            if (name.lower() == 'binomial'):
                if ((val[1] >= 0) and (val[1] <= 1) and (val[0] % 1 == 0)):
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = scipy.stats.binom(n=self.Par[0], p=self.Par[1])
                else:
                    raise RuntimeError("The Binomial distribution is not "
                                       "defined for your parameters")

            elif (name.lower() == 'geometric'):
                if (val > 0 and val <= 1):
                    self.Par = val
                    self.Dist = scipy.stats.geom(p=val)
                else:
                    raise RuntimeError("The Geometric distribution is not "
                                       "defined for your parameters")

            elif (name.lower() == 'negativebinomial'):
                if (
                        (val[1] > 0) and
                        (val[1] <= 1) and
                        (val[0] > 0) and
                        (val[0] % 1 == 0)):
                    self.Par = val
                    self.Dist = scipy.stats.nbinom(n=val[0], p=val[1])
                else:
                    raise RuntimeError("The Negative Binomial distribution "
                                       "is not defined for your parameters")

            elif (name.lower() == 'poisson'):
                n = len(val)
                if (n == 1):
                    if (val > 0):
                        self.Par = val
                        self.Dist = scipy.stats.poisson(mu=val)
                    else:
                        raise RuntimeError("The Poisson distribution is not "
                                           "defined for your parameters")
                if (n == 2):
                    if (val[0] > 0 and val[1] > 0):
                        self.Par = val[0] * val[1]
                        self.Dist = scipy.stats.poisson(mu=val[0] * val[1])
                    else:
                        raise RuntimeError("The Poisson distribution is not "
                                           "defined for your parameters")

            elif (name.lower() == 'exponential'):
                if (val > 0):
                    self.Dist = scipy.stats.expon(scale=1/val)
                else:
                    raise RuntimeError("The Exponential distribution is not "
                                       "defined for your parameters")

            # parameters of the gamma distribution: a = k, scale = 1/lambda
            elif (name.lower() == 'gamma'):
                if (val[0] > 0 and val[1] > 0):
                    self.Par[0] = val[0]
                    self.Par[1] = 1/val[1]
                    self.Dist = scipy.stats.gamma(a=val[0], scale=1 / val[1])
                else:
                    raise RuntimeError("The Gamma distribution is not defined "
                                       "for your parameters")

            elif (name.lower() == 'beta'):
                '''
                beta distribution in lecture notes can be shifted in order to
                account for ranges [a,b] -> this is not implemented yet
                '''
                if ((val[0] > 0) and (val[1] > 0) and (val[2] < val[3])):
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Par[2] = val[2]
                    self.Par[3] = val[3]
                    self.Dist = scipy.stats.beta(a=val[0], b=val[1])
                else:
                    raise RuntimeError("The Beta distribution is not defined "
                                       "for your parameters")

            elif (name.lower() == 'gumbelmin'):
                '''
                this distribution can be used to model minima
                '''
                if (val[1] > 0):
                    '''
                    sigma is the scale parameter
                    mu is the location parameter
                    '''
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = scipy.stats.genextreme(c=0,
                                                       scale=val[1],
                                                       loc=-val[0])
                else:
                    raise RuntimeError("The Gumbel distribution is not defined"
                                       " for your parameters")

            elif (name.lower() == 'gumbel'):
                '''
                mirror image of this distribution can be used to model maxima
                '''
                if (val[1] > 0):
                    '''
                    sigma is the scale parameter
                    mu is the location parameter
                    '''
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = scipy.stats.genextreme(c=0,
                                                       scale=val[1],
                                                       loc=val[0])
                else:
                    raise RuntimeError("The Gumbel distribution is not defined"
                                       " for your parameters")

            elif (name.lower() == 'frechet'):
                if ((val[0] > 0) and (val[1] > 0)):
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = scipy.stats.genextreme(c=-1/val[1],
                                                       scale=val[0]/val[1],
                                                       loc=val[0])
                else:
                    raise RuntimeError("The Frechet distribution is not define"
                                       "d for your parameters")

            elif (name.lower() == 'weibull'):
                if ((val[0] > 0) and (val[1] > 0)):
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Dist = scipy.stats.weibull_min(c=val[0], scale=val[1])
                else:
                    raise RuntimeError("The Weibull distribution is not "
                                       "definied for your parameters")

            elif (name.lower() == 'gev'):
                if (val[1] > 0):
                    self.Par[0] = val[0]
                    self.Par[1] = val[1]
                    self.Par[2] = val[2]
                    self.Dist = scipy.stats.genextreme(c=-val[0],
                                                       scale=val[1],
                                                       loc=val[2])
                else:
                    raise RuntimeError("The Generalized Extreme Value Distribu"
                                       "tion is not defined for your parameter"
                                       "s")

            elif (name.lower() == 'gevmin'):  # double check definition
                if (val[1] > 0):
                    self.Dist = scipy.stats.genextreme(c=-val[0],
                                                       scale=val[1],
                                                       loc=-val[2])
                else:
                    raise RuntimeError("The Generalized Extreme Value Distribu"
                                       "tion is not defined for your parameter"
                                       "s")

            elif (name.lower() == 'pareto'):
                if (val[0] > 0 and val[1] > 0):
                    self.Dist = scipy.stats.genpareto(c=1/val[1],
                                                      scale=val[0]/val[1],
                                                      loc=val[0])
                else:
                        raise RuntimeError("The Pareto distribution is not def"
                                           "ined for your parameters")

            elif (name.lower() == 'rayleigh'):
                if (val > 0):
                    self.Dist = scipy.stats.rayleigh(scale=val)
                else:
                    raise RuntimeError("The Rayleigh distribution is not "
                                       "defined for your parameters")

            elif (name.lower() == 'chisquare'):
                if (val > 0 and val % 1 == 0):
                    self.Dist = scipy.stats.gamma(a=val / 2.0, scale=2)
                else:
                    raise RuntimeError("The Chisquared distribution is not "
                                       "defined for your parameters")

            elif (name.lower() == 'uniform'):
                '''
                the distribution defined in scipy is uniform between loc and
                loc + scale
                '''
                self.Dist = scipy.stats.uniform(loc=val[0],
                                                scale=val[1]-val[0])

            elif ((name.lower() == 'standardnormal') or
                  (name.lower() == 'standardgaussian')):
                self.Dist = scipy.stats.norm(loc=0, scale=1)

            elif (name.lower() == 'normal' or name.lower() == 'gaussian'):
                if (val[1] > 0):
                    self.Dist = scipy.stats.norm(loc=val[0], scale=val[1])
                else:
                    raise RuntimeError("The Normal distribution is not defined"
                                       " for your parameters")

            elif (name.lower() == 'lognormal'):
                if (val[0] > 0 and val[1] > 0):
                    '''
                    a parametrization in terms of the underlying normally
                    distributed variable corresponds to s = sigma,
                    scale = exp(mu)
                    '''
                    self.Dist = scipy.stats.lognorm(s=val[1],
                                                    scale=np.exp(val[0]))
                else:
                    raise RuntimeError("The Lognormal distribution is not "
                                       "defined for your parameters")

            else:
                raise RuntimeError('Distribution type not available')


#       if the distribution is to be defined using the moments
        elif (opt.upper() == 'MOM'):
            if (val[1] < 0):
                raise RuntimeError("The standard deviation must not be "
                                   "negative")

            elif (name.lower() == 'binomial'):
                # Solve system of two equations for the parameters
                self.Par[1] = 1 - (val[1])**2 / val[0]
                self.Par[0] = val[0]/self.Par[1]
                # Evaluate if distribution can be defined on the parameters
                if (self.Par[1] % 1 <= 10**(-4)):
                    self.Par[1] = round(self.Par[1], 0)
                    if (0 <= self.Par[0] and self.Par[1] <= 1):  # OK
                        self.Dist = scipy.stats.binom(n=self.Par[0],
                                                      p=self.par[1])
                    else:
                        raise RuntimeError('Please select other moments')
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'geometric'):
                # Solve Equation for the parameter based on the first moment
                self.Par = 1/val[0]
                '''
                Evaluate if distribution can be defined on the parameter and if
                the moments are well defined
                '''
                if (0 <= self.Par and self.Par <= 1):
                    self.Dist = scipy.stats.geom(p=self.Par)
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'negativebinomial'):
                # Solve System of two equations for the parameters
                self.Par[1] = val[0]/((val[0]+val[1])**2)
                self.Par[0] = val[1] * self.Par[1]
                # Evaluate if distribution can be defined on the parameters
                if (self.Par[0] % 1 <= 10**(-4)):
                    self.Par[0] = round(self.Par[0], 0)
                    if (0 <= self.Par[1] and self.Par[1] <= 1):
                        self.Dist = scipy.stats.nbinom(n=self.Par[0],
                                                       p=self.Par[1])
                    else:
                        raise RuntimeError('Please select other moments')
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'poisson'):
                self.Par = val[0]
                # Evaluluate if moments match
                if (0 <= self.Par):
                    self.Dist = scipy.stats.poisson(mu=self.Par)
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'exponential'):
                '''
                Solve Equation for the parameter of the distribution based on
                the first moment
                '''
                try:
                    self.Par = 1/val[0]
                except ZeroDivisionError:
                    raise RuntimeError('The first moment cannot be zero!')
                if (0 <= self.Par):
                    self.Dist = scipy.stats.expon(scale=1/self.Par)
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'gamma'):
                # Solve system of equations for the parameters
                self.Par[0] = val[0]/(val[1]**2)  # parameter lambda
                self.Par[1] = self.Par[0] * val[0]  # parameter k
                # Evaluate if distribution can be defined on the parameters
                if (self.Par[0] > 0 and self.Par[1] > 0):
                    self.Dist = scipy.stats.gamma(a=self.Par[1],
                                                  scale=1/self.Par[0])
                else:
                    raise RuntimeError('Please select other moments')

#            if (name.lower() == 'beta')

            elif (name.lower() == 'gumbelmin'):
                ne = 0.57721566490153  # euler constant
                # solve two equations for the parameters of the distribution
                self.Par[1] = val[1] * np.sqrt(6)/np.pi  # scale parameter
                self.Par[0] = val[0] - ne*self.Par[1]  # location parameter
                if (self.Par[1] > 0):
                    self.Dist = scipy.stats.gumbel_l(loc=self.Par[0],
                                                     scale=self.Par[1])
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'gumbel'):
                ne = 0.57721566490153  # euler constant
                # solve two equations for the parameters of the distribution
                self.Par[1] = val[1] * np.sqrt(6)/np.pi  # scale parameter
                self.Par[0] = val[0] - ne * self.Par[1]  # location parameter
                if (self.Par[1] > 0):
                    self.Dist = scipy.stats.gumbel_r(loc=self.Par[0],
                                                     scale=self.Par[1])
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'frechet'):
                par0 = 2.0001

                def equation(par):
                    return (np.sqrt(scipy.special.gamma(1 - 2/par) -
                                    scipy.special.gamma(1 - 1/par)**2) /
                            scipy.special.gamma(1 - 1/par) - val[1]/val[0])

                sol = scipy.optimize.fsolve(equation,
                                            x0=par0,
                                            full_output=True)
                if (sol[2] == 1):
                    self.Par[1] = sol[0][0]
                    self.Par[0] = val[0]/scipy.special.gamma(1 - 1/self.Par[1])
                else:
                    raise RuntimeError("fsolve could not converge to a solutio"
                                       "n for determining the parameters of th"
                                       "e Frechet distribution")
                if (self.Par[0] > 0 and self.Par[1] > 0):
                    c = 1/self.Par[1]
                    scale = self.Par[0]/self.Par[1]
                    loc = self.Par[0]
                    self.Dist = scipy.stats.genextreme(c=c,
                                                       scale=scale,
                                                       loc=loc)
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'weibull'):

                def equation(par):
                    return(np.sqrt(scipy.special.gamma(1 + 2 / par) -
                                   (scipy.special.gamma(1 + 1 / par))**2) /
                           scipy.special.gamma(1 + 1/par) - val[1]/val[0])

                sol = scipy.optimize.fsolve(equation,
                                            x0=0.02,
                                            full_output=True)
                if (sol[2] == 1):
                    self.Par[1] = sol[0][0]
                    self.Par[0] = val[0]/scipy.special.gamma(1 + 1/self.Par[1])
                else:
                    raise RuntimeError("fsolve could not converge to a solutio"
                                       "n for determining the parameters of th"
                                       "e Weibull distribution")
                if (self.Par[0] > 0 and self.Par[1] > 0):
                    self.Dist = scipy.stats.weibull_min(c=self.Par[1],
                                                        scale=self.Par[0])
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'gev'):  # doublecheck fsolve convergence
                if (val[0] == val[2]):
                    self.Par[0] = -1
                    self.Par[1] = val[1]
                    self.Par[2] = val[2]
                else:
                    if (val[0] > val[2]):
                        par0 = 0.3
                    else:
                        par0 = -1.5

                    def equation(par):
                        return ((scipy.special.gamma(1 - 2*par) -
                                 scipy.special.gamma(1 - par)**2) /
                                (scipy.special.gamma(1 - par) - 1)**2 -
                                (val[2]/(val[1] - val[2]))**2)

                    sol = scipy.optimize.fsolve(equation,
                                                x0=par0,
                                                full_output=True)
                    if (sol[2] == 1):
                        self.Par[0] = sol[0][0]
                        self.Par[1] = ((val[0]-val[3])*self.Par[0] /
                                       (scipy.special.gamma(1-self.Par[0])-1))
                    else:
                        raise RuntimeError("fsolve could not converge to a sol"
                                           "ution for determining the paramete"
                                           "rs of the GEV distribution")
                if (self.Par[1] > 0):
                    self.Dist = scipy.stats.genextreme(c=-self.Par[0],
                                                       scale=self.Par[1],
                                                       loc=self.Par[2])
                else:
                    raise RuntimeError("Please select other moments")

            elif (name.lower() == 'pareto'):
                self.Par[1] = 1 + np.sqrt(1-(val[0]/val[1])**2)
                self.Par[0] = val[0] * (self.Par[1]-1)/self.Par[1]
                if (self.Par[0] > 0 and self.Par[1] > 0):
                    c = 1/self.Par[1]
                    scale = self.Par[0]/self.Par[1]
                    loc = self.Par[0]
                    self.Dist = scipy.stats.genpareto(c=c,
                                                      scale=scale,
                                                      loc=loc)
                else:
                        raise RuntimeError('Please select other moments')

            elif (name.lower() == 'rayleigh'):
                self.Par = val[0]/np.sqrt(np.pi/2)
                if (self.Par > 0):
                    self.Dist = scipy.stats.rayleigh(scale=self.Par)
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'chisquare'):
                self.Par = val[0]
                if (self.Par % 1 <= 10**(-4)):
                    self.Par = round(self.Par, 0)
                else:
                    raise RuntimeError('Please select other moments')
                if (self.Par > 0):
                    self.Dist = scipy.stats.gamma(a=self.Par/2.0, b=2.0)
                else:
                    raise RuntimeError('Please select other moments')

            elif (name.lower() == 'uniform'):

                def equations(p):
                    a, b = p
                    return ((a+b)/2.0 - val[0], ((b-a)**2)/12.0 - val[1]**2)

                a, b = scipy.optimize.fsolve(equations, (1, 1))
                self.Par[0] = a
                self.Par[1] = b
                self.Dist = scipy.stats.uniform(loc=a, scale=b - a)

            elif ((name.lower() == 'standardnormal') or
                  (name.lower() == 'standardgaussian')):
                self.Par = [0, 1]
                self.Dist = scipy.stats.norm()

            elif ((name.lower() == 'normal') or
                  (name.lower() == 'gaussian')):
                self.Par = val
                self.Dist = scipy.stats.norm(loc=self.Par[0],
                                             scale=self.Par[1])

            elif (name.lower() == 'lognormal'):
                # solve two equations for the parameters of the distribution
                self.Par[0] = (np.log(val[0]) - np.log(np.sqrt(1 +
                               (val[1]/val[0])**2)))
                self.Par[1] = np.sqrt(np.log(1 + (val[1]/val[0])**2))
                self.Dist = scipy.stats.lognorm(s=self.Par[0],
                                                scale=np.exp(self.Par[1]))

        # if the distribution is to be fitted to a data vector
        elif (opt.upper() == 'DATA'):
            if (name.lower() == 'binomial'):
                raise RuntimeError("The binomial distribution is not supported"
                                   " in DATA")

            elif (name.lower() == 'negativebinomial'):
                raise RuntimeError("The negative binomial distribution is not "
                                   "supported in DATA")

            elif (name.lower() == 'geometric'):
                raise RuntimeError("The geometric distribution is not "
                                   "supported in DATA")

            elif (name.lower() == 'poisson'):
                raise RuntimeError("The poisson distribution is not supported"
                                   " in DATA")

            elif (name.lower() == 'exponential'):
                pars = scipy.stats.expon.fit(val, floc=0)
                self.Par = 1/pars[1]
                self.Dist = scipy.stats.expon(scale=1/self.Par)

            elif (name.lower() == 'gamma'):
                pars = scipy.stats.gamma.fit(val, floc=0)
                self.Par[0] = pars[0]
                self.Par[1] = 1/pars[2]
                self.Dist = scipy.stats.gamma(a=self.Par[0],
                                              scale=1/self.Par[1])

            elif (name.lower() == 'beta'):
                raise RuntimeError("The beta distribution is not supported "
                                   "in DATA")

            elif (name.lower() == 'gumbel'):
                pars = scipy.stats.gumbel_r.fit(val)
                self.Par[0] = pars[0]
                self.Par[1] = pars[1]
                self.Dist = scipy.stats.gumbel_r(loc=self.Par[0],
                                                 scale=self.Par[1])

            elif (name.lower() == 'gumbelmin'):
                pars = scipy.stats.gumbel_l.fit(val)
                self.Par[0] = pars[0]
                self.Par[1] = pars[1]
                self.Dist = scipy.stats.gumbel_l(loc=self.Par[0],
                                                 scale=self.Par[1])

            elif (name.lower() == 'frechet'):
                raise RuntimeError("The frechet distribution is not supported "
                                   "in DATA")

            elif (name.lower() == 'weibull'):
                pars = scipy.stats.weibull_min.fit(val, floc=0)
                self.Par[0] = pars[0]
                self.Par[1] = pars[2]
                self.Dist = scipy.stats.weibull_min(c=self.Par[0],
                                                    scale=self.Par[1])

            elif (name.lower() == 'normal' or name.lower() == 'gaussian'):
                pars = scipy.stats.norm.fit(val)
                self.Par[0] = pars[0]
                self.Par[1] = pars[1]
                self.Dist = scipy.stats.norm(loc=self.Par[0],
                                             scale=self.Par[1])

            elif (name.lower() == 'lognormal'):
                pars = scipy.stats.lognorm.fit(val, floc=0)
                self.Par[0] = pars[0]
                self.Par[1] = np.log(pars[2])
                self.Dist = scipy.stats.lognorm(s=self.Par[0],
                                                scale=np.exp(self.Par[1]))

            elif (name.lower() == 'gev'):
                pars = scipy.stats.genextreme.fit(val)
                self.Par[0] = -pars[0]
                self.Par[1] = pars[2]
                self.Par[2] = pars[1]
                self.Dist = scipy.stats.genextreme(c=-self.Par[0],
                                                   scale=self.Par[1],
                                                   loc=self.Par[2])

            elif (name.lower() == 'gevmin'):
                pars = scipy.stats.genextreme.fit(-val)
                self.Par[0] = -pars[0]
                self.Par[1] = pars[2]
                self.Par[2] = pars[1]
                self.Dist = scipy.stats.genextreme(c=-self.Par[0],
                                                   scale=self.Par[1],
                                                   loc=self.Par[2])

            elif (name.lower() == 'pareto'):
                raise RuntimeError("The pareto distribution is not supported "
                                   "in DATA")

            elif (name.lower() == 'rayleigh'):
                pars = scipy.stats.rayleigh.fit(val, floc=0)
                self.Par = pars[1]
                self.Dist = scipy.stats.rayleigh(scale=self.Par)

            elif (name.lower() == 'chisquare'):
                raise RuntimeError("The Chisquare distribution is not "
                                   " supported in DATA")
            else:
                raise RuntimeError('Distribution type not available')

        else:
            raise RuntimeError('Unknown option :' + opt)

    def mean(self):
        if (self.Name == 'negativebinomial'):
            return self.Dist.mean() + self.Par[0]

        if (self.Name == 'gumbel'):
            ne = 0.57721566490153
            return self.Par[0] + self.Par[1]*ne

        if (self.Name == 'beta'):
            return ((self.Par[1] * self.Par[2] + self.Par[0] * self.Par[3]) /
                    (self.Par[0] + self.Par[1]))

        if (self.Name == 'gevmin'):
            return -self.Dist.mean()

        else:
            return self.Dist.mean()

    def std(self):
        if (self.Name == 'beta'):
            return self.Dist.std() * (self.Par[3]-self.Par[2])
        return self.Dist.std()

    def pdf(self, x):

        if (self.Name == 'binomial'):
            return self.Dist.pmf(x)

        elif (self.Name == 'geometric'):
            return self.Dist.pmf(x)

        elif (self.Name == 'negativebinomial'):
            return self.Dist.pmf(x - self.Par[0])

        elif (self.Name == 'poisson'):
            return self.Dist.pmf(x)

        elif (self.Name == 'beta'):
            '''
            I believe there is a mistake in the matlab implementation of
            ERADist since the pdf value in the center is the same no matter
            if the support is [0, 1] or [0, 2]
            '''
            return (self.Dist.pdf((x - self.Par[2])
                    / (self.Par[3] - self.Par[2]))
                    / (self.Par[3] - self.Par[2]))

        elif (self.Name == 'gevmin'):
            return self.Dist.pdf(-x)

        else:
            return self.Dist.pdf(x)

    def cdf(self, x):
        if (self.Name == 'negativebinomial'):
            return self.Dist.cdf(x - self.Par[0])

        if (self.Name == 'beta'):
            return self.Dist.cdf((x - self.Par[2])/(self.Par[3]-self.Par[2]))

        if (self.Name == 'gevmin'):
            return self.Dist.cdf(-x)  # <-- this is not a proper cdf !

        else:
            return self.Dist.cdf(x)

    def random(self, m, n=0):
        # Matlab ERADist returns mxm samples if n isnt given, is this behaviour wanted?
        if (n == 0):
            if (self.Name == 'binomial'):
                samples = scipy.stats.binom.rvs(int(self.Par[0]),
                                                p=self.Par[1],
                                                size=m)
                return samples

            elif (self.Name == 'negativebinomial'):
                samples = self.Dist.rvs(size=m) + self.Par[0]
                return samples

            elif (self.Name == 'beta'):
                samples = (self.Dist.rvs(size=m) *
                           (self.Par[3]-self.Par[2]) + self.Par[2])
                return samples

            elif (self.Name == 'gevmin'):
                return self.Dist.rvs(size=m)*(- 1)

            else:
                samples = self.Dist.rvs(size=m)
                return samples

    def icdf(self, y):
        if (self.Name == 'negativebinomial'):
            return self.Dist.ppf(y) + self.Par[0]

        elif (self.Name == 'beta'):
            return self.Dist.ppf(y) * (self.Par[3] - self.Par[2]) + self.Par[2]

        elif (self.Name == 'gevmin'):
            return -self.Dist.ppf(y)

        else:
            return self.Dist.ppf(y)