import numpy as np
from scipy import stats
import data_preparation
from snow_models import TIndex
import ERADist as ED
from ERANataf import ERANataf

def aBUS_SUS(N,p0,log_likelihood,T_nataf):
    dim = len(T_nataf.Marginals) + 1

    # Initialization of variables
    i = 0
    lam = 0.6
#    leval   = np.zeros(1,N)  # space for the log-likelihood evaluations
    ll   = np.array([])   # space for the parameter log(c_{i}^{-1})
    h    = np.array([])   # space for the intermediate leveles
    prob = np.array([])   # space for the failure probability at each level
    Nf   = np.array([])   # space for the number of failure point per level
    samplesU_total = list()
    samplesU_seeds = list()
    geval = np.zeros(N)
    # Likelihood Function in standard normal space
    log_L = lambda u: log_likelihood(T_nataf.U2X(u[:-1]))
    def lsf(u, log_c):
        geval = np.log(stats.norm.cdf([u[-1, :]])) - log_c - log_L(u)
        return geval.flatten()
    #lsf = lambda u, log_c: np.log(stats.norm.cdf([u[-1, :]])) - log_c - log_L(u)

    # SUS procedure
    ## Initial MCS - Step
    uj = np.random.rand(dim, N)
    leval = log_L(uj)
    ll = np.append(ll, max(leval))
    log_c = -ll[-1]
    for j in range(N):
        geval[j] = lsf(np.reshape(uj[:, j], [6,1]), log_c)

    Nf = np.append(Nf, np.sum(geval <= 0))

    h = np.append(h, np.inf)
    while (h[-1] > 0):
        i = i + 1
#        print 'Scaling constant log_c = ' + str(log_c)

        # sort values in ascending order
        order = np.argsort(geval)
        uj = uj[:, order]
        # choose the intermediate level
        h = np.append(h, np.percentile(geval, p0*100))

        #print geval < h[-1]
        # number of failure points
        Nf = np.append(Nf, np.sum(geval <= np.max([h[-1],0])))
        #print Nf[-1]
        # assign conditional probability to the level
        if (h[-1] < 0):
            h[-1] = 0
            prob = np.append(prob, Nf[-1]/N)
        else:
            prob = np.append(prob, p0)

        # store ordered samples
        samplesU_total.append(uj)
        samplesU_seeds.append(uj[:,:int(Nf[-1])])

        # randomize the ordering of the samples (to avoid possible bias)
        idx_rnd = np.random.permutation(int(Nf[-1]))
        rnd_seed = samplesU_seeds[-1][:, idx_rnd]

        # sampling process using adaptive conditional sampling
        uj, leval, geval, lam = aCS(N, lam, h[-1], rnd_seed, log_L, lsf, log_c)
        #print geval < h[-1]

        # update the value of the scaling constant c
        ll = np.append(ll, np.max([ll[-1], np.max(leval)]))
        log_c = -ll[-1]

        # adjust the intermediate level
        # print ll[i] - ll[i-1]

        h[-1] = h[-1] + (ll[-1] - ll[-2])
        print ('Threshold level ' + str(i - 1) + ' = ' + str(np.round(h[-1])))

        # decrease the dependence of the samples
        p = np.random.uniform(low=np.zeros(N),
                              high=np.min([np.ones(N),
                                           np.exp(leval - ll[i] + h[i])],
                                          axis=0))

        #uj[-1, :] = stats.norm.ppf(p) # the problem is here!!!! 

    # number of intermediate levels
    m = i

    # store final posterior samples
    samplesU_total.append(uj)

    # acceptance probability and model evidence
    p_acc = np.prod(prob)
    cE = p_acc*np.exp(ll[m])

    # transform the samples to the physical space
    samplesX_total = list()
    for i in range(m):
        p = stats.norm.cdf(samplesU_total[i][-1, :])
        samplesX_total.append(np.array([T_nataf.U2X(samplesU_total[i][:-1, :]), p]))

    return (h, samplesU_total, samplesX_total, cE)


def aCS(N, lamb, b, u, log_likelihood, lsf, log_c):
    # Initialize variables
    # pa = 0.1;
    dim = u.shape[0]  # number of uncertain parameters
    Ns = u.shape[1]  # number of seeds
    Na = np.ceil(100*Ns/N)  # number of chains after which the proposal is adapted (Na = pa*Ns)
    Nchain = np.ones([Ns, 1])*np.floor(N/Ns)  # number of samples per chain
    Nchain[:(N % Ns),:] = Nchain[: N % Ns]+1
    #print Nchain
    # need this step to adjust the indexing in the last level
    uj = u

    # initialization
    ujk = np.zeros([dim, N])  # generated samples
    geval = np.zeros(N)  #store lsf evaluations
    geval_2 = np.zeros(Ns)
    for j in range(Ns):
        geval_2[j] = lsf(np.reshape(uj[:, j], [dim, 1]), log_c)
    # print geval_2 < b
    
    leval = np.zeros(N)  # store likelihood evaluations
    acc = np.zeros(N)  # store acceptance
    mu_acc = np.zeros(int(np.floor(Ns/Na))+1)  # store acceptance
    hat_a = np.zeros(int(np.floor(Ns/Na)))  # average acceptance rate of the chains
    lam = np.zeros(int(np.floor(Ns/Na))+1)  # scaling parameter \in (0,1)

    # 1. compute the standard deviation
    sigma_0 = np.std(uj, axis=1)
    # 2. iteration
    star_a = 0.44  # optimal acceptance rate 
    lam[0] = lamb  # initial scaling parameter \in (0,1)

    # a. compute correlation parameter
    i = 0  # index for adaptation of lambda
    sigma = np.min([lam[i]*sigma_0, np.ones(dim)], axis = 0) # Ref. 1 Eq. 23
    rho = np.sqrt(1-sigma**2)  #  Ref. 1 Eq. 24

    mu_acc[i] = 0

    # b. apply conditional sampling
    for k in range(Ns):
        idx = int(np.sum(Nchain[:k]))  # ((k-1)/pa+1)
        acc[idx] = 0  # store acceptance
        ujk[:, idx] = uj[:, k]  # pick a seed at random

        # evaluate likelihood and LSF
        leval[idx] = log_likelihood(np.reshape(ujk[:, idx], [dim, 1]))
        geval[idx] = lsf(np.reshape(ujk[:, idx], [dim, 1]), log_c)
        for t in np.arange(1, int(Nchain[k])):
            # generate candidate sample
            v = np.random.normal(loc=rho*ujk[:, idx+t-1], scale=sigma)

            # accept or reject sample
            Le = log_likelihood(np.reshape(v, [dim, 1]))
            He = lsf(np.reshape(v, [dim, 1]), log_c)

            if (He <= b):

                ujk[:, idx+t] = v  # accept the candidate in failure region
                acc[idx+t] = 1  # note the acceptance
                leval[idx+t] = Le  # store the likelihood evaluation
                geval[idx+t] = He  # store the lsf evaluation
            else:
                ujk[:, idx+t] = ujk[:, idx+t-1]  # reject the candidate
                acc[idx+t] = 0  # note the rejection
                leval[idx+t] = leval[idx+t-1]  # store the likelihood eval
                geval[idx+t] = geval[idx+t-1]  # store the lsf evaluation

        # average of the accepted samples for each seed
        mu_acc[i] = mu_acc[i] + np.min([1, np.mean(acc[idx:int(idx+Nchain[k]-1)])])
        if (k % Na == 0 and k > 0):
            # c. evaluate average acceptance rate
            hat_a[i] = mu_acc[i]/Na  # Ref. 1 Eq. 25

            # d. compute new scaling parameter
            zeta = 1/np.sqrt(i + 1)  # ensures that the variation of lam[i] vanishes
            lam[i+1] = np.exp(np.log(lam[i]) + zeta*(hat_a[i]-star_a))  # Ref. 1 Eq. 26
            # update parameters
            sigma = np.min([lam[i+1]*sigma_0, np.ones(dim)])  # Ref. 1 Eq. 23
            rho = np.sqrt(1-sigma**2)  # Ref. 1 Eq. 24

            # update counter
            i = i+1
    # for ii in range(N):
    #     print lsf(np.reshape(ujk[:, ii], [dim, 1]), log_c) < b

        
    # next level lambda
    new_lambda = lam[i]
    # compute mean acceptance rate of all chains
    accrate = np.mean(hat_a)
    return (ujk, leval, geval, new_lambda)