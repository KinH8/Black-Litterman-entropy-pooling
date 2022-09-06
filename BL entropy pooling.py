# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:14:57 2022

@author: Wu
"""
import numpy as np
import pandas as pd
import scipy.optimize as spo

def market_cap_scalar(Pw,w):
    for x,y in enumerate(Pw):
        temp = np.multiply(np.sign(y),w)
        pos = np.where(temp>0,temp,0)
        neg = np.where(temp<0,temp,0)
        
        if not np.any(neg):
            Pw[x] = pos/pos.sum()
        elif not np.any(pos):
            Pw[x] = -neg/neg.sum()
        else:
            Pw[x] = pos/pos.sum() + -neg/neg.sum()

    return Pw

def objective(X, data):
    cov_matrix_d_sample = data.cov()

    port_volatility = np.sqrt(np.dot(X.T, np.dot(cov_matrix_d_sample, X)))
    
    port_return = data.mean().dot(X)

    sample_size = np.sqrt(252)
    
    sharpe = sample_size * port_return / port_volatility

    return -sharpe

def optimizer(weights, data):
    cons = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) -1  })
    bnds = [(0, 1)] * len(weights)    # https://stackoverflow.com/questions/29150064/python-scipy-indexerror-the-length-of-bounds-is-not-compatible-with-that-of-x0
    best_mix = spo.minimize(objective, weights, args=(data), method='SLSQP', bounds = bnds, constraints = cons, options={'disp':False})

    return best_mix.x

def entropy_pooling(x0, p, H,h,F,f):
    inq_mat = -np.eye(3)
    inq_mat = inq_mat[:2,:]
    
    inq_constraint = lambda x: inq_mat.dot(x)
    jac_constraint = lambda x: inq_mat
    
    def gradient(lv, p, H, h, F, f):
        lv = lv.reshape(len(lv), 1)
        
        k = F.shape[0]
        l = lv[:k]
        v = lv[k:]
        x = np.exp(np.log(p) - 1 - F.T.dot(l) - H.T.dot(v))
        x = np.clip(x, 1e-32, np.inf)
    
        return np.vstack((f - F.dot(x), h - H.dot(x)))
    
    def fx(lv, p, H, h, F, f):
        lv = lv.reshape(len(lv),1)
        
        k = F.shape[0]
        l = lv[:k]
        v = lv[k:]
        
        x = np.exp(np.log(p)-1-H.T.dot(v)-F.T.dot(l))
        x = np.clip(x, 1e-32, np.inf)
    
        L = x.T.dot(np.log(x)-np.log(p)) + l.T.dot(F.dot(x) - f) + v.T.dot(H.dot(x) - h)
    
        return -L
    
    cons = {'type': 'ineq', 
            'fun': inq_constraint, 
            'jac': jac_constraint}
    
    res = spo.minimize(fx, x0, method='SLSQP', jac=gradient, args=(p, H, h, F, f), constraints = cons, tol=1e-6, options = {'ftol': 1e2 * np.finfo(float).eps})
    k_ = F.shape[0]
    lv = res.x
    lv = lv.reshape(len(lv), 1)
    l = lv[0:k_]
    v = lv[k_:]
    p_ = np.exp(np.log(p) - 1 - F.T.dot(l) - H.T.dot(v))
    return p_

def merge_prior_posterior(p, p_, x, c):
    if (c < -1e8) or (c > (1 + 1e-8)):
        raise Exception("Confidence must be in [0, 1]")

    j, n = x.shape

    p_ = (1. - c) * p + c * p_
    exps = x.transpose().dot(p_)

    scnd_mom = x.transpose().dot(np.multiply(x, p_.dot(np.ones([1, n]))))
    scnd_mom = (scnd_mom + scnd_mom.transpose()) / 2

    covs = scnd_mom - exps.dot(exps.transpose())

    return exps, covs

# K = number of views
# N = number of assets

np.random.seed(5)

# Generate synthetic data
stocks = ['A','B','C']
ret = [0.02,0.03,0.07]
sigma = [0.15,0.2,0.3]

n = 1000
# Generated daily historical returns; assume normally distributed
data = pd.DataFrame(data={'A':np.random.normal(ret[0],sigma[0],n),
                         'B':np.random.normal(ret[1],sigma[1],n),
                         'C':np.random.normal(ret[2],sigma[2],n)})
cov = data.cov()

# Market cap weights
w = np.array([1/len(stocks)]*3)  # assume equal weights

# Scalar tau varies from 0 to 1; we fall back to 0.025 as in origianl BL paper
tau = 0.025

# Inverse of tau (scalar) * sigma
scaled_cov = np.linalg.inv(tau*cov) # NxN matrix of excess return; assume risk free is 0 for simplicity

# Idzorek uses a market cap weighting scheme, e.g. if the P entry is
# [0.5,-1,0.5] then a market cap weight may look like [0.7,-1,0.3].
P = np.array([[0,0,1],[1,-1,0]])  # KxN matrix that identifies the asset involved in the view
Pw = market_cap_scalar(P,w)
print('We assume asset C outperform all other assets by 5%, and asset A will outperform B by 2.5%')

# Q = expected return of the portfolios from the views described in P (Kx1)
# C to outperform all other assets by 5%; A to outpeform B by 2.5%
Q = np.array([0.05,0.025])
Q = Q.reshape(len(Q),1)

# Now calculate implied excess equilibrium return vector (Nx1)
# First, calculate risk aversion coefficient lambda given by (E(r)-rf)/sigma^2
# Assume 5% excess benchmark return & ~2% variance; sidenote BBG calculates it from aggregate corporate earnings + div yield
lam = 0.05/0.01667
Pi = lam * cov.dot(w)

# Calculate diagonal covariance matrix Omega
omega = tau * Pw.dot(cov).dot(Pw.T)
omega = np.diag(np.diag(omega))

# Finally calculate E(R), a Nx1 vector of assets under Black-Litterman model
ER1 = scaled_cov + Pw.T.dot(np.linalg.inv(omega)).dot(Pw)
ER1 = np.linalg.inv(ER1) 

ER2 = scaled_cov.dot(Pi) + np.hstack(Pw.T.dot(np.linalg.inv(omega)).dot(Q))  

ER = ER1.dot(ER2)

### Entropy pooling ###

j = 100000
x = np.random.multivariate_normal(ret, cov, j)

p = np.ones([j,1]) / j

#indicator = data.corrwith(data.shift()).rank()  # Example. Can be any signal
indicator = [0,1,2]

x0 = np.zeros([len(indicator),1])  # number of views

# Equality constraints in matrix-vector pair H,h
H = np.ones([1,j])
h = np.array([1.])

# Inequality constraints in matrix-vector pair F,f
F = np.diff(x, axis=1).T
f = np.zeros([F.shape[0],1])

p_ = entropy_pooling(x0, p, H,h,F,f)

mu_ep, sigma_ep = merge_prior_posterior(p, p_, x, 1)
mu_ep = mu_ep.flatten()
#######################


# Aggregate data
agg_ret = pd.DataFrame([ret, Pi, ER, mu_ep], columns=stocks,index=['Historical returns (synthetic)',\
                        'Implied equilibrium returns','BL new combined returns','Entropy pooling (C>B>A)'])


cov_est = cov + ER1
w_est = np.linalg.inv(cov_est * lam).dot(Pi)
w_est = w_est/w_est.sum()
w_mvo = optimizer(w, data)

w_ep=np.linalg.inv(sigma_ep * lam).dot(Pi)
w_ep = w_ep/w_ep.sum()

agg_weights = pd.DataFrame([w,w_est,w_mvo,w_ep],columns=stocks,\
                   index=['Market cap weights','Black Litterman weights',\
                          'Traditional MVO','Entropy pooling weights'])

print(agg_ret, '\n')
print(agg_weights)