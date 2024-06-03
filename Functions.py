import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import hmc_Lab as hmc


def compute_posterior(X, y, alph, beta):
    Mu = np.array((np.linalg.inv((X.T@X) + ((1/beta)*alph*np.identity(9))) @ (X.T @ y)).ravel())
    SIGMA = np.array((1/beta) * np.linalg.inv((X.T@X) + ((1/beta)*alph*np.identity(9))))
    return Mu, SIGMA

def calc_RMSE(y, y_est):
    RMSE = np.sqrt(np.mean((y - y_est) ** 2))
    return RMSE

def calc_MAE(y, y_est):
    abs_err = np.abs(y_est-y)
    MAE = np.mean(abs_err)
    return MAE

def compute_log_marginal(X, y, alph, beta):
    N, M = X.shape

    C = 1/beta * np.eye(N) + X @ X.T / alph

    lgp = -N/2 * np.log(2*np.pi)

    _, log_det = np.linalg.slogdet(C)

    lgp -= log_det / 2

    lgp -= y.T @ np.linalg.inv(C) @ y / 2
    return lgp


def energy_func_lr(hps, x, y):
    w = hps[2:]
    alpha = hps[0]
    beta = hps[1]
    w_X = x@w
    N, M = x.shape
    likelihood = -N*np.log(2*np.pi)/2 + N*(beta)/2 - (np.exp(beta)/2 * np.sum((y-w_X)**2))
    prior = M*(alpha)/2 - (M*np.log((2*np.pi))/2) - ((np.exp(alpha)/2)*np.sum(w**2))
    neglgp = -(likelihood+prior)
    return neglgp

def energy_grad_lr(hps, x, y):
    
    w = hps[2:]
    alpha = hps[0]
    beta = hps[1]
    N, M = x.shape
    w_X = x@w

    dl_da = -(M/2 - (np.exp(alpha)*np.sum(w**2)/2))
    dl_dB = -(N/2 - (np.exp(beta)/2 * np.sum((y-w_X)**2)))
    d_dw = -(((y-w_X)@x)*np.exp(beta) - w*np.exp(alpha))
    
    a_b = [dl_da, dl_dB]
    g = np.array(a_b + list(d_dw))
   
    return g

def transform_y(y_train, y_test):
    y_log_train = np.array([1 if y>23 else 0 for y in y_train])
    y_log_test = np.array([1 if y>23 else 0 for y in y_test])

    return y_log_train, y_log_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bernoulli_likelihood(X, y):
    p_success = sigmoid(X)
    pmf = np.sum(stats.bernoulli.logpmf(y, p_success))
    return pmf

def energy_func_logistic(hps, x, y):
    alpha = hps[0]
    w = hps[1:]
    M = len(w)
    w_X = x@w
    prior = M*(alpha)/2 - (M*np.log((2*np.pi))/2) - ((np.exp(alpha)/2)*np.sum(w**2))
    neglgp = -(bernoulli_likelihood(w_X, y)+prior)
    return neglgp

def energy_grad_logistic(hps, x, y):
    alpha = hps[0]
    w = hps[1:]
    M = len(w)
    s_w_x = sigmoid(x@w)
    dl_da = [-(M/2 - (np.exp(alpha)*np.sum(w**2)/2))]
    d_w = -((y-s_w_x)@x - np.exp(alpha)*w)
    g = np.array(dl_da + list(d_w))
    
    return g

def logistic_error(y_pred, y_act):
    return 100*(np.sum(np.abs(y_pred-y_act))/len(y_pred))

def VI(X_train, Y_train):
    a0 = 0.0001
    b0 = 0.0001
    c0 = 0.0001
    d0 = 0.0001
    
    an = 0.0001
    bn = 0.0001
    cn = 0.0001
    dn = 0.0001
    mu_0 = 0
    beta_0 = 1

    N, M = X_train.shape
    an = a0 + M / 2
    cn = c0 + N / 2

    max_iter = 1000
    for _ in range(max_iter):
        s2 = 1/(cn/dn)
        e_alpha = an/bn
        sig_n = np.linalg.inv(((X_train.T@X_train)/(s2)) + (e_alpha*np.eye(M)))
        mu_n = (sig_n@(X_train.T@Y_train))/(s2)
        wTw = (mu_n.T@mu_n) + np.trace(sig_n)
        bn = b0 + 0.5 * (wTw)
        mu_B = (beta_0*mu_0 + N*np.mean(X_train))/(beta_0+N)
        dn = d0 + 0.5*np.sum((X_train-mu_B)**2 + beta_0*(mu_B-mu_0)**2)
    
    return an,bn,cn,dn, mu_n, sig_n