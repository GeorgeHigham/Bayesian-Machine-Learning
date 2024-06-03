import Functions as f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hmc_Lab as hmc

dat = pd.read_csv("ENB2012_data.csv")
train_dat = dat[:384]
test_dat = dat[384:]

X_train_dat = train_dat[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y_train = np.array(train_dat['Y1'].tolist())

X_test_dat = test_dat[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y_test = np.array(test_dat['Y1'].tolist())

X_train_norm_dat = (X_train_dat - X_train_dat.mean()) / X_train_dat.std()
X_train_norm_dat['const'] = 1
X_train = np.array(X_train_norm_dat)
print(X_train.shape)
print(y_train.shape)
X_test_norm_dat = (X_test_dat - X_train_dat.mean()) / X_train_dat.std()
X_test_norm_dat['const'] = 1
X_test = np.array(X_test_norm_dat)


log_a_range = np.linspace(-5, 0, 100)
log_B_range = np.linspace(-5, 0, 100)
a_range = np.exp(log_a_range)
B_range = np.exp(log_B_range)

log_prob_y = np.zeros((len(log_a_range), len(log_B_range)))
for i, a_ in enumerate(a_range):
    for j, B_ in enumerate(B_range):
        log_prob_y[j, i]  = f.compute_log_marginal(X_train, y_train, a_, B_)

ind = np.unravel_index(np.argmax(log_prob_y), log_prob_y.shape)
mp_a = a_range[ind[1]]
mp_B = B_range[ind[0]]
mp_l_a = log_a_range[ind[1]]
mp_l_B = log_B_range[ind[0]]

print(f"\nBayesian Linear Regression Results\n")

print(f"Most probable Alpha: {mp_a}")
print(f"Most probable Beta: {mp_B}")
print(f"Most probable log Alpha: {mp_l_a}")
print(f"Most probable log Beta: {mp_l_B}")

plt.contourf(log_a_range, log_B_range, log_prob_y)
plt.plot(mp_l_a,mp_l_B, 'ro')
plt.xlabel('log Alpha')
plt.ylabel('log Beta')
plt.title("Log Probability of Alpha & Beta given Y")

plt.savefig('Heating Contour Graph.png')

Mu, SIGMA = f.compute_posterior(X_train, y_train, mp_a, mp_B)
y_train_post = Mu @ X_train.T
y_test_post = Mu @ X_test.T
RMSE_train = f.calc_RMSE(y_train, y_train_post)
RMSE_test = f.calc_RMSE(y_test, y_test_post)
MAE_train = f.calc_MAE(y_train, y_train_post)
MAE_test = f.calc_MAE(y_test, y_test_post)
print(f"Weight values: {Mu}")
print(f"Training RMSE: {RMSE_train}")
print(f"Test RMSE: {RMSE_test}")
print(f"Training MAE: {MAE_train}")
print(f"Test MAE: {MAE_test}")

np.random.seed(seed=1) 

R = 8500
L = 20
eps = 0.0085
# high acceptance rate because of high burn
burn = int(R/5)

hps = np.random.uniform(0,1, size=(11,1)).ravel()

print(f"\nHMC Sampling Regression Results\n")

S, *_ = hmc.sample(hps, f.energy_func_lr, f.energy_grad_lr, R, L, eps, burn=burn, checkgrad=True, args=[X_train, y_train])

ham_a = np.mean(S[:,0])
ham_B = np.mean(S[:,1])
Mu, SIGMA = f.compute_posterior(X_train, y_train, ham_a, ham_B)
y_train_post = Mu @ X_train.T
y_test_post = Mu @ X_test.T
train_err = f.calc_RMSE(y_train, y_train_post)
test_err = f.calc_RMSE(y_test, y_test_post)
mae_train_err = f.calc_MAE(y_train, y_train_post)
mae_test_err = f.calc_MAE(y_test, y_test_post)

print(f"\nWeight values: {Mu}")
print(f"Log Alpha = {ham_a}")
print(f"Log Beta = {ham_B}")
print(f"training RMSE: {train_err}")
print(f"test RMSE: {test_err}")
print(f"training MAE: {mae_train_err}")
print(f"test MAE: {mae_test_err}")
fsz = (10,8)
plt.figure(figsize=fsz)
plt.plot(S[:, 0], S[:, 1], '.', ms=6, color='CadetBlue', alpha=0.25, zorder=0)
plt.xlabel("Log Beta")
plt.ylabel("Log Alpha")

y_log_train, y_log_test = f.transform_y(y_train, y_test)

R = 10000
L = 25
eps = 0.17
burn = int(R/10)
np.random.seed(seed=1)
hps = (np.random.uniform(0, 1, size=(10, 1)).ravel())

print(f"\nHMC Classification Results\n")

S, *_ = hmc.sample(hps, f.energy_func_logistic, f.energy_grad_logistic, R, L, eps, burn=burn, checkgrad=True, args=[X_train, y_log_train])

log_alpha = S[-1][0]
weights = S[-1][1:]
x_weights_train = X_train @ weights
x_weights_test = X_test @ weights

y_train_pred = np.array([1 if x_w>0 else 0 for x_w in x_weights_train])
y_test_pred = np.array([1 if x_w>0 else 0 for x_w in x_weights_test])

def logistic_error(y_pred, y_act):
    return 100*(np.sum(np.abs(y_pred-y_act))/len(y_pred))

train_error = logistic_error(y_train_pred, y_log_train)
test_error = logistic_error(y_test_pred, y_log_test)

print(f"\nWeight values: {weights}")
print(f"Classification train error: {train_error}%")
print(f"Classification test error: {test_error}%")

an,bn,cn,dn, mu_n, sig_n = f.VI(X_train, y_train)
exp_alpha = an/bn
exp_beta = cn/dn

print(f"\nVariational Inference Results\n")
print(f"Expectation Alpha: {exp_alpha}")
print(f"Expectation Beta: {exp_beta}")
print(f"Expectation log Alpha: {np.log(exp_alpha)}")
print(f"Expectation log Beta: {np.log(exp_beta)}")
print(f"Weights: {mu_n}")

y_train_post = mu_n @ X_train.T
y_test_post = mu_n @ X_test.T
train_err = f.calc_RMSE(y_train, y_train_post)
test_err = f.calc_RMSE(y_test, y_test_post)
mae_train_err = f.calc_MAE(y_train, y_train_post)
mae_test_err = f.calc_MAE(y_test, y_test_post)
print(f"training RMSE: {train_err}")
print(f"test RMSE: {test_err}")
print(f"training MAE: {mae_train_err}")
print(f"test MAE: {mae_test_err}")