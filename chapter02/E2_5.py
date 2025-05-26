import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, N):
        self.N = N
        self.Q_true = np.zeros(N)
        self.std = 1
        self.var_std = 0.01 
    
    def update(self):
        self.Q_true += np.random.normal(0, self.var_std, size=self.N)
    
    def step(self, act):
        return np.random.normal(self.Q_true[act], self.std)

np.random.seed(2025)
N, T, N_repeat = 10, 10000, 2000
eps, alpha = 0.1, 0.1
R_sa_avg, R_cs_avg = np.zeros(T), np.zeros(T)
Opt_sa, Opt_cs = np.zeros(T), np.zeros(T)
for _ in range(N_repeat):
    Q_sa, N_sa, R_sa = np.zeros(N), np.zeros(N), np.zeros(T)
    Q_cs, R_cs = np.zeros(N), np.zeros(T)
    bandit = Bandit(N)
    for t in range(T):
        # non-stationary bandit
        bandit.update()
        # sample-average
        if np.random.rand() < eps:
            act = np.random.randint(0, N)
        else:
            act = np.argmax(Q_sa)
        if act == np.argmax(bandit.Q_true):
            Opt_sa[t] += 1
        R_sa[t] = bandit.step(act)
        N_sa[act] += 1
        Q_sa[act] = Q_sa[act] + (1/N_sa[act]) * (R_sa[t] - Q_sa[act])
        # constant step-size
        if np.random.rand() < eps:
            act = np.random.randint(0, N)
        else:
            act = np.argmax(Q_cs)
        if act == np.argmax(bandit.Q_true):
            Opt_cs[t] += 1
        R_cs[t] = bandit.step(act)
        Q_cs[act] = Q_cs[act] + alpha * (R_cs[t] - Q_cs[act])
    R_sa_avg += R_sa / N_repeat
    R_cs_avg += R_cs / N_repeat
Opt_sa /= N_repeat
Opt_cs /= N_repeat

plt.figure(1, dpi=800)
plt.plot(range(1, T+1), R_sa_avg, label='sample-average')
plt.plot(range(1, T+1), R_cs_avg, label='constant step-size')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()
plt.grid()
plt.savefig('E2_5_reward.pdf')

plt.figure(2, dpi=800)
plt.plot(range(1, T+1), Opt_sa, label='sample-average')
plt.plot(range(1, T+1), Opt_cs, label='constant step-size')
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.legend()
plt.grid()
plt.savefig('E2_5_opt.pdf')

plt.show()
