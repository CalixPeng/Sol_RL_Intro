import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from datetime import datetime
from fractions import Fraction

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

def one_case(N, T, Eps, Alpha, C, Q_0, task_id):
    # eps-greedy, sample average
    R_sa = np.zeros(Eps.size)
    for i_eps, eps in enumerate(Eps):
        bandit = Bandit(N)
        Q_sa, N_sa = np.zeros(N), np.zeros(N)
        for t in range(T):
            bandit.update()
            if np.random.rand() < eps:
                act = np.random.randint(0, N)
            else:
                act = np.argmax(Q_sa)
            reward = bandit.step(act)
            N_sa[act] += 1
            Q_sa[act] = Q_sa[act] + (1/N_sa[act]) * (reward - Q_sa[act])
            if t >= T/2:
                R_sa[i_eps] += reward
    # eps-greedy, constant step-size
    R_cs = np.zeros(Eps.size)
    for i_eps, eps in enumerate(Eps):
        bandit = Bandit(N)
        Q_cs = np.zeros(N)
        for t in range(T):
            bandit.update()
            if np.random.rand() < eps:
                act = np.random.randint(0, N)
            else:
                act = np.argmax(Q_cs)
            reward = bandit.step(act)
            Q_cs[act] = Q_cs[act] + 0.1 * (reward - Q_cs[act])
            if t >= T/2:
                R_cs[i_eps] += reward
    # gradient bandit
    R_gb = np.zeros(Alpha.size)
    for i_alpha, alpha in enumerate(Alpha):
        bandit = Bandit(N)
        H_gb = np.zeros(N)
        base = 0
        for t in range(T):
            bandit.update()
            Prob = np.exp(H_gb) / np.sum(np.exp(H_gb))
            act = np.random.choice(np.arange(N), p=Prob)
            reward = bandit.step(act)
            H_gb += alpha * (reward - base) * ((np.arange(N) == act) - Prob)
            base = base + (1 / (t + 1)) * (reward - base)
            if t >= T/2:
                R_gb[i_alpha] += reward
    # UCB
    R_ucb = np.zeros(C.size)
    for i_c, c in enumerate(C):
        bandit = Bandit(N)
        Q_ucb, N_ucb = np.zeros(N), np.zeros(N)
        for t in range(T):
            bandit.update()
            if t < N:
                act = t
            else:
                act = np.argmax(Q_ucb + c * np.sqrt(np.log(t) / N_ucb))
            reward = bandit.step(act)
            N_ucb[act] += 1
            Q_ucb[act] = Q_ucb[act] + (1/N_ucb[act]) * (reward - Q_ucb[act])
            if t >= T/2:
                R_ucb[i_c] += reward
    # greedy with optimistic initialization
    R_oi = np.zeros(Q_0.size)
    for i_q, q_0 in enumerate(Q_0):
        bandit = Bandit(N)
        Q_oi = q_0 * np.ones(N)
        for t in range(T):
            bandit.update()
            act = np.argmax(Q_oi)
            reward = bandit.step(act)
            Q_oi[act] = Q_oi[act] + 0.1 * (reward - Q_oi[act])
            if t >= T/2:
                R_oi[i_q] += reward
    return R_sa, R_cs, R_gb, R_ucb, R_oi

if __name__ == '__main__':
    np.random.seed(2025)
    N, T = 10, 200000
    N_repeat, N_worker = 2000, 24
    Eps = np.array([1/256, 1/128, 1/64, 1/32, 1/16, 1/8])
    Alpha = np.array([1/128, 1/64, 1/32, 1/16, 1/8, 1/4])
    C = np.array([2, 4, 8, 16, 32, 64])
    Q_0 = np.array([1/8, 1/4, 1/2, 1, 2, 4])
    func = partial(one_case, N, T, Eps, Alpha, C, Q_0)
    print('Start at ' + datetime.now().strftime('%H:%M:%S'))
    p = Pool(N_worker)
    R_sa, R_cs = np.zeros(Eps.size), np.zeros(Eps.size)
    R_gb, R_ucb, R_oi = np.zeros(Alpha.size), np.zeros(C.size), np.zeros(Q_0.size)
    complete = 0
    for Result in p.imap(func, range(N_repeat)):
        R_sa += Result[0] / ((T / 2) * N_repeat)
        R_cs += Result[1] / ((T / 2) * N_repeat)
        R_gb += Result[2] / ((T / 2) * N_repeat)
        R_ucb += Result[3] / ((T / 2) * N_repeat)
        R_oi += Result[4] / ((T / 2) * N_repeat)
        complete += 1
        print(f'\r{complete}/{N_repeat}, ' + datetime.now().strftime('%H:%M:%S'), 
              end='', flush=True)
    p.close()
    
    Parameter = np.array([1/256, 1/64, 1/16, 1/4, 1, 4, 16, 64])
    plt.figure(1, dpi=800)
    plt.plot(Eps, R_sa, 'o-', label=r'$\epsilon$-greedy, sample-avg')
    plt.plot(Eps, R_cs, 'o-', label=r'$\epsilon$-greedy, const step')
    plt.plot(Alpha, R_gb, 'o-', label='gradient bandit')
    plt.plot(C, R_ucb, 'o-', label='UCB')
    plt.plot(Q_0, R_oi, 'o-', label='optim init')
    plt.xscale('log')
    x_tick = [Fraction(item).limit_denominator() for item in Parameter]
    plt.xticks(np.unique(Parameter), x_tick)
    plt.xlabel(r'$\epsilon$, $\alpha$, c, $Q_0$')
    plt.ylabel('Average reward')
    plt.legend(ncol=2, columnspacing=0.2, labelspacing=0.2, loc='upper right', 
               bbox_to_anchor=(1, 1))
    plt.grid()
    plt.savefig('E2_11_reward.pdf')
    plt.show()
