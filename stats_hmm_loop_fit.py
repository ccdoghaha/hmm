"""Gaussian HMM fits (hand-rolled EM, no deps) on the per-loop win series
of the two biggest tables in summary_step0.25_20260911T013630Z.csv."""
import numpy as np, pandas as pd

def fb(x, pi, A, mu, var):
    T, K = len(x), len(pi)
    B = np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]) \
        / np.sqrt(2 * np.pi * var[None, :])
    B = np.maximum(B, 1e-300)
    al = np.zeros((T, K)); c = np.zeros(T)
    al[0] = pi * B[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, T):
        al[t] = (al[t-1] @ A) * B[t]; c[t] = al[t].sum(); al[t] /= c[t]
    be = np.zeros((T, K)); be[-1] = 1.0
    for t in range(T-2, -1, -1):
        be[t] = (A @ (B[t+1] * be[t+1])) / c[t+1]
    ll = np.log(c).sum()
    g = al * be; g /= g.sum(1, keepdims=True)
    xi = np.zeros((K, K))
    for t in range(T-1):
        xi += (al[t][:, None] * A * (B[t+1] * be[t+1])[None, :]) / c[t+1]
    return ll, g, xi

def viterbi(x, pi, A, mu, var):
    T, K = len(x), len(pi)
    lb = -0.5 * ((x[:, None] - mu[None, :]) ** 2) / var[None, :] \
        - 0.5 * np.log(2 * np.pi * var[None, :])
    d = np.zeros((T, K)); psi = np.zeros((T, K), int)
    d[0] = np.log(pi + 1e-300) + lb[0]
    for t in range(1, T):
        m = d[t-1][:, None] + np.log(A + 1e-300)
        psi[t] = m.argmax(0)
        d[t] = m.max(0) + lb[t]
    s = np.zeros(T, int); s[-1] = d[-1].argmax()
    for t in range(T-2, -1, -1):
        s[t] = psi[t+1][s[t+1]]
    return s

def fit(x, K, restarts=40, iters=600, seed=0):
    x = np.asarray(x, float); T = len(x)
    best = None
    rng = np.random.RandomState(seed)
    for r in range(restarts):
        mu = np.percentile(x, np.linspace(15, 85, K)) + rng.randn(K) * x.std() * 0.05
        var = np.full(K, x.var() / K) * (0.5 + rng.rand(K))
        pi = np.full(K, 1.0 / K)
        A = rng.rand(K, K) + np.eye(K) * 2; A /= A.sum(1, keepdims=True)
        ll_old = -np.inf; ll = None
        for it in range(iters):
            ll, g, xi = fb(x, pi, A, mu, var)
            if ll - ll_old < 1e-9 and it > 5:
                break
            ll_old = ll
            pi = g[0] + 1e-12; pi /= pi.sum()
            A = xi + 1e-12; A /= A.sum(1, keepdims=True)
            w = g.sum(0) + 1e-12
            mu = (g * x[:, None]).sum(0) / w
            var = (g * (x[:, None] - mu[None, :]) ** 2).sum(0) / w + 1e-10
        if best is None or ll > best[0]:
            best = (ll, pi.copy(), A.copy(), mu.copy(), var.copy())
    ll, pi, A, mu, var = best
    order = np.argsort(mu)
    pi, mu, var = pi[order], mu[order], var[order]
    A = A[np.ix_(order, order)]
    return dict(ll=ll, pi=pi, A=A, mu=mu, var=var,
                vit=viterbi(x, pi, A, mu, var))

def report(name, x, K):
    r = fit(x, K)
    st = r['pi'] @ r['A']
    dur = 1.0 / (1.0 - np.diag(r['A']))
    print(f"\n--- {name}  K={K}  loglik={r['ll']:.1f}  "
          f"BIC={-2*r['ll'] + K*K*np.log(len(x)):.1f}")
    print("  means :", np.round(r['mu'], 3))
    print("  vars  :", np.round(r['var'], 3))
    print("  pi0   :", np.round(r['pi'], 3))
    print("  A     :")
    for row in np.round(r['A'], 3):
        print("    ", row)
    print("  stationary:", np.round(st, 3),
          "  exp. duration (loops):", np.round(dur, 1))
    v = r['vit']
    runs = []
    i = 0
    while i < len(v):
        j = i
        while j < len(v) and v[j] == v[i]:
            j += 1
        runs.append((int(v[i]), i, j - i, round(float(x[i:j].mean()), 2)))
        i = j
    n_hi = sum(1 for s, _, _, _ in runs if s == K - 1)
    print(f"  viterbi: {len(runs)} runs, {n_hi} top-state runs, "
          f"longest runs: "
          + "; ".join(f"st{s}@{a} n={n} x̄={m}" for s, a, n, m in
                      sorted(runs, key=lambda t: -t[2])[:5]))
    return r

f = r"report0251/summary_step0.25_20260911T013630Z.csv"
cols = ['win', 'loop_round', 'table_name']
df = pd.read_csv(f, usecols=cols)

# sanity: synthetic 2-state check
rng = np.random.RandomState(7)
s = 0; xs = []
for _ in range(3000):
    s = s if rng.rand() < (0.95 if s == 0 else 0.9) else 1 - s
    xs.append(rng.randn() + (0 if s == 0 else 6))
r = fit(np.array(xs), 2)
print("synthetic check: means", np.round(r['mu'], 2), "A diag",
      np.round(np.diag(r['A']), 3), "(expect ~0 and ~6; 0.95/0.9)")

t1 = 'R7_CB0_A1_525600_W1_ON_FULL_T1_R20_1440_43_15_10_BUY'
t2 = 'R6_CB0_A1_1051200_W1_ON_FULL_T1_R20_1440_43_15_10_BUY'
for t, nm in [(t1, 'R7_525600'), (t2, 'R6_1051200')]:
    L = df[df.table_name == t].groupby('loop_round')['win'].mean().sort_index()
    x = L.values
    print(f"\n===== {nm}  n={len(x)}  win: first={x[0]:.2f} "
          f"peak={x.max():.2f}@{x.argmax()} plateau(>=200)={x[200:].mean():.2f} "
          f"last={x[-1]:.2f}")
    for K in (2, 3, 4):
        report(nm, x, K)
