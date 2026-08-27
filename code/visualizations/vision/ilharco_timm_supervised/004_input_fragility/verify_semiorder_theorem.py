"""Brute-force verification of the semiorder characterization.

  Claim:  { rankings reachable by z + delta, ||delta||_inf <= eps }
        = { linear extensions of P_eps },   P_eps:  i < j  iff  z_i - z_j >= 2 eps

Reachability is tested EXACTLY via its pairwise characterization (no floating
tolerance): a strictly decreasing t with |t_k - z_{c_k}| <= eps exists iff
z_{c_k} - z_{c_j} < 2 eps for every j preceding k.  Derivation: necessity from
t_j <= z_{c_j}+eps, t_k >= z_{c_k}-eps, t_j > t_k; sufficiency by the greedy
t_k = min(z_{c_k}+eps, t_{k-1}-eta) for small enough eta.

NOTE: the >= in P_eps is load-bearing. With a strict > the claim is FALSE at
z_i - z_j == 2 eps exactly: the pair is then incomparable (so both orders are
linear extensions) but only one order is reachable.
"""
import itertools
from fractions import Fraction as F
import random

def reachable(z, perm, eps):
    for a in range(len(perm)):
        for b in range(a + 1, len(perm)):
            if z[perm[b]] - z[perm[a]] >= 2 * eps:      # strict < required
                return False
    return True

def is_linear_extension(z, perm, eps):
    pos = {c: p for p, c in enumerate(perm)}
    C = len(z)
    for i in range(C):
        for j in range(C):
            if i != j and z[i] - z[j] >= 2 * eps and pos[i] > pos[j]:
                return False
    return True

def sweep(instances, label):
    mism = tot = 0
    for z, eps in instances:
        for perm in itertools.permutations(range(len(z))):
            r, l = reachable(z, perm, eps), is_linear_extension(z, perm, eps)
            tot += 1
            if r != l:
                mism += 1
                if mism <= 3:
                    print(f"    MISMATCH z={z} eps={eps} perm={perm} reach={r} linext={l}")
    print(f"  {label}: {tot:,} permutations, {mism} mismatches"
          f"  {'OK' if mism == 0 else '<-- FALSE'}")
    return mism

rng = random.Random(0)
# 1. generic position, exact rationals
gen = []
for _ in range(3000):
    C = rng.randint(3, 6)
    gen.append(([F(rng.randint(-400, 400), 100) for _ in range(C)],
                F(rng.randint(5, 120), 100)))
# 2. adversarial: many pairs sitting EXACTLY at the 2*eps boundary
bnd = []
for _ in range(3000):
    C = rng.randint(3, 6)
    eps = F(rng.randint(1, 20), 10)
    base = F(rng.randint(-20, 20), 10)
    bnd.append(([base + 2 * eps * rng.randint(0, 2) for _ in range(C)], eps))

m1 = sweep(gen, "generic position ")
m2 = sweep(bnd, "boundary-saturated")
print(f"\n  characterization {'VERIFIED (both regimes)' if m1 + m2 == 0 else 'FALSIFIED'}")
