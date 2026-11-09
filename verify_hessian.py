"""
verify_hessian.py — independent check of the math used in index.html
(CSE 840, Lecture 16 concept demo: Hessian, critical points, least squares).

Run:  python3 verify_hessian.py
Needs: numpy
"""

import numpy as np

TOL = 1e-9
RNG = np.random.default_rng(840)  # seeded, reproducible


# ---------------------------------------------------------------------------
# Mode 1: f(x, y) = 1/2 a x^2 + b x y + 1/2 c y^2
# ---------------------------------------------------------------------------

def f_quad(a, b, c, x, y):
    return 0.5 * a * x * x + b * x * y + 0.5 * c * y * y


def eig_closed_form(a, b, c):
    """Same formulas as eigSym2() in index.html."""
    tr = a + c
    half = (a - c) / 2.0
    disc = np.sqrt(half * half + b * b)
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    phi = 0.5 * np.arctan2(2.0 * b, a - c)
    v1 = np.array([np.cos(phi), np.sin(phi)])
    v2 = np.array([-np.sin(phi), np.cos(phi)])
    return l1, l2, v1, v2


def classify(l1, l2):
    """Same rule as classify() in index.html."""
    if l2 > TOL:
        return "Strict Local Minimum"
    if l1 < -TOL:
        return "Strict Local Maximum"
    if l1 > TOL and l2 < -TOL:
        return "Saddle Point"
    return "Semidefinite / Inconclusive"


def fd_hessian(func, p, h=1e-4):
    """Central finite-difference Hessian of func: R^n -> R at p."""
    n = len(p)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h
            ej[j] = h
            H[i, j] = (func(p + ei + ej) - func(p + ei - ej)
                       - func(p - ei + ej) + func(p - ei - ej)) / (4 * h * h)
    return H


def check_matrix(a, b, c):
    H = np.array([[a, b], [b, c]])
    l1, l2, v1, v2 = eig_closed_form(a, b, c)

    # Eigenvalues vs numpy.linalg.eigh (ascending order)
    w, _ = np.linalg.eigh(H)
    assert np.allclose([l2, l1], w, atol=1e-9), (a, b, c, l1, l2, w)

    # Eigenvector equations H v = lambda v, orthonormality
    assert np.allclose(H @ v1, l1 * v1, atol=1e-9)
    assert np.allclose(H @ v2, l2 * v2, atol=1e-9)
    assert abs(v1 @ v2) < 1e-12 and abs(np.linalg.norm(v1) - 1) < 1e-12

    # Finite-difference Hessian of f equals [[a, b], [b, c]]
    func = lambda p: f_quad(a, b, c, p[0], p[1])
    p0 = RNG.uniform(-1, 1, size=2)
    assert np.allclose(fd_hessian(func, p0), H, atol=1e-5)

    # Directional curvature identity u^T H u = l1 (u.v1)^2 + l2 (u.v2)^2
    th = RNG.uniform(0, 2 * np.pi)
    u = np.array([np.cos(th), np.sin(th)])
    lhs = u @ H @ u
    rhs = l1 * (u @ v1) ** 2 + l2 * (u @ v2) ** 2
    assert abs(lhs - rhs) < 1e-9
    assert l2 - 1e-12 <= lhs <= l1 + 1e-12

    # Second derivative of g(t) = f(t u) at t = 0 by finite differences
    g = lambda t: f_quad(a, b, c, t * u[0], t * u[1])
    h = 1e-4
    g2 = (g(h) - 2 * g(0) + g(-h)) / (h * h)
    assert abs(g2 - lhs) < 1e-5

    # Newton step from any x0 lands on the critical point (0, 0) if H invertible
    if abs(np.linalg.det(H)) > 1e-6:
        x0 = RNG.uniform(-2, 2, size=2)
        x1 = x0 - np.linalg.solve(H, H @ x0)
        assert np.allclose(x1, 0, atol=1e-9)

    return l1, l2, v1, v2


def gd_converges(a, b, c, eta, x0, steps=2000):
    H = np.array([[a, b], [b, c]])
    x = np.array(x0, dtype=float)
    for _ in range(steps):
        x = x - eta * (H @ x)
        if np.linalg.norm(x) > 1e8:
            return False
    return np.linalg.norm(x) < 1e-6


def main():
    print("== Mode 1: presets ==")
    presets = {
        "Strict Local Min (a=2, b=0, c=1)": (2, 0, 1, "Strict Local Minimum"),
        "Strict Local Max (a=-2, b=1, c=-2)": (-2, 1, -2, "Strict Local Maximum"),
        "Classic Saddle (a=2, b=0, c=-1.5)": (2, 0, -1.5, "Saddle Point"),
        "Cross-Term Saddle Trap (a=2, b=3, c=2)": (2, 3, 2, "Saddle Point"),
        "Degenerate Valley (a=2, b=0, c=0)": (2, 0, 0, "Semidefinite / Inconclusive"),
    }
    for name, (a, b, c, expected) in presets.items():
        l1, l2, v1, v2 = check_matrix(a, b, c)
        label = classify(l1, l2)
        assert label == expected, (name, label)
        print(f"  {name:42s} l1={l1:+.3f} l2={l2:+.3f} "
              f"v1=({v1[0]:+.3f},{v1[1]:+.3f}) v2=({v2[0]:+.3f},{v2[1]:+.3f}) -> {label}")

    print("== Mode 1: 2000 seeded random (a, b, c) in [-4, 4]^3 ==")
    for _ in range(2000):
        a, b, c = RNG.uniform(-4, 4, size=3)
        check_matrix(a, b, c)
    print("  eigenpairs, FD Hessian, directional curvature, g''(0), Newton: all OK")

    print("== Prediction prompt: a = c = 2, b = 2.5, theta = 135 deg ==")
    l1, l2, v1, v2 = check_matrix(2, 2.5, 2)
    assert abs(l1 - 4.5) < 1e-12 and abs(l2 + 0.5) < 1e-12
    assert np.allclose(v2, [-1 / np.sqrt(2), 1 / np.sqrt(2)])
    u = np.array([np.cos(np.radians(135)), np.sin(np.radians(135))])
    uHu = u @ np.array([[2, 2.5], [2.5, 2]]) @ u
    assert abs(uHu + 0.5) < 1e-12
    print(f"  l1={l1}, l2={l2}, v2={v2.round(4)}, u^T H u at 135 deg = {uHu:.4f}")
    for b in [0, 1, 1.9, 2, 2.1, 3]:
        l1, l2, _, _ = eig_closed_form(2, b, 2)
        print(f"  b={b:<4} det={4 - b * b:+.2f} -> {classify(l1, l2)}")

    print("== GD stability: converges iff 0 < eta * l_max < 2 (for PD Hessians) ==")
    count = 0
    for _ in range(300):
        a, b, c = RNG.uniform(-4, 4, size=3)
        l1, l2, _, _ = eig_closed_form(a, b, c)
        if l2 <= 0.05:
            continue
        eta = RNG.uniform(0.02, 0.30)
        if abs(eta * l1 - 2) < 0.02:
            continue
        ok = gd_converges(a, b, c, eta, [1.2, 0.8])
        assert ok == (eta * l1 < 2), (a, b, c, eta, l1)
        count += 1
    print(f"  checked {count} positive-definite cases: OK")

    print("== Mode 2(a): Hf(0,0) = diag(2, 0) for x^2+y^4, x^2+y^3, x^2 ==")
    fns = {
        "x^2 + y^4": lambda p: p[0] ** 2 + p[1] ** 4,
        "x^2 + y^3": lambda p: p[0] ** 2 + p[1] ** 3,
        "x^2":       lambda p: p[0] ** 2,
    }
    for name, fn in fns.items():
        H = fd_hessian(fn, np.zeros(2), h=1e-3)
        assert np.allclose(H, [[2, 0], [0, 0]], atol=1e-5), (name, H)
        print(f"  {name:10s} Hf(0,0) ~ {H.round(6).tolist()}")

    print("== Mode 2(b): f(w) = ||y - A w||_2^2 ==")
    m, n = 30, 2
    A = np.column_stack([np.ones(m), RNG.uniform(-2, 2, size=m)])
    y = A @ np.array([0.5, 1.5]) + 0.3 * RNG.standard_normal(m)
    f = lambda w: float(np.sum((y - A @ w) ** 2))
    w = RNG.standard_normal(n)
    grad_formula = -2 * A.T @ (y - A @ w)
    h = 1e-6
    grad_fd = np.array([(f(w + h * e) - f(w - h * e)) / (2 * h) for e in np.eye(n)])
    assert np.allclose(grad_formula, grad_fd, rtol=1e-6, atol=1e-5)
    H_formula = 2 * A.T @ A
    assert np.allclose(fd_hessian(f, w, h=1e-3), H_formula, rtol=1e-6, atol=1e-4)
    assert np.all(np.linalg.eigvalsh(H_formula) >= -1e-12)
    w_star = np.linalg.solve(A.T @ A, A.T @ y)
    assert np.allclose(-2 * A.T @ (y - A @ w_star), 0, atol=1e-9)
    print(f"  gradient and Hessian match finite differences; eig(2A^TA) = {np.linalg.eigvalsh(H_formula).round(3)} >= 0")
    print(f"  normal-equation solution w* = {w_star.round(4)} has grad f(w*) = 0")

    A_def = np.column_stack([A[:, 1], A[:, 1]])  # rank-deficient: duplicated column
    ev = np.linalg.eigvalsh(2 * A_def.T @ A_def)
    assert abs(ev[0]) < 1e-9
    print(f"  rank-deficient A: eig(2A^TA) = {ev.round(6)} -> lambda_min = 0 (flat valley)")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
