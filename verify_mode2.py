"""Independent numerical and analytical checks for CSE 840 Concept Demo Mode 2.

Run: python verify_mode2.py
Requires: numpy
This validates the mathematical model, not browser rendering or JavaScript event handlers.
"""
import numpy as np

RNG = np.random.default_rng(840)


def finite_hessian(f, p, h=1e-3):
    p = np.asarray(p, dtype=float)
    n = len(p)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.eye(n)[i] * h
            ej = np.eye(n)[j] * h
            out[i, j] = (f(p+ei+ej)-f(p+ei-ej)-f(p-ei+ej)+f(p-ei-ej))/(4*h*h)
    return out


def boundary_checks():
    print('MODE 2(a): SEMIDEFINITE BOUNDARY')
    cases = {
        'x^2 + y^4': (lambda p: p[0]**2 + p[1]**4, 'strict local minimum'),
        'x^2 + y^3': (lambda p: p[0]**2 + p[1]**3, 'saddle point'),
        'x^2': (lambda p: p[0]**2, 'non-strict local minimum'),
    }
    H_expected = np.diag([2., 0.])
    for name, (f, label) in cases.items():
        H = finite_hessian(f, [0., 0.])
        assert np.allclose(H, H_expected, atol=1e-5), (name, H)
        assert np.allclose(np.linalg.eigvalsh(H), [0., 2.], atol=1e-5)
        assert abs(np.linalg.det(H)) < 2e-5
        assert np.isclose(f(np.array([0., 0.])), 0)
        if 'y^4' in name:
            for _ in range(100):
                p = RNG.uniform(-1, 1, 2)
                assert f(p) > 0 or np.allclose(p, 0)
        elif 'y^3' in name:
            assert f(np.array([0., 0.5])) > 0
            assert f(np.array([0., -0.5])) < 0
        else:
            assert f(np.array([0., 0.5])) == f(np.array([0., 0.]))
            assert f(np.array([0.5, 0.])) > f(np.array([0., 0.]))
        print(f'  PASS {name}: H(0,0)=diag(2,0); eigenvalues (0,2); {label}')


def least_squares_checks():
    print('MODE 2(b): LEAST SQUARES')
    # Exact hand-checkable examples
    for A, expected_H, expected_eigs in [
        (np.diag([1., 2.]), np.diag([2., 8.]), [2., 8.]),
        (np.array([[1., 1.], [2., 2.]]), np.array([[10., 10.], [10., 10.]]), [0., 20.]),
    ]:
        H = 2 * A.T @ A
        assert np.allclose(H, expected_H)
        assert np.allclose(np.linalg.eigvalsh(H), expected_eigs)
    print('  PASS hand-checkable full-rank and duplicate-column Hessians')

    m = 24
    x = RNG.uniform(-2, 2, m)
    y = 0.5 + 1.5*x + 0.3*RNG.normal(size=m)
    for s in [0., 0.25, 0.5, 0.75, 0.95, 1.]:
        # Same column interpolation used in the HTML: A(s)=[1,(1-s)x+s]
        A = np.column_stack((np.ones(m), (1-s)*x+s))
        H = 2*A.T@A
        f = lambda w: float(np.sum((y-A@w)**2))
        w = np.array([0.4, 1.2])
        analytic_grad = -2*A.T@(y-A@w)
        h = 1e-6
        numeric_grad = np.array([(f(w+h*e)-f(w-h*e))/(2*h) for e in np.eye(2)])
        assert np.allclose(analytic_grad, numeric_grad, atol=1e-5)
        assert np.allclose(finite_hessian(f, w), H, atol=1e-4)
        eigs = np.linalg.eigvalsh(H)
        assert eigs[0] >= -1e-9
        if s == 1.:
            assert np.linalg.matrix_rank(A) == 1
            assert np.isclose(eigs[0], 0, atol=1e-9)
            mean = y.mean()
            w_star = np.array([mean/2, mean/2])
            assert np.isclose(f(w_star), f(w_star + np.array([3., -3.])))
            assert np.allclose(-2*A.T@(y-A@w_star), 0, atol=1e-8)
        else:
            assert np.linalg.matrix_rank(A) == 2
            assert eigs[0] > 0
            # Compare centered-regression implementation with numpy's least-squares solver
            v = A[:, 1]
            slope = np.sum((v-v.mean())*(y-y.mean()))/np.sum((v-v.mean())**2)
            w_star = np.array([y.mean()-slope*v.mean(), slope])
            w_numpy = np.linalg.lstsq(A, y, rcond=None)[0]
            assert np.allclose(w_star, w_numpy, atol=1e-8)
            assert np.allclose(-2*A.T@(y-A@w_star), 0, atol=1e-7)
        print(f'  PASS similarity {s:.0%}: eigenvalues {eigs.round(5)}, loss {f(w_star):.5f}')


if __name__ == '__main__':
    boundary_checks()
    least_squares_checks()
    print('\nALL MODE 2 MATHEMATICAL CHECKS PASSED')
