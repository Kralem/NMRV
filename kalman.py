import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
from ex4_utils import kalman_step

def NCV(qq, rr):
    T, q, r = sp.symbols('T q r')
    F = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    Fi = sp.exp(F * T)
    #print(Fi)

    L = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])

    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    #print(Q)

    Fi = Fi.subs(T, 1)
    Q = Q.subs(T, 1)

    Q = Q.subs(q, qq)
    R = r * sp.Matrix([[1, 0], [0, 1]])
    R = R.subs(r, rr)

    H = sp.Matrix([[1,0,0,0], [0,1,0,0]])

    return np.array(Fi, dtype=float), np.array(H, dtype=float), np.array(Q, dtype=float), np.array(R, dtype=float)

def RW(qq, rr):
    T, q, r = sp.symbols('T q r')
    F = sp.zeros(2)
    Fi = sp.exp(F * T)

    L = sp.eye(2)

    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))

    Fi = Fi.subs(T, 1)
    Q = Q.subs(T, 1)

    Q = Q.subs(q, qq)

    R = r * sp.Matrix([[1, 0], [0, 1]])
    R = R.subs(r, rr)
    H = L

    return np.array(Fi, dtype=float), np.array(H, dtype=float), np.array(Q, dtype=float), np.array(R, dtype=float)

def NCA(qq, rr):
    T, q, r = sp.symbols('T q r')
    F = sp.Matrix([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]])
    Fi = sp.exp(F * T)

    L = sp.Matrix([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]])

    Q = sp.integrate((Fi * L) * q * (Fi * L).T, (T, 0, T))
    print(Q)

    Fi = Fi.subs(T, 1)
    Q = Q.subs(T, 1)

    Q = Q.subs(q, qq)

    R = r * sp.Matrix([[1, 0], [0, 1]])
    R = R.subs(r, rr)
    H = sp.Matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    return np.array(Fi, dtype=float), np.array(H, dtype=float), np.array(Q, dtype=float), np.array(R, dtype=float)

def run():
    N = 40
    v = np.linspace(5 * math.pi, 0, N)
    x = np.cos(v) * v
    y = np.sin(v) * v

    sx = np.zeros((x.size, 1), dtype=np.float32).flatten()
    sy = np.zeros((y.size, 1), dtype=np.float32).flatten()

    sx[0] = x[0]
    sy[0] = y[0]

    A, C, Q_i, R_i = NCA(100, 1)

    state = np.zeros((A.shape[0], 1), dtype=np.float32).flatten()

    state[0] = x[0]
    state[1] = y[1]
    covariance = np.eye(A.shape[0], dtype=np.float32)

    for j in range(1, x.size):
        state, covariance, _, _ = kalman_step(A, C, Q_i, R_i, np.reshape(np.array([x[j], y[j]]), (-1, 1)),
                                              np.reshape(state, (-1, 1)), covariance)
        sx[j] = state[0]
        sy[j] = state[1]

    plt.plot(x, y, 'b', label='measurements')
    plt.plot(sx, sy, 'ro', label='filtered')
    plt.title("q= 100, r = 1")
    plt.legend()
    plt.show()