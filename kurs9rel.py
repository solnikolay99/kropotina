from plotter import *

B0 = 1e-6  # Gs, magnetic field
E0 = 0.0  # Gs
v0 = 3e8  # cm/s
q = 4.8e-10  # CGS units, electric charge of proton/electron
m = 1.67e-24  # g, proton mass
qm = q / m
c = 3.0e10  # velocity of light
B_qmc = []  # qm * B / c


def F(E_v: float, v1_v: float, v2_v: float, B1: float, B2: float):
    return E_v + v1_v * B1 - v2_v * B2


def pusher_boris_c(dt: float, r: list, v: list, u: list, B: list, E: list):
    """
    Boris-B method
    """
    return r, v, u


def pusher_boris_b(dt: float, r: list, v: list, u: list, B: list, E: list):
    """
    Boris-B method
    """
    v1_l = [v[i] + qm * E[i] * dt / 2 for i in range(3)]
    a1 = [q * dt / (2 * m * c) for _ in range(3)]
    a2 = [2 * a1[i] / (1 + a1[i] * a1[i] * np.dot(B, B)) for i in range(3)]

    v3_l = np.cross(v1_l, B)
    v3_l = v1_l + a1 * v3_l

    v2_l = np.cross(v3_l, B)
    v2_l = v1_l + a2 * v2_l

    v = [v2_l[i] + qm * E[i] * dt / 2 for i in range(3)]

    r = [r[i] + v[i] * dt for i in range(3)]

    return r, v, u


def pusher_rk4(dt: float, r: list, v: list, u: list, B: list, E: list):
    """
    Runge-Kutta method RK4
    """
    a_r = [val * dt for val in v]

    a_v = np.empty(3, dtype=float)
    a_v[0] = F(E[0], v[1], v[2], B_qmc[2], B_qmc[1]) * dt
    a_v[1] = F(E[1], v[2], v[0], B_qmc[0], B_qmc[2]) * dt
    a_v[2] = F(E[2], v[0], v[1], B_qmc[1], B_qmc[0]) * dt

    b_r = [dt * (v[i] + 1 / 2 * a_v[i]) for i in range(3)]

    b_v = np.empty(3, dtype=float)
    b_v[0] = F(E[0], v[1] + 1 / 2 * a_v[1], v[2] + 1 / 2 * a_v[2], B_qmc[2], B_qmc[1]) * dt
    b_v[1] = F(E[1], v[2] + 1 / 2 * a_v[2], v[0] + 1 / 2 * a_v[0], B_qmc[0], B_qmc[2]) * dt
    b_v[2] = F(E[2], v[0] + 1 / 2 * a_v[0], v[1] + 1 / 2 * a_v[1], B_qmc[1], B_qmc[0]) * dt

    c_r = [dt * (v[i] + 1 / 2 * b_v[i]) for i in range(3)]

    c_v = np.empty(3, dtype=float)
    c_v[0] = F(E[0], v[1] + 1 / 2 * b_v[1], v[2] + 1 / 2 * b_v[2], B_qmc[2], B_qmc[1]) * dt
    c_v[1] = F(E[1], v[2] + 1 / 2 * b_v[2], v[0] + 1 / 2 * b_v[0], B_qmc[0], B_qmc[2]) * dt
    c_v[2] = F(E[2], v[0] + 1 / 2 * b_v[0], v[1] + 1 / 2 * b_v[1], B_qmc[1], B_qmc[0]) * dt

    d_r = [dt * (v[i] + c_v[i]) for i in range(3)]

    d_v = np.empty(3, dtype=float)
    d_v[0] = F(E[0], v[1] + c_v[1], v[2] + c_v[2], B_qmc[2], B_qmc[1]) * dt
    d_v[1] = F(E[1], v[2] + c_v[2], v[0] + c_v[0], B_qmc[0], B_qmc[2]) * dt
    d_v[2] = F(E[2], v[0] + c_v[0], v[1] + c_v[1], B_qmc[1], B_qmc[0]) * dt

    r = [r[i] + (a_r[i] + 2 * b_r[i] + 2 * c_r[i] + d_r[i]) / 6 for i in range(3)]
    v = [v[i] + (a_v[i] + 2 * b_v[i] + 2 * c_v[i] + d_v[i]) / 6 for i in range(3)]

    return r, v, u


def test_1():
    """
     Table 1, test 1: B = (0,0,0) E = (1,0,0) - direct acceleration by E
    """
    dt_step = np.pi / 6
    beta0 = 0.9  # velocity in units
    gamma0 = 1.0 / np.sqrt(1.0 - beta0 * beta0)

    B = [0.0, 0.0, 0.0]
    E = [1.0, 0.0, 0.0]

    r1 = [0.0, 0.0, 0.0]  # particle positions for pusher 1
    r2 = [0.0, 0.0, 0.0]  # particle positions for pusher 2
    v1 = [v0, 0.0, 0.0]  # particle velocities for pusher 1
    v2 = [v0, 0.0, 0.0]  # particle velocities for pusher 2
    u1 = [v0 * gamma0, 0.0, 0.0]  # particle velocities for pusher 1
    u2 = [v0 * gamma0, 0.0, 0.0]  # particle velocities for pusher 2

    graph1, graph2 = [], []  # graphs points

    for _ in np.arange(0, 12 / np.pi, dt_step):
        r1, v1, u1 = pusher_boris_c(dt_step, r1, v1, u1, B, E)
        r2, v2, u2 = pusher_boris_b(dt_step, r2, v2, u2, B, E)

        graph1.append(r1.copy())
        graph2.append(r2.copy())

    plot_2d_graph([graph1, graph2],
                  colors=['red', 'green', 'blue'],
                  names=['Метод Бориса C', 'Метод Бориса B'],
                  directions=['xy'],
                  with_markers=True)

    pass


def test_2():
    """
     Table 1, test 2: B = (0,0,0.1) E = (1,0,0) - E-dominated
    """
    pass


def test_3():
    """
     Table 1, test 3: B = (0,0,1) E = (1,0,0) - |E| = |B|
    """
    pass


def test_4():
    """
     Table 1, test 4: B = (0,0,1) E = (0.1,0,0) - E×B drift
    """
    pass


def test_5():
    """
     Table 1, test 5: B = (0,0,1) E = (0,0,0) - gyration about B
    """
    pass


def test_6():
    """
     Comparison of Boris-C and RK4 methods
    """
    pass


def test_7():
    """
     Speed test
    """
    pass


if __name__ == '__main__':
    test_1()
