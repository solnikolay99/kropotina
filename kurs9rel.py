from plotter import *

B0 = 1e-6  # Gs, magnetic field
E0 = 0.0  # Gs
# q = 4.8e-10  # CGS units, electric charge of proton/electron
# m = 1.67e-24  # g, proton mass
# c = 3.0e10  # velocity of light
q = 1  # CGS units, electric charge of proton/electron
m = 1  # g, proton mass
c = 1  # velocity of light
qm = q / m
qm2 = q / (2 * m)
qm2c = q / (2 * m * c)
c2 = c * c
B_qmc = []  # qm * B / c


def F(E_v: float, v1_v: float, v2_v: float, B1: float, B2: float):
    return E_v + v1_v * B1 - v2_v * B2


def pusher_boris_c(dt: float, r: list, u: list, B: list, E: list):
    """
    Boris-C method
    """
    return r, u


def pusher_boris_b(dt: float, r: list, u: list, B: list, E: list):
    """
    Boris-B method
    """
    u_minus = [u[i] + qm2 * E[i] * dt for i in range(3)]
    gamma = np.sqrt(1 + np.dot(u_minus, u_minus) / c2)
    t_v = [qm2c * dt * B[i] / gamma for i in range(3)]
    s = [2 * t_v[i] / (1 + np.dot(t_v, t_v)) for i in range(3)]
    u_prime = u_minus + np.cross(u_minus, t_v)
    u_plus = u_minus + np.cross(u_prime, s)
    u = [u_plus[i] + qm2 * E[i] * dt for i in range(3)]

    r = [r[i] + u[i] * dt / gamma for i in range(3)]

    return r, u


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


def main_test_1_5(dt_step: float, r1: list, r2: list, u1: list, u2: list, B: list, E: list):
    """
    Main part of tests from Table 1
    """
    graph1, graph2 = [], []  # graphs points

    for t in np.arange(0, 12 * np.pi, dt_step):
        u1_prev = u1[0]
        u2_prev = u2[0]
        r1, u1 = pusher_boris_c(dt_step, r1, u1, B, E)
        r2, u2 = pusher_boris_b(dt_step, r2, u2, B, E)
        u1_next = u1[0]
        u2_next = u2[0]
        du1 = np.abs(u1_next - u1_prev)
        du2 = np.abs(u2_next - u2_prev)
        du1u1t = [du1 / np.abs(u1[0]), t, 0]
        du2u2t = [du2 / np.abs(u2[0]), t, 0]

        graph1.append(du1u1t.copy())
        graph2.append(du2u2t.copy())

    plot_2d_graph([graph1, graph2],
                  colors=['red', 'green'],
                  names=['Метод Бориса C', 'Метод Бориса B'],
                  directions=['xy'],
                  with_markers=True)

    pass


def test_1():
    """
     Table 1, test 1: B = (0,0,0) E = (1,0,0) - direct acceleration by E
    """
    dt_step = np.pi / 6

    B = [0.0, 0.0, 0.0]
    E = [1.0, 0.0, 0.0]

    r1 = [0.0, 0.0, 0.0]  # particle positions for pusher 1
    r2 = [0.0, 0.0, 0.0]  # particle positions for pusher 2
    u1 = [1.0, 0.0, 0.0]  # particle velocities for pusher 1
    u2 = [1.0, 0.0, 0.0]  # particle velocities for pusher 2

    main_test_1_5(dt_step, r1, r2, u1, u2, B, E)

    pass


def test_2():
    """
     Table 1, test 2: B = (0,0,0.1) E = (1,0,0) - E-dominated
    """
    dt_step = np.pi / 6

    B = [0.0, 0.0, 0.1]
    E = [1.0, 0.0, 0.0]

    r1 = [0.0, 0.0, 0.0]  # particle positions for pusher 1
    r2 = [0.0, 0.0, 0.0]  # particle positions for pusher 2
    u1 = [1.0, 0.0, 0.0]  # particle velocities for pusher 1
    u2 = [1.0, 0.0, 0.0]  # particle velocities for pusher 2

    main_test_1_5(dt_step, r1, r2, u1, u2, B, E)

    pass


def test_3():
    """
     Table 1, test 3: B = (0,0,1) E = (1,0,0) - |E| = |B|
    """
    dt_step = np.pi / 6

    B = [0.0, 0.0, 1.0]
    E = [1.0, 0.0, 0.0]

    r1 = [0.0, 0.0, 0.0]  # particle positions for pusher 1
    r2 = [0.0, 0.0, 0.0]  # particle positions for pusher 2
    u1 = [1.0, 0.0, 0.0]  # particle velocities for pusher 1
    u2 = [1.0, 0.0, 0.0]  # particle velocities for pusher 2

    main_test_1_5(dt_step, r1, r2, u1, u2, B, E)

    pass


def test_4():
    """
     Table 1, test 4: B = (0,0,1) E = (0.1,0,0) - E×B drift
    """
    dt_step = np.pi / 6

    B = [0.0, 0.0, 1.0]
    E = [0.1, 0.0, 0.0]

    r1 = [0.0, 0.0, 0.0]  # particle positions for pusher 1
    r2 = [0.0, 0.0, 0.0]  # particle positions for pusher 2
    u1 = [1.0, 0.0, 0.0]  # particle velocities for pusher 1
    u2 = [1.0, 0.0, 0.0]  # particle velocities for pusher 2

    main_test_1_5(dt_step, r1, r2, u1, u2, B, E)

    pass


def test_5():
    """
     Table 1, test 5: B = (0,0,1) E = (0,0,0) - gyration about B
    """
    dt_step = np.pi / 6

    B = [0.0, 0.0, 1.0]
    E = [0.0, 0.0, 0.0]

    r1 = [0.0, 0.0, 0.0]  # particle positions for pusher 1
    r2 = [0.0, 0.0, 0.0]  # particle positions for pusher 2
    u1 = [1.0, 0.0, 0.0]  # particle velocities for pusher 1
    u2 = [1.0, 0.0, 0.0]  # particle velocities for pusher 2

    main_test_1_5(dt_step, r1, r2, u1, u2, B, E)

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
    test_2()
    test_3()
    test_4()
    test_5()
