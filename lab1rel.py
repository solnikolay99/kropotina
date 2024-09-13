from plotter import *

B0 = 1e-3  # Gs, magnetic field
E0 = 0.9e-3  # Gs
beta0 = 0.9  # velocity in units of c
q = 4.8e-10  # CGS units, electric charge of proton/electron
m = 1.67e-24  # g, proton mass
c = 3.0e10  # velocity of light
qm = q / m
qm2 = q / (2 * m)
qm2c = q / (2 * m * c)
c2 = c * c
dt_step = 1.0  # seconds, time step
B = []  # magnetic field
E = []  # electric field
B_qmc = []  # qm * B / c


def F(E_v: float, v1_v: float, v2_v: float, B1: float, B2: float):
    return E_v + v1_v * B1 - v2_v * B2


def pusher1(dt: float, r: list, v: list):
    """
    Explicit Euler method
    """
    f_x = F(E[0], v[1], v[2], B_qmc[2], B_qmc[1])
    f_y = F(E[1], v[2], v[0], B_qmc[0], B_qmc[2])
    f_z = F(E[2], v[0], v[1], B_qmc[1], B_qmc[0])

    r = [r[i] + v[i] * dt for i in range(3)]

    v[0] = v[0] + f_x * dt
    v[1] = v[1] + f_y * dt
    v[2] = v[2] + f_z * dt

    return r, v


def pusher2(dt: float, r: list, v: list):
    """
    Runge-Kutta method RK2
    """
    a_r = [val * dt for val in v]

    a_v = np.empty(3, dtype=float)
    a_v[0] = F(E[0], v[1], v[2], B_qmc[2], B_qmc[1]) * dt
    a_v[1] = F(E[1], v[2], v[0], B_qmc[0], B_qmc[2]) * dt
    a_v[2] = F(E[2], v[0], v[1], B_qmc[1], B_qmc[0]) * dt

    b_r = [dt * (v[i] + a_v[i]) for i in range(3)]

    b_v = np.empty(3, dtype=float)
    b_v[0] = F(E[0], v[1] + a_v[1], v[2] + a_v[2], B_qmc[2], B_qmc[1]) * dt
    b_v[1] = F(E[1], v[2] + a_v[2], v[0] + a_v[0], B_qmc[0], B_qmc[2]) * dt
    b_v[2] = F(E[2], v[0] + a_v[0], v[1] + a_v[1], B_qmc[1], B_qmc[0]) * dt

    r = [r[i] + (a_r[i] + b_r[i]) / 2 for i in range(3)]
    v = [v[i] + (a_v[i] + b_v[i]) / 2 for i in range(3)]

    return r, v


def pusher3(dt: float, r: list, v: list):
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

    return r, v


def pusher5_old(dt: float, r: list, v: list):
    """
    Boris method
    """
    v1_l = [v[i] + qm * E[i] * dt / 2 for i in range(3)]
    gamma = np.sqrt(1 + np.dot(v1_l, v1_l) / c2)
    a1 = [q * dt / (2 * m * c * gamma) for _ in range(3)]
    a2 = [2 * a1[i] / (1 + a1[i] * a1[i] * np.dot(B, B)) for i in range(3)]

    v3_l = np.cross(v1_l, B)
    v3_l = v1_l + a1 * v3_l

    v2_l = np.cross(v3_l, B)
    v2_l = v1_l + a2 * v2_l

    v = [v2_l[i] + qm * E[i] * dt / 2 for i in range(3)]

    r = [r[i] + v[i] * dt / gamma for i in range(3)]

    return r, v


def pusher5(dt: float, r: list, u: list):
    """
    Boris method
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


if __name__ == '__main__':
    t = 0  # current time
    max_t = 1000

    gamma0 = 1.0 / np.sqrt(1.0 - beta0 * beta0)
    v0 = beta0 * c

    B = [0.0, 0.0, B0]
    E = [0.0, E0, 0.0]

    # E = [qm * e for e in E]
    B_qmc = [qm * b / c for b in B]

    # r1 = [0.0, 0.0, 0.0]  # particle positions for pusher 1
    # r2 = [0.0, 0.0, 0.0]  # particle positions for pusher 2
    r3 = [0.0, 0.0, 0.0]  # particle positions for pusher 3
    r5 = [0.0, 0.0, 0.0]  # particle positions for pusher 5
    # v1 = [v0, 0.0, 0.0]  # particle velocities for pusher 1
    # v2 = [v0, 0.0, 0.0]  # particle velocities for pusher 2
    v3 = [v0, 0.0, 0.0]  # particle velocities for pusher 3
    v5 = [v0, 0.0, 0.0]  # particle velocities for pusher 5
    # u1 = [v0 * gamma0, 0.0, 0.0]  # particle 4-velocities for the pusher 1
    # u2 = [v0 * gamma0, 0.0, 0.0]  # particle 4-velocities for the pusher 2
    u3 = [v0 * gamma0, 0.0, 0.0]  # particle 4-velocities for the pusher 3
    u5 = [v0 * gamma0, 0.0, 0.0]  # particle 4-velocities for the pusher 5
    graph1, graph2, graph3, graph5 = [], [], [], []  # graphs points

    fileName = "Compare_particle_movers_dt_{}.dat".format(dt_step)

    out_tmpl = lambda r_l, v_l: "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} " \
        .format(r_l[0], r_l[1], r_l[2], v_l[0], v_l[1], v_l[2])
    with open(fileName, 'w') as output:
        for t in np.arange(0, max_t, dt_step):
            output.write("{:.4f} ".format(t))
            # output.write(out_tmpl(r1, v1))
            # output.write(out_tmpl(r2, v3))
            output.write(out_tmpl(r3, v3))
            output.write(out_tmpl(r5, u5))
            output.write("\n")
            # r1, v1 = pusher1(dt_step, r1, v1)
            # r2, v2 = pusher2(dt_step, r2, v2)
            # r3, v3 = pusher3(dt_step, r3, v3)
            r5, u5 = pusher5(dt_step, r5, u5)

            # graph1.append(r1.copy())
            # graph2.append(r2.copy())
            # graph3.append(r3.copy())
            graph5.append(r5.copy())

    plot_2d_graph([graph1, graph2, graph3, graph5],
                  colors=['blue', 'red', 'green', 'black'],
                  names=['Явный метод Эйлера', 'Метод Рунге-Кутты 2-ого порядка', 'Метод Рунге-Кутты 4-ого порядка',
                         'Метод Бориса'],
                  directions=['xy'],
                  with_markers=True)
