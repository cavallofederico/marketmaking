import numpy as np

def generate_h(p):
    print("Starting h calculation")
    Upsilon = p.Delta + p.epsilon

    dt = p.dt
    print(dt)
    q_a = np.arange(-p.q_max, p.q_max + 1, 1)
    alpha = np.arange(-p.A, p.A + 1, p.dalpha)

    alpha_smaller_0 = np.where(alpha < 0)[0]
    alpha_greater_0 = np.where(alpha > 0)[0]
    alpha_0 = np.where(alpha == 0)[0]   

    n_q = len(q_a)
    n_alpha = len(alpha)
    n_t = int(p.T / dt)
    print(n_t)

    h = np.full((n_t, n_alpha, n_q), np.nan)
    # print(h)
    d_alpha_h = np.zeros(n_alpha)
    dd_alpha_h = np.zeros(n_alpha)

    l_plus = np.zeros((n_t, n_alpha, n_q))
    l_minus = np.zeros((n_t, n_alpha, n_q))

    h_eta_up = np.full((n_t, n_alpha, n_q), np.nan)
    h_eta_down = np.full((n_t, n_alpha, n_q), np.nan)

    # print(h)

    # def calculate_h():
    print("Starting to calculate h")
    import time
    # Terminal and boundary conditions
    h[-1, :, :] = (
        np.ones((1, n_alpha)) *
        np.array([(q_a * (-np.sign(q_a) * Upsilon - p.psi * q_a))]).T
    ).T

    time_2 = time.time()
    for t_i in range(n_t - 2, -1, -1):
        time_1 = time.time()
        print(f"{n_t-t_i}/{n_t} - {(n_t-t_i)/n_t} - {t_i * (time_1 - time_2)/3600} h                          ", end='\r')
        time_2 = time.time()
        for q_i in range(n_q):
            h_q_t_1 = h[t_i + 1, :, q_i]
            d_alpha_h = calculate_d_alpha_h(p, d_alpha_h, alpha_smaller_0, alpha_greater_0, alpha_0, h_q_t_1)
            dd_alpha_h = calculate_dd_alpha_h(p, dd_alpha_h, h_q_t_1)
            l_plus[t_i + 1, :, q_i], l_minus[t_i + 1, :, q_i] = find_optimal_postings(
                p, h_eta_up, h_eta_down, n_alpha, q_a, h, t_i, q_i
            )
            h[t_i, :, q_i] = S_dt_dalpha(Upsilon, h_eta_up, h_eta_down, p, q_a, alpha, dt, h, t_i, q_i, d_alpha_h, dd_alpha_h)

    def find_optimal_MO(h, t_i, q_i):
        if q_a[q_i] > -(p.q_max - 1):
            # h_eta_down_q_m_1 = interpolate(p, h[t_i + 1, :, q_i - 1], up=False)
            # mo_minus_i = np.where(- Upsilon + h_eta_down_q_m_1 > h[t_i + 1, :, q_i], 1, 0)
            mo_minus_i = np.where((h[t_i + 1, :, q_i - 1] - Upsilon) > h[t_i + 1, :, q_i], 1, 0)
        else:
            mo_minus_i = np.zeros(n_alpha)

        if q_a[q_i] < (p.q_max - 1):
            # h_eta_up_q_p_1 = interpolate(p, h[t_i + 1, :, q_i + 1])
            # mo_plus_i = np.where(- Upsilon + h_eta_up_q_p_1 > h[t_i + 1, :, q_i], 1, 0)
            mo_plus_i = np.where((h[t_i + 1, :, q_i + 1] - Upsilon) > h[t_i + 1, :, q_i],1,0)
        else:
            mo_plus_i = np.zeros(n_alpha)

        return mo_plus_i, mo_minus_i

    def get_mo():
        mo_plus = np.zeros((n_t, n_alpha, n_q))
        mo_minus = np.zeros((n_t, n_alpha, n_q))

        for t_i in range(n_t - 2, -1, -1):
            for q_i in range(n_q):
                mo_plus[t_i + 1, :, q_i], mo_minus[t_i + 1, :, q_i] = find_optimal_MO(
                    h, t_i, q_i
                )
        return mo_plus, mo_minus
    

    # calculate_h()
    mo_plus, mo_minus = get_mo()
    return h, l_plus, l_minus, alpha, mo_plus, mo_minus




def S_dt_dalpha(Upsilon, h_eta_up, h_eta_down, p, q_a, alpha, dt, h, t_i, q_i, d_alpha_h, dd_alpha_h):
    T_dt_dalpha_i = T_dt_dalpha(h_eta_up, h_eta_down, p, q_a, alpha, dt, h, t_i, q_i, d_alpha_h, dd_alpha_h)
    M_dt_dalpha_i = M_dt_dalpha(p, Upsilon, q_a, h, t_i, q_i)
    return np.maximum(T_dt_dalpha_i, M_dt_dalpha_i)


def M_dt_dalpha(p, Upsilon, q_a, h, t_i, q_i):
    if q_a[q_i] < p.q_max and q_a[q_i] > -p.q_max:
        return np.maximum(
            (h[t_i + 1, :, q_i - 1] - Upsilon), (h[t_i + 1, :, q_i + 1] - Upsilon)
        )
    elif q_a[q_i] > -p.q_max:
        return h[t_i + 1, :, q_i - 1] - Upsilon
    elif q_a[q_i] < p.q_max:
        return h[t_i + 1, :, q_i + 1] - Upsilon
    else:
        raise ValueError(f"Imposible Case {q_a[q_i]}")

def get_l_minus_term(p, q_a, h_eta_down, t_i, q_i, h_t_1_q):
    if q_a[q_i] < p.q_max:
        l_minus_term = p.lambda_minus * np.maximum(
            (p.Delta + h_eta_down[t_i + 1, :, q_i + 1] - h_t_1_q),
            (h_eta_down[t_i + 1, :, q_i] - h_t_1_q),
        )
    else:
        l_minus_term = h_eta_down[t_i + 1, :, q_i] - h_t_1_q
    return l_minus_term


def get_l_plus_term(p, q_a, h_eta_up, t_i, q_i, h_t_1_q):
    if q_a[q_i] > -p.q_max:
        l_plus_term = p.lambda_plus * np.maximum(
            (p.Delta + h_eta_up[t_i + 1, :, q_i - 1] - h_t_1_q),
            (h_eta_up[t_i + 1, :, q_i] - h_t_1_q),
        )
    else:
        l_plus_term = h_eta_up[t_i + 1, :, q_i] - h_t_1_q
    return l_plus_term


def T_dt_dalpha(h_eta_up, h_eta_down, p, q_a, alpha, dt, h, t_i, q_i, d_alpha_h, dd_alpha_h):
    h_t_1_q = h[t_i + 1, :, q_i]
    q_ = q_a[q_i]

    l_plus_term = get_l_plus_term(p, q_a, h_eta_up, t_i, q_i, h_t_1_q)

    l_minus_term = get_l_minus_term(p, q_a, h_eta_down, t_i, q_i, h_t_1_q)

    h_t_q = h_t_1_q + dt * (
        alpha * p.sigma * q_
        - p.k * alpha * d_alpha_h
        + ((p.xi**2) / 2) * dd_alpha_h
        - p.phi_ * q_**2
        + l_plus_term
        + l_minus_term
    )

    # impose second derivative vanishes along maximum and minimum values of alpha grid
    h_t_q[0] = 2 * h_t_q[1] - h_t_q[2]
    h_t_q[-1] = 2 * h_t_q[-2] - h_t_q[-3]
    return h_t_q


def calculate_d_alpha_h(p, d_alpha_h, alpha_smaller_0, alpha_greater_0, alpha_0, h_q_t):
    d_alpha_h[alpha_smaller_0] = (
        h_q_t[alpha_smaller_0 + 1] - h_q_t[alpha_smaller_0]
    ) / p.dalpha
    d_alpha_h[alpha_greater_0] = (
        h_q_t[alpha_greater_0] - h_q_t[alpha_greater_0 - 1]
    ) / p.dalpha
    d_alpha_h[alpha_0] = (
        (h_q_t[alpha_0 + 1] - h_q_t[alpha_0]) +
        (h_q_t[alpha_0] - h_q_t[alpha_0 - 1])
    ) / (2 * p.dalpha)
    return d_alpha_h


def calculate_dd_alpha_h(p, dd_alpha_h, h_q_t):
    dd_alpha_h[1:-1] = (h_q_t[2:] - 2 * h_q_t[1:-1] - h_q_t[:-2]) / (p.dalpha**2)
    return dd_alpha_h


def find_optimal_postings(p, h_eta_up, h_eta_down, n_alpha, q_a, h, t_i, q_i):
    h_eta_up[t_i + 1, :, q_i] = interpolate(p, h[t_i + 1, :, q_i])
    if q_a[q_i] > -p.q_max:
        h_eta_up[t_i + 1, :, q_i - 1] = interpolate(p, h[t_i + 1, :, q_i - 1])
        l_plus_i = np.where(
            p.Delta + h_eta_up[t_i + 1, :, q_i -
                            1] > h_eta_up[t_i + 1, :, q_i], 1, 0
        )
    else:
        l_plus_i = np.zeros(n_alpha)

    h_eta_down[t_i + 1, :, q_i] = interpolate(p, h[t_i + 1, :, q_i], up=False)
    if q_a[q_i] < p.q_max:
        h_eta_down[t_i + 1, :, q_i +
                1] = interpolate(p, h[t_i + 1, :, q_i + 1], up=False)
        l_minus_i = np.where(
            p.Delta + h_eta_down[t_i + 1, :, q_i +
                            1] > h_eta_down[t_i + 1, :, q_i], 1, 0
        )
    else:
        l_minus_i = np.zeros(n_alpha)
    return l_plus_i, l_minus_i



def interpolate(p, phi, up=True):
    if up:
        eta = p.eta_plus
    else:
        eta = p.eta_minus
    eta_dalpha = eta / p.dalpha
    eta_dalpha_floor = np.floor(eta_dalpha)
    eta_dalpha_diff = eta_dalpha - eta_dalpha_floor
    eta_move = int(eta_dalpha_floor)

    phi_eta = phi if up else np.flip(phi)

    phi_eta = np.roll(phi_eta, -eta_move)
    phi_eta[-eta_move:] = np.nan

    phi_eta_1 = np.roll(phi_eta, -1)
    phi_eta_1[-1:] = np.nan

    phi_eta += (phi_eta_1 - phi_eta) * eta_dalpha_diff
    phi_eta[-eta_move - 1:] = extrapolate_up(
        phi if up else np.flip(phi), len(
            phi_eta[-eta_move - 1:]), eta_dalpha_diff
    )

    phi_eta = phi_eta if up else np.flip(phi_eta)

    return phi_eta

def extrapolate_up(phi, n, diff):
    delta_phi = phi[-1] - phi[-2]
    phi_extrapolated = (
        np.ones(n) * phi[-1] + diff * delta_phi + np.arange(0, n) * delta_phi
    )
    return phi_extrapolated