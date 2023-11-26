# TODO: Create class Simulator or similar
from matplotlib import pyplot as plt
import numpy as np
import time

from h import generate_h


def generate_simulations(p, l_p, l_m, alpha, mo_p, mo_m, plot=False):
    drift = True if p.drift=='Yes' else False

    Upsilon = p.Delta + p.epsilon

    dt = (p.k * p.A / p.dalpha + p.lambda_plus + p.lambda_minus)**(-1)
    
    m = int(p.T/dt)
    
    # p.Alpha setup
    alpha = np.full((p.n, m), np.nan)
    alpha[:, 0] = 0
    alpha_range = np.arange(-p.A, p.A + 1, p.dalpha)

    tau_plus_amounts = np.random.poisson(p.lambda_plus*p.T, p.n)
    tau_minus_amounts = np.random.poisson(p.lambda_minus*p.T, p.n)
    tau_plus = [np.sort(np.random.rand(tau_i) * p.T) for tau_i in tau_plus_amounts]
    tau_minus = [np.sort(np.random.rand(tau_i) * p.T) for tau_i in tau_minus_amounts]

    dMt0_plus = np.array([np.histogram(tau_i,np.linspace(0,p.T,m+1))[0] for tau_i in tau_plus])
    dMt0_minus = np.array([np.histogram(tau_i,np.linspace(0,p.T,m+1))[0] for tau_i in tau_minus])

    # S setup
    s = np.full((p.n, m), np.nan)
    s[:, 0] = p.s0

    mu_plus = np.full((p.n, m), np.nan)
    mu_plus[:, 0] = p.theta
    mu_minus = np.full((p.n, m), np.nan)
    mu_minus[:, 0] = p.theta

    dJ_plus = np.full((p.n, m), np.nan)
    dJ_plus[:, 0] = 0

    dJ_minus = np.full((p.n, m), np.nan)
    dJ_minus[:, 0] = 0

    # Positions setup
    l_p_position = np.full((p.n, m), np.nan)
    l_m_position = np.full((p.n, m), np.nan)

    p_postings = np.full((p.n, m), np.nan)
    m_postings = np.full((p.n, m), np.nan)

    p_executions = np.full((p.n, m), np.nan)
    m_executions = np.full((p.n, m), np.nan)

    p_executions_count = np.full((p.n, m), np.nan)
    m_executions_count = np.full((p.n, m), np.nan)

    mo_p_executions = np.full((p.n, m), np.nan)
    mo_m_executions = np.full((p.n, m), np.nan)

    dMt_plus = np.full((p.n, m), np.nan) # np.zeros((n, m))
    dMt_minus = np.full((p.n, m), np.nan) # np.zeros((n, m))

    pnl = np.full((p.n, m), np.nan)
    pnl[:, 0] = 0

    X = np.full((p.n, m), np.nan)
    X[:, 0] = 0

    def get_closest_index(val):
        return int(np.round(min(max(-p.A,val),p.A) / p.dalpha, 0)) + int(p.A / p.dalpha)

    def get_l_p(t_i, alpha_val, q):
        alpha_i = get_closest_index(alpha_val)
        q_i = int(q + p.q_max)
        return l_p[t_i, alpha_i, q_i]
    get_l_p_v = np.vectorize(get_l_p)

    def get_l_m(t_i, alpha_val, q):
        alpha_i = get_closest_index(alpha_val)
        q_i = int(q + p.q_max)
        return l_m[t_i, alpha_i, q_i]
    get_l_m_v = np.vectorize(get_l_m)

    def get_MM_MO_p(t_i, alpha_val, q):
        alpha_i = get_closest_index(alpha_val)
        q_i = int(q + p.q_max)
        return mo_p[t_i, alpha_i, q_i]
    get_MM_MO_p_v = np.vectorize(get_MM_MO_p)
    
    def get_MM_MO_m(t_i, alpha_val, q):
        alpha_i = get_closest_index(alpha_val)
        q_i = int(q + p.q_max)
        return mo_m[t_i, alpha_i, q_i]
    get_MM_MO_m_v = np.vectorize(get_MM_MO_m)

    # Inventory setup
    q = np.full((p.n, m), np.nan)
    q[:, 0] = 0

    # Simulations
    print(f"Amount of simulations: {p.n}")
    time_2 = time.time()
    for i in range(m-1):
        time_1 = time.time()
        print(f"%{float(i)/float(m)*100.} - {(m-i) * (time_1 - time_2)/3600} h                          ", end="\r")
        time_2 = time.time()
        # Set market order and limit order strategy
        # consider alpha for positining or not
        if drift:  
            alpha_i = alpha[:, i]
        else:
            alpha_i = np.zeros([1, len(alpha[:, i])])
        # dMt_minus and dMt_plus depend on the MM
        dMt_plus[:, i] = get_MM_MO_p_v(i, alpha_i, q[:, i])
        dMt_minus[:, i] = get_MM_MO_m_v(i, alpha_i, q[:, i])
        # limit order positions 
        l_p_position[:, i] = get_l_p_v(i, alpha_i, q[:, i])
        l_m_position[:, i] = get_l_m_v(i, alpha_i, q[:, i])

        alpha[:, i+1] = alpha[:,i] * np.exp(-p.k * dt) + p.xi * np.sqrt(dt) * (np.random.randn(p.n)) + p.eta_plus *(dMt0_plus[:,i] + dMt_plus[:, i]) - p.eta_minus * (dMt0_minus[:,i] + dMt_minus[:, i])

        mu_plus[:, i+1] = np.where(alpha[:, i+1]>0, alpha[:, i+1],0) + p.theta
        mu_minus[:, i+1] = np.where(alpha[:, i+1]<0, -alpha[:, i+1],0) + p.theta

        dJ_plus[:, i+1] = np.where(np.random.rand(p.n) < np.around((1 - np.exp(-dt * (mu_plus[:,i+1]))), decimals=4),1,0)
        dJ_minus[:, i+1] = np.where(np.random.rand(p.n) < np.around((1 - np.exp(-dt * (mu_minus[:,i+1]))), decimals=4),1,0)
        
        s[:,i+1] = s[:,i] + p.sigma * (dJ_plus[:, i+1] - dJ_minus[:, i+1])

        q[:, i+1] = q[:, i] - np.where(l_p_position[:, i] * dMt0_plus[:, i] > 0,1,0) + np.where((l_m_position[:, i] * dMt0_minus[:, i]) > 0,1,0) - np.where(dMt_minus[:, i] > 0,1,0) + np.where(dMt_plus[:,i] > 0,1,0)

        p_postings[:, i] = np.where(l_p_position[:,i]==0, np.nan, (s[:,i]+p.Delta)*l_p_position[:,i])
        p_executions_count[:,i] = np.where(l_p_position[:,i]*dMt0_plus[:,i]==0, 0, 1)
        p_executions[:, i] = np.where(l_p_position[:,i]*dMt0_plus[:,i]==0, np.nan, (s[:,i]+p.Delta)*l_p_position[:,i]*np.where(dMt0_plus[:,i]>0,1,0))
        
        m_postings[:,i] = np.where(l_m_position[:,i]==0, np.nan, (s[:,i]-p.Delta)*l_m_position[:,i])
        m_executions_count[:,i] = np.where(l_m_position[:,i]*dMt0_minus[:,i]==0, 0, 1)
        m_executions[:,i] = np.where(l_m_position[:,i]*dMt0_minus[:,i]==0, np.nan, (s[:,i]-p.Delta)*l_m_position[:,i]*np.where(dMt0_minus[:,i]>0,1,0))

        mo_p_executions[:,i] = np.where(dMt_plus[:, i]==0, np.nan, (s[:,i]+Upsilon)*dMt_plus[:, i])
        mo_m_executions[:,i] = np.where(dMt_minus[:, i]==0, np.nan, (s[:,i]-Upsilon)*dMt_minus[:, i])

        pnl[:,i+1] = pnl[:,i] + np.where(p_executions[:,i] > 0, p.Delta, 0) + np.where(m_executions[:,i] > 0, p.Delta, 0)\
            + q[:, i] * (s[:, i+1] - s[:, i]) \
            - np.where(mo_p_executions[:,i+1] > 0, Upsilon, 0) \
            - np.where(mo_m_executions[:,i+1] > 0, Upsilon, 0)
        
    X[:,-1] = X[:,-1] - q[:, -1] * (s[:, -1]) - np.abs(q[:,-1])*Upsilon
    # pnl[:,-1] = pnl[:,-1] - Upsilon * np.abs(q[:,-1])

    print(f"Mean of PNL:{np.average(pnl[:,-1])}")
    print(f"Stde of PNL:{np.std(pnl[:,-1])}")

    if plot:
        plt_i = 1
        plt.figure(figsize=(25/2,7/2))
        # plt.title('Alpha')
        plt.xlabel('t')
        plt.ylabel('alfa')
        plt.step(np.linspace(0,p.T,m),alpha[plt_i])
        # plt.savefig("../Propuesta/figuras/alpha_final",dpi=150,bbox_inches="tight",pad_inches=0.1)

        plt.figure(figsize=(25/2,7/2))

        # plt.title('S')
        plt.xlabel('t')
        plt.ylabel('S')
        plt.step(np.linspace(0,p.T,m), s[plt_i], c='black')
        
        plt.step(np.linspace(0,p.T,m), p_postings[plt_i], c='b')
        plt.scatter(np.linspace(0,p.T,m), p_executions[plt_i], marker='x', c='b')

        plt.step(np.linspace(0,p.T,m), m_postings[plt_i], c='r')
        plt.scatter(np.linspace(0,p.T,m), m_executions[plt_i], marker='x', c='r')

        plt.scatter(np.linspace(0,p.T,m), mo_m_executions[plt_i], marker='s', c='b')
        plt.scatter(np.linspace(0,p.T,m), mo_p_executions[plt_i], marker='s', c='r')
        
        # plt.savefig("../Propuesta/figuras/orders_final",dpi=150,bbox_inches="tight",pad_inches=0.1)

        print(f"MO_p: {np.nansum(dMt_plus[plt_i])}")
        print(f"MO_m: {np.nansum(dMt_minus[plt_i])}")
        print(f"LO_p: {np.nansum(m_executions_count[plt_i])}")
        print(f"LO_m: {np.nansum(p_executions_count[plt_i])}")

        plt.figure(figsize=(5,5))
        # plt.title('Limit Orders Minus Executions')
        plt.xlabel('Ordenes ejecutadas')
        plt.ylabel('Conteo')
        plt.hist(m_executions_count[:,:-1].sum(axis=1))
        # plt.savefig("../Propuesta/figuras/limit_orders_minus_executions_final",dpi=150,bbox_inches="tight",pad_inches=0.1)
        
        plt.figure(figsize=(5,5))
        # plt.title('Limit Orders Plus Executions')
        plt.xlabel('Ordenes ejecutadas')
        plt.ylabel('Conteo')
        plt.hist(p_executions_count[:,:-1].sum(axis=1))
        # plt.savefig("../Propuesta/figuras/limit_orders_plus_executions_final",dpi=150,bbox_inches="tight",pad_inches=0.1)

        plt.figure(figsize=(5,5))
        # plt.title('Market Orders Minus Executions')
        plt.xlabel('Ordenes ejecutadas')
        plt.ylabel('Conteo')
        plt.hist(dMt_minus[:, :-1].sum(axis=1))
        # plt.savefig("../Propuesta/figuras/market_orders_minus_executions_final",dpi=150,bbox_inches="tight",pad_inches=0.1)
        
        plt.figure(figsize=(5,5))
        # plt.title('Market Orders Plus Executions')
        plt.xlabel('Ordenes ejecutadas')
        plt.ylabel('Conteo')
        plt.hist(dMt_plus[:, :-1].sum(axis=1))
        # plt.savefig("../Propuesta/figuras/market_orders_plus_executions_final",dpi=150,bbox_inches="tight",pad_inches=0.1)

        if False:
            plt.figure()
            plt.title('$\mu_+$')
            plt.step(np.linspace(0,p.T,m),mu_plus[plt_i])

            plt.figure()
            plt.title('$\mu_-$')
            plt.step(np.linspace(0,p.T,m),mu_minus[plt_i])
        
        plt.figure(figsize=(25/2,7/2))
        # plt.title('$q$')
        plt.xlabel('t')
        plt.ylabel('q')
        plt.step(np.linspace(0,p.T,m),q[plt_i])
        # plt.savefig("../Propuesta/figuras/q_final",dpi=150,bbox_inches="tight",pad_inches=0.1)

        plt.figure(figsize=(25/2,7/2))
        # plt.title('$pnl$')
        plt.xlabel('t')
        plt.ylabel('PnL')
        plt.step(np.linspace(0,p.T,m),pnl[plt_i])
        # plt.savefig("../Propuesta/figuras/pnl_final",dpi=150,bbox_inches="tight",pad_inches=0.1)

    return {
        "pnl": pnl,
        "p_postings": p_postings,
        "m_executions_count": m_executions_count,
        "p_executions_count": p_executions_count,
        "dMt_minus": dMt_minus,
        "dMt_plus": dMt_plus,
        "dMt0_minus": dMt0_minus,
        "dMt0_plus": dMt0_plus,
        "s": s,
    }

def generate_simulations_bunchs(p, plot=False):
    print("Calculating h")
    h, l_p, l_m, alpha, mo_p, mo_m = generate_h(p)
    del h
    print("Finished calculating h")

    np.random.seed(2)
    b = 500
    n_bunchs = p.n // b if p.n > b else 1
    p.n = b if p.n > b else p.n

    bunch = []
    for i in range(n_bunchs):
        data = generate_simulations(p, l_p, l_m, alpha, mo_p, mo_m, plot)
        bunch.append(data)

    p.n = n_bunchs * b
    print(p.n)
    return {
        "pnl": np.concatenate([d["pnl"] for d in bunch]),
        "p_postings": np.concatenate([d["p_postings"] for d in bunch]),
        "m_executions_count": np.concatenate([d["m_executions_count"] for d in bunch]),
        "p_executions_count": np.concatenate([d["p_executions_count"] for d in bunch]),
        "dMt_minus": np.concatenate([d["dMt_minus"] for d in bunch]),
        "dMt_plus": np.concatenate([d["dMt_plus"] for d in bunch]),
        "s": np.concatenate([d["s"] for d in bunch]),
    }