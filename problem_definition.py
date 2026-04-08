"""
MPC benchmark problem definitions.

Each problem returns a dict with all parameters needed by the solver wrappers.
To benchmark a different problem, add a new function and call get_problem() with
its name, or directly modify the parameters in the existing function.
"""

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.signal import cont2discrete


def get_problem(name: str = "inverted_pendulum", overrides: dict | None = None) -> dict:
    """Return benchmark problem dict by name.

    Supported names:
        'inverted_pendulum'  – cart-pole from lmpc_invpend.py (default)
        'double_integrator'  – simple 2-state double integrator example
    """
    problems = {
        "inverted_pendulum": _inverted_pendulum,
        "double_integrator": _double_integrator,
    }
    if name not in problems:
        raise ValueError(f"Unknown problem '{name}'. Available: {list(problems.keys())}")
    prob = problems[name]()
    if overrides:
        prob = _apply_overrides(prob, overrides)
    return prob


# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

def _inverted_pendulum() -> dict:
    """
    Cart-pole (inverted pendulum on a cart) linear MPC.

    State: x = [cart_pos, cart_vel, pole_angle, pole_ang_vel]
    Input: u = [force on cart]
    Output: y = [cart_pos, pole_angle]

    Mirrors the setup in lmpc_invpend.py with a time-varying reference and
    tighter input bounds to ensure the input constraint activates during transients.
    """
    # --- Continuous-time system ---
    A = np.array(
        [
            [0,   1,    0,    0],
            [0, -10,  9.81,   0],
            [0,   0,    0,    1],
            [0, -20, 39.24,   0],
        ],
        dtype=float,
    )
    B = 100 * np.array([[0], [1.0], [0], [2.0]], dtype=float)
    C = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]], dtype=float)

    # --- MPC horizon and sample time (matches lmpc_invpend.py) ---
    Np = 50   # prediction horizon
    Nc = 50   # control horizon (= Np → no move blocking; all solvers use full horizon)
    Ts = 0.01  # sample time [s]

    # --- Objective weights ---
    # All five solvers use the augmented-state increment (Δu) cost:
    #   J = Σ ||C x_k - r||²_Q  +  Σ ||Δu_k||²_Rr  +  terminal
    # R=0 (no direct input cost): controllers are only penalised for changing u.
    # This is achieved for all solvers by augmenting the state with u_prev so that
    # all five literally solve the same OCP.
    Q  = np.array([1.44, 1.0])  # output weight (diagonal of Q_y)
    R  = np.array([0.0])        # direct input weight — set to 0
    Rr = np.array([0.01])       # input-increment weight (used by all solvers)

    # --- Constraints ---
    # Original lmpc_invpend.py bounds.  With R=0.01 the unconstrained optimal
    # initial force is ~11.6 N → saturates at ±2.0, creating a constrained
    # transient for all solvers.
    u_lb = np.array([-2.0])
    u_ub = np.array([2.0])
    # Output bounds kept for reference; only used when soft_weight > 0.
    y_lb = np.array([-10.0, -0.30])
    y_ub = np.array([10.0,   0.30])
    # soft_weight > 0 → pole-angle soft constraint active; set to 0 to disable
    soft_weight = 1e3

    # --- Simulation setup ---
    # Larger initial pole tilt (11.5°) forces input saturation on early steps.
    x0_sim = np.array([0.0, 0.0, 0.2, 0.0])
    n_steps = 500  # simulation steps (5 s at Ts=0.01)

    # Time-varying reference: cart moves to +1.0 m for the first half, then
    # reverses to -0.5 m for the second half.  This creates two constrained
    # transients and clearly distinguishes MPC from a fixed-gain LQR.
    r1 = np.array([1.0, 0.0])   # first  half reference
    r2 = np.array([-0.5, 0.0])  # second half reference
    r_sim = r1.copy()           # "nominal" single reference (used for warm-up)
    r_traj = np.vstack([
        np.tile(r1, (n_steps // 2, 1)),
        np.tile(r2, (n_steps - n_steps // 2, 1)),
    ])

    return _build_problem(
        name="inverted_pendulum",
        A=A, B=B, C=C,
        Np=Np, Nc=Nc, Ts=Ts,
        Q=Q, Rr=Rr, R=R,
        u_lb=u_lb, u_ub=u_ub,
        y_lb=y_lb, y_ub=y_ub,
        soft_weight=soft_weight,
        x0_sim=x0_sim,
        r_sim=r_sim,
        r_traj=r_traj,
        n_steps=n_steps,
    )


def _double_integrator() -> dict:
    """
    Simple double integrator: position + velocity.

    State: x = [position, velocity]
    Input: u = [acceleration]
    Output: y = [position]
    """
    A = np.array([[0, 1], [0, 0]], dtype=float)
    B = np.array([[0], [1.0]], dtype=float)
    C = np.array([[1.0, 0]], dtype=float)

    Np = 20
    Nc = 5
    Ts = 0.1

    Q  = np.array([1.0])
    Rr = np.array([0.1])
    R  = np.array([0.0])

    u_lb = np.array([-1.0])
    u_ub = np.array([1.0])
    y_lb = np.array([-5.0])
    y_ub = np.array([5.0])
    soft_weight = 1e3

    x0_sim = np.array([0.0, 0.0])
    r_sim  = np.array([3.0])
    n_steps = 200

    return _build_problem(
        name="double_integrator",
        A=A, B=B, C=C,
        Np=Np, Nc=Nc, Ts=Ts,
        Q=Q, Rr=Rr, R=R,
        u_lb=u_lb, u_ub=u_ub,
        y_lb=y_lb, y_ub=y_ub,
        soft_weight=soft_weight,
        x0_sim=x0_sim,
        r_sim=r_sim,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _build_problem(
    name, A, B, C,
    Np, Nc, Ts,
    Q, Rr, R,
    u_lb, u_ub,
    y_lb, y_ub,
    soft_weight,
    x0_sim, r_sim, n_steps,
    r_traj=None,
) -> dict:
    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]

    # --- Discretize (zero-order hold) ---
    sys_d = cont2discrete((A, B, C, np.zeros((ny, nu))), Ts, method="zoh")
    Ad, Bd = sys_d[0], sys_d[1]

    # --- Augmented increment form (kept for reference / lmpc internal use) ---
    # ξ = [x; u_{k-1}],  Δu input,  ξ_{k+1} = Aa ξ + Ba Δu
    Aa = np.block([[Ad, Bd], [np.zeros((nu, nx)), np.eye(nu)]])
    Ba = np.vstack([Bd, np.eye(nu)])
    Ca = np.hstack([C, np.zeros((ny, nu))])
    na = nx + nu

    # --- Direct-input condensed prediction matrices ---
    # y_k = C Ad^k x0 + Σ_{j=0}^{k-1} C Ad^{k-1-j} Bd u_j
    # Y = F x0 + G U   where Y ∈ R^{Np*ny}, U ∈ R^{Nc*nu}
    # F[k]   = C Ad^{k+1}               (free response, k=0..Np-1)
    # G[k,j] = C Ad^{k-j} Bd            (forced response, j ≤ k)
    # With Nc = Np (no move blocking) G is lower block triangular.
    F = np.zeros((Np * ny, nx))
    G = np.zeros((Np * ny, Nc * nu))

    Adpow = np.eye(nx)
    for i in range(Np):
        Adpow = Adpow @ Ad                          # Ad^{i+1}
        F[i * ny:(i + 1) * ny, :] = C @ Adpow
        for j in range(min(i + 1, Nc)):
            G[i * ny:(i + 1) * ny, j * nu:(j + 1) * nu] = (
                C @ np.linalg.matrix_power(Ad, i - j) @ Bd
            )

    # --- Stage output weight Q_bar (pure stage, same weight at every step) ---
    Q_bar_stage = np.kron(np.eye(Np), np.diag(Q))  # Np*ny × Np*ny  (no separate terminal)

    # --- DARE terminal cost on the augmented state ξ = [x; u_prev] ---
    # All five solvers use this Riccati terminal so the OCP is identical.
    # TinyMPC computes this internally; the other four are set explicitly.
    Q_aug = Ca.T @ np.diag(Q) @ Ca         # na × na  (stage state cost)
    P_f_aug = solve_discrete_are(Aa, Ba, Q_aug, np.diag(Rr))  # DARE solution
    W_f_dare = Ca @ P_f_aug @ Ca.T         # ny × ny  output-space terminal (for lmpc Qf)

    # Q_bar with uniform stage weight at all steps (terminal = stage weight).
    # Using DARE output terminal W_f_dare here would make the terminal 58× larger
    # and cause SQP_RTI-based solvers (acados) to be overly aggressive.
    W_f = np.diag(Q)
    Q_bar = Q_bar_stage.copy()
    Q_bar[(Np - 1) * ny:Np * ny, (Np - 1) * ny:Np * ny] = W_f

    # Legacy stage-only P_f (not used by any solver now)
    P_f = C.T @ np.diag(Q) @ C

    # Direct-input Hessian: H = G' Q_bar G + R_bar  (Nc*nu × Nc*nu)
    R_bar = np.kron(np.eye(Nc), np.diag(R))  # Nc*nu × Nc*nu
    H_qp  = G.T @ Q_bar @ G + R_bar

    # --- Augmented (Δu) condensed prediction matrices ---
    # ξ = [x; u_prev],  Δu input.  All five solvers use this identical OCP:
    #   J = Σ_{k=0}^{N-1}||Ca ξ_k - r||²_Q + Σ||Δu_k||²_Rr + ξ_N' P_f_aug ξ_N
    #
    # Fa[k]   = Ca Aa^{k+1}           (output free response)
    # Ga[k,j] = Ca Aa^{k-j} Ba        (output forced response)
    # Ga_state_last[:,j] = Aa^{N-1-j} Ba  (last-step STATE forced response, na×Nc*nu)
    # H_qp_a_dare = Ga' Q_bar_stage Ga + Rr_bar + Ga_state_last' P_f_aug Ga_state_last
    #
    # Cumulative-u constraint matrix L_tri (for CasADi condensed form):
    # u_k = u_prev + Σ_{j=0}^{k} Δu_j  ⟹  U_vec = u_prev_rep + L_tri ⊗ I_nu · ΔU
    Fa = np.zeros((Np * ny, na))
    Ga = np.zeros((Np * ny, Nc * nu))

    Aapow = np.eye(na)
    for i in range(Np):
        Aapow = Aapow @ Aa
        Fa[i * ny:(i + 1) * ny, :] = Ca @ Aapow
        for j in range(min(i + 1, Nc)):
            Ga[i * ny:(i + 1) * ny, j * nu:(j + 1) * nu] = (
                Ca @ np.linalg.matrix_power(Aa, i - j) @ Ba
            )

    # Last-step state predictions: xi_N = Aa^N xi0 + Ga_state_last @ dU
    Ga_state_last = np.zeros((na, Nc * nu))
    for j in range(Nc):
        Ga_state_last[:, j * nu:(j + 1) * nu] = np.linalg.matrix_power(Aa, Np - 1 - j) @ Ba

    # Aa^N and pseudoinverse Ca† for gradient computation in condensed solvers
    Aap_N   = np.linalg.matrix_power(Aa, Np)                   # na × na
    Ca_pinv = Ca.T @ np.linalg.inv(Ca @ Ca.T)                   # na × ny

    Rr_bar  = np.kron(np.eye(Nc), np.diag(Rr))                  # Nc*nu × Nc*nu
    H_dare_terminal = Ga_state_last.T @ P_f_aug @ Ga_state_last # Nc*nu × Nc*nu  (informational)
    # H_qp_a uses Q_bar (with DARE output terminal at last block), matching CasADi/lmpc.
    # The full-state DARE terminal (H_qp_a_dare) is available for reference.
    H_qp_a      = Ga.T @ Q_bar @ Ga + Rr_bar                    # Δu Hessian w/ output terminal
    H_qp_a_dare = Ga.T @ Q_bar_stage @ Ga + Rr_bar + H_dare_terminal  # w/ state DARE terminal

    # Lower-triangular cumulation matrix: [L_tri ⊗ I_nu] @ ΔU = cumsum(ΔU)
    L_tri = np.kron(np.tril(np.ones((Nc, Nc))), np.eye(nu))

    return dict(
        name=name,
        # Continuous-time
        A=A, B=B, C=C,
        # Discrete-time
        Ad=Ad, Bd=Bd,
        # Augmented increment form (for lmpc reference)
        Aa=Aa, Ba=Ba, Ca=Ca,
        # Direct-input condensed prediction
        F=F, G=G,
        # Augmented (Δu) condensed prediction (output)
        Fa=Fa, Ga=Ga,
        # Augmented (Δu) last-step STATE prediction (for DARE terminal gradient)
        Ga_state_last=Ga_state_last,
        Aap_N=Aap_N,
        Ca_pinv=Ca_pinv,
        # Cumulative-u constraint matrix (for CasADi condensed Δu form)
        L_tri=L_tri,
        # Sizes
        nx=nx, nu=nu, ny=ny, na=na,
        # Horizons
        Np=Np, Nc=Nc, Ts=Ts,
        # Weights
        Q=Q, Rr=Rr, R=R,
        Q_bar=Q_bar,
        Q_bar_stage=Q_bar_stage,
        R_bar=R_bar,
        Rr_bar=Rr_bar,
        H_qp=H_qp,
        H_qp_a=H_qp_a,
        H_qp_a_dare=H_qp_a_dare,
        H_dare_terminal=H_dare_terminal,
        # Terminal cost (DARE-based, same across all solvers)
        P_f_aug=P_f_aug,   # na × na  DARE terminal on augmented state (used by all)
        W_f_dare=W_f_dare, # ny × ny  output-space DARE terminal (for lmpc Qf)
        P_f=P_f,           # nx × nx  legacy (C'QC, kept for backward compat.)
        W_f=W_f,           # ny × ny  = W_f_dare
        # Bounds
        u_lb=u_lb, u_ub=u_ub,
        y_lb=y_lb, y_ub=y_ub,
        soft_weight=soft_weight,
        # Simulation
        x0_sim=x0_sim,
        r_sim=r_sim,
        r_traj=r_traj,
        n_steps=n_steps,
    )


if __name__ == "__main__":
    prob = get_problem("inverted_pendulum")
    print(f"Problem: {prob['name']}")
    print(f"  States nx={prob['nx']}, inputs nu={prob['nu']}, outputs ny={prob['ny']}")
    print(f"  Horizon Np={prob['Np']}, control horizon Nc={prob['Nc']}, Ts={prob['Ts']}")
    print(f"  H_qp shape: {prob['H_qp'].shape}, cond(H_qp)={np.linalg.cond(prob['H_qp']):.2f}")


def _apply_overrides(prob: dict, overrides: dict) -> dict:
    """Rebuild a problem with simple parameter overrides such as Np, Nc, Ts, n_steps."""
    if not overrides:
        return prob
    keys = {"Np", "Nc", "Ts", "n_steps"}
    if not any(k in overrides for k in keys):
        out = prob.copy()
        out.update(overrides)
        return out

    return _build_problem(
        name=prob["name"],
        A=prob["A"], B=prob["B"], C=prob["C"],
        Np=int(overrides.get("Np", prob["Np"])),
        Nc=int(overrides.get("Nc", prob["Nc"])),
        Ts=float(overrides.get("Ts", prob["Ts"])),
        Q=prob["Q"], Rr=prob["Rr"], R=prob["R"],
        u_lb=prob["u_lb"], u_ub=prob["u_ub"],
        y_lb=prob["y_lb"], y_ub=prob["y_ub"],
        soft_weight=prob["soft_weight"],
        x0_sim=prob["x0_sim"],
        r_sim=prob["r_sim"],
        n_steps=int(overrides.get("n_steps", prob["n_steps"])),
        r_traj=prob.get("r_traj"),
    )
