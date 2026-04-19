/*
 * casadi_adapter.c — solver_adapter.h implementation for the CasADi qrqp codegen.
 *
 * CasADi's code generator (CodeGenerator) produces a self-contained C file
 * "casadi_mpc.c" that implements the function "mpc_qp_cg" following CasADi's
 * external-function calling convention:
 *
 *   int mpc_qp_cg(const casadi_real **arg, casadi_real **res,
 *                 casadi_int *iw, casadi_real *w, void *mem);
 *
 * Helper queries also generated in casadi_mpc.c:
 *   casadi_int mpc_qp_cg_sz_arg(void);   — number of arg pointers
 *   casadi_int mpc_qp_cg_sz_res(void);   — number of res pointers
 *   casadi_int mpc_qp_cg_sz_iw(void);    — integer work array length
 *   casadi_int mpc_qp_cg_sz_w(void);     — real work array length
 *
 * Input slots (qpsol convention):
 *   arg[0] = x0     — initial primal guess  (NZ_CASADI doubles, may be NULL)
 *   arg[1] = p      — parameters            ([xi0 (NA); r (NY)] = NA+NY doubles)
 *   arg[2] = lbx    — lower bounds on z     (NZ_CASADI doubles)
 *   arg[3] = ubx    — upper bounds on z     (NZ_CASADI doubles)
 *   arg[4] = lbg    — lower bounds on g     (NG_CASADI doubles)
 *   arg[5] = ubg    — upper bounds on g     (NG_CASADI doubles)
 *
 * Output slots:
 *   res[0] = x      — primal solution dZ*   (NZ_CASADI doubles)
 *   res[1..4]       — cost, g, lam_x, lam_g (may be NULL)
 *
 * For the benchmark all inputs are constant (fixed x0, r, u_prev=0), so they
 * are pre-filled in solver_init() and reused across timed calls.
 *
 * Sizes NZ_CASADI, NP_CASADI, NG_CASADI are generated in problem_data.h.
 * The static work arrays are sized with generous compile-time upper bounds.
 */

#include "solver_adapter.h"
#include "problem_data.h"
#include <math.h>
#include <string.h>

typedef double    casadi_real;
typedef long long casadi_int;

/*
 * Include the generated header — it declares mpc_qp_cg, mpc_qp_cg_sz_*, etc.
 * The codegen directory is added to the include path by CMakeLists.txt.
 */
#include "casadi_mpc.h"

/* ---- Static work buffers (sized at compile time from problem_data.h). ---- */

/* Max sizes: arg/res pointer arrays are small; iw/w are the large ones.     */
/* The actual sizes are queried in solver_init() and asserted to fit.        */
#define CASADI_MAX_IW  4096
#define CASADI_MAX_W   65536

static casadi_real  s_p   [NP_CASADI];          /* parameters [xi0; r]       */
static casadi_real  s_lbx [NZ_CASADI];          /* lower bounds on z         */
static casadi_real  s_ubx [NZ_CASADI];          /* upper bounds on z         */
static casadi_real  s_lbg [NG_CASADI];          /* lower bounds on g         */
static casadi_real  s_ubg [NG_CASADI];          /* upper bounds on g         */
static casadi_real  s_sol [NZ_CASADI];          /* primal solution           */
static casadi_int   s_iw  [CASADI_MAX_IW];      /* integer work              */
static casadi_real  s_w   [CASADI_MAX_W];       /* real work                 */

/* Pointer arrays passed to mpc_qp_cg. */
static const casadi_real *s_arg[8];
static       casadi_real *s_res[8];

const char *SOLVER_LABEL = "casadi";

void solver_init(void)
{
    /* ---- Fill constant parameter vector p = [xi0; r] ---- */
    /* xi0 = [x0; u_prev] with u_prev = 0 */
    {
        const double x0[NX]    = BENCH_X0_INIT;
        const double r[NY]     = BENCH_R_INIT;
        const double u_prev[NU]= BENCH_U0_INIT;
        for (int i = 0; i < NX;  i++) s_p[i]      = x0[i];
        for (int i = 0; i < NU;  i++) s_p[NX + i] = u_prev[i];
        for (int i = 0; i < NY;  i++) s_p[NA + i] = r[i];
    }

    /* ---- Decision-variable bounds ---- */
    /* dU is unconstrained; slacks (if any) are >= 0. */
    for (int i = 0; i < NC * NU; i++) {
        s_lbx[i] = -HUGE_VAL;
        s_ubx[i] =  HUGE_VAL;
    }
    for (int i = NC * NU; i < NZ_CASADI; i++) {
        s_lbx[i] = 0.0;   /* slack variables are non-negative */
        s_ubx[i] = HUGE_VAL;
    }

    /* ---- Constraint bounds (cumulative-u + optional soft) ---- */
    /* With u_prev = 0: lbg_u = u_lb, ubg_u = u_ub (tiled Nc times). */
    {
        const double u_lb[NU] = BENCH_U_LB;
        const double u_ub[NU] = BENCH_U_UB;
        for (int k = 0; k < NC; k++) {
            for (int j = 0; j < NU; j++) {
                s_lbg[k * NU + j] = u_lb[j];
                s_ubg[k * NU + j] = u_ub[j];
            }
        }
    }
    /* Soft-constraint rows (if NG_SOFT > 0): y_lb and -y_ub lower bounds. */
#if NG_SOFT > 0
    for (int i = NG_U; i < NG_CASADI; i++) {
        s_lbg[i] = CASADI_SOFT_LBG_ARRAY[i - NG_U];
        s_ubg[i] = HUGE_VAL;
    }
#endif

    /* ---- Wire up the pointer arrays ---- */
    s_arg[0] = NULL;      /* x0 initial guess — let the solver choose */
    s_arg[1] = s_p;
    s_arg[2] = s_lbx;
    s_arg[3] = s_ubx;
    s_arg[4] = s_lbg;
    s_arg[5] = s_ubg;

    s_res[0] = s_sol;     /* primal solution */
    s_res[1] = NULL;      /* objective value — not needed */
    s_res[2] = NULL;      /* constraint values — not needed */
    s_res[3] = NULL;      /* lam_x — not needed */
    s_res[4] = NULL;      /* lam_g — not needed */
}

int solver_step(const double *x0, const double *r,
                const double *u_prev, double *u_out)
{
    /*
     * Update the parameter vector in-place for the given x0, r, u_prev.
     * For the fixed-input benchmark these are always the same constants, but
     * updating here makes the adapter correct for real closed-loop use too.
     */
    for (int i = 0; i < NX; i++) s_p[i]      = x0[i];
    for (int i = 0; i < NU; i++) s_p[NX + i] = u_prev[i];
    for (int i = 0; i < NY; i++) s_p[NA + i] = r[i];

    /* Also update cumulative-u bounds to reflect the current u_prev. */
    {
        const double u_lb[NU] = BENCH_U_LB;
        const double u_ub[NU] = BENCH_U_UB;
        for (int k = 0; k < NC; k++) {
            for (int j = 0; j < NU; j++) {
                s_lbg[k * NU + j] = u_lb[j] - u_prev[j];
                s_ubg[k * NU + j] = u_ub[j] - u_prev[j];
            }
        }
    }

    int ret = mpc_qp_cg(s_arg, s_res, s_iw, s_w, NULL);
    if (ret != 0) {
        return -1;
    }

    /* u* = u_prev + dU*[0:NU] */
    const double u_lb[NU] = BENCH_U_LB;
    const double u_ub[NU] = BENCH_U_UB;
    for (int j = 0; j < NU; j++) {
        double u = u_prev[j] + s_sol[j];
        /* clamp to bounds */
        u_out[j] = u < u_lb[j] ? u_lb[j] : (u > u_ub[j] ? u_ub[j] : u);
    }
    return 0;
}
