/*
 * cvxpygen_adapter.c — solver_adapter.h implementation for CVXPYgen/OSQP codegen.
 *
 * CVXPYgen generates a complete OSQP-based solver in C under
 * codegen/cvxpygen_codegen/.  The key generated files are:
 *
 *   cpg_solve.c / cpg_solve.h — the entry point
 *   cpg_workspace.c           — static OSQP workspace (matrices, vectors)
 *   cpg_params.h              — parameter and variable type definitions
 *
 * Generated API (from cpg_solve.h):
 *
 *   // Parameter setter functions — call before every cpg_solve() call
 *   void cpg_update_xi0(const CPGData_xi0_t *data);
 *   void cpg_update_r  (const CPGData_r_t   *data);
 *
 *   // Solve (uses the static workspace; no heap allocation in normal use)
 *   void cpg_solve(void);
 *
 *   // Solution access
 *   extern CPGResult_t cpg_result;   // contains .obj_val, .iters, …
 *   // Variable values are in the generated global arrays, e.g.
 *   extern double Xi_value[NA * (NP_HORIZON + 1)];
 *   extern double dU_value[NU * NP_HORIZON];
 *
 * NOTE: The exact field names in CPGData_* and the variable arrays depend on
 * the version of CVXPYgen and on the parameter names used in the CVXPY problem.
 * Verify them against the generated cpg_params.h and cpg_solve.h before building.
 * The names below match CVXPYgen ≥ 0.4 with the inverted-pendulum problem as
 * defined in this repository (parameters "xi0" and "r", variables "Xi", "dU").
 *
 * Memory: OSQP uses the static workspace embedded in cpg_workspace.c.  No
 * malloc() is called in normal OSQP operation.  The heap reserved in the
 * linker script (_Min_Heap_Size = 8 KB) is a safety margin for newlib stubs.
 */

#include "solver_adapter.h"
#include "problem_data.h"

/* Generated headers — paths resolved by CMakeLists.txt include directories. */
#include "cpg_solve.h"     /* cpg_solve(), cpg_update_xi0(), cpg_update_r()   */
#include "cpg_workspace.h" /* cpg_result, variable arrays                     */

const char *SOLVER_LABEL = "cvxpygen";

void solver_init(void)
{
    /* Nothing extra to do — the generated cpg_workspace.c defines the static
     * OSQP workspace.  A warm-up solve is performed here so OSQP's internal
     * state (warm-starting) is consistent before the timed loop. */
    const double x0[NX]    = BENCH_X0_INIT;
    const double r[NY]     = BENCH_R_INIT;
    const double u_prev[NU]= BENCH_U0_INIT;
    double u_dummy[NU];
    solver_step(x0, r, u_prev, u_dummy);
}

int solver_step(const double *x0, const double *r,
                const double *u_prev, double *u_out)
{
    /* Build xi0 = [x0; u_prev] and pass it to the generated parameter setter. */
    CPGData_xi0_t xi0_data;
    for (int i = 0; i < NX; i++) xi0_data.xi0[i]      = x0[i];
    for (int i = 0; i < NU; i++) xi0_data.xi0[NX + i] = u_prev[i];

    CPGData_r_t r_data;
    for (int i = 0; i < NY; i++) r_data.r[i] = r[i];

    cpg_update_xi0(&xi0_data);
    cpg_update_r  (&r_data);
    cpg_solve();

    /* dU* is the first column of the dU trajectory. */
    const double u_lb[NU] = BENCH_U_LB;
    const double u_ub[NU] = BENCH_U_UB;
    for (int j = 0; j < NU; j++) {
        double u = u_prev[j] + dU_value[j];   /* dU_value[] from cpg_workspace */
        u_out[j] = u < u_lb[j] ? u_lb[j] : (u > u_ub[j] ? u_ub[j] : u);
    }
    return (cpg_result.iters > 0) ? 0 : -1;
}
