/*
 * tinympc_adapter.cpp — solver_adapter.h implementation for TinyMPC generated C++.
 *
 * TinyMPC generates a CMake project under codegen/tinympc_codegen/ that contains:
 *   include/tinympc/   — TinyMPC headers (admm.hpp, types.hpp, …)
 *   src/tinympc/       — TinyMPC solver sources
 *   problem_data/      — problem_data.hpp with precomputed ADMM matrices
 *                        (Kinf, Pinf, AmBKt, coeff_d2p, …) and macros:
 *                           NSTATES  == NA  (augmented state dimension)
 *                           NINPUTS  == NU  (control increment dimension)
 *                           NHORIZON == NP_HORIZON
 *   ext/               — Eigen headers (header-only, no extra compilation)
 *
 * The TinyMPC C++ API used here:
 *   TinySolver solver;
 *   tiny_setup(&solver,           // pointer to solver struct
 *              Adyn, Bdyn,        // discrete dynamics (from problem_data.hpp)
 *              Q, R,              // cost matrices
 *              rho, n_iters, …);  // ADMM parameters (from problem_data.hpp)
 *
 *   // Per step:
 *   solver.work->x.col(0) = x0_vec;       // set initial state
 *   solver.work->Xref = Xref_matrix;      // set reference (tiled)
 *   tiny_solve(&solver);
 *   u_out[j] = solver.work->u(j, 0);      // read first control
 *
 * Note: TinyMPC internally uses float (tinytype = float).  This adapter
 * converts to/from double at the boundary.
 *
 * The XI_REF_ROWS macro in problem_data.h lists the augmented-state indices
 * where each component of r is placed (derived from the Ca matrix).
 */

#include "solver_adapter.h"
#include "problem_data.h"

#include "problem_data/problem_data.hpp"
#include "tinympc/admm.hpp"

#include <cstring>
#include <cmath>

/* TinyMPC uses its own scalar type (usually float). */
using tinytype = float;

/* ---- Static solver instance ---- */
static TinySolver   s_solver;
static TinyWorkspace s_work;
static TinySettings  s_settings;
static TinySolution  s_solution;

extern "C" {

const char *SOLVER_LABEL = "tinympc";

void solver_init(void)
{
    /* Wire up the solver struct. */
    s_solver.work     = &s_work;
    s_solver.settings = &s_settings;
    s_solver.solution = &s_solution;

    /* tiny_setup fills s_work from the precomputed data in problem_data.hpp. */
    tiny_setup(&s_solver,
               Adyn, Bdyn, Q, R,
               rho_value, NHORIZON,
               x_min, x_max, u_min, u_max,
               1 /* en_state_bound */, 1 /* en_input_bound */);

    /* Apply any tuning from the generated settings. */
    s_settings.max_iter         = max_iter;
    s_settings.check_termination= check_termination;
    s_settings.abs_pri_tol      = abs_pri_tol;
    s_settings.abs_dua_tol      = abs_dua_tol;

    /* Warm-up solve with the benchmark initial conditions. */
    {
        const double x0[NA]  = BENCH_X0_INIT_AUG;  /* augmented: [x; u_prev] */
        const double r[NY]   = BENCH_R_INIT;
        double u_dummy[NU];
        solver_step(x0, r, (const double *)0 /* unused */, u_dummy);
    }
}

int solver_step(const double *x0, const double *r,
                const double *u_prev, double *u_out)
{
    /* Set initial augmented state x0 = [plant_state; u_prev].            */
    /* x0 here is already the full augmented state (NA elements).         */
    for (int i = 0; i < NA; i++) {
        s_solver.work->x(i, 0) = (tinytype)x0[i];
    }

    /* Build the augmented reference vector xi_ref from output reference r. */
    /* XI_REF_ROWS is a brace-enclosed list of state indices, e.g. {0, 2}. */
    {
        static const int ref_rows[] = XI_REF_ROWS;
        for (int k = 0; k < NHORIZON; k++) {
            for (int i = 0; i < NA; i++) {
                s_solver.work->Xref(i, k) = 0.0f;
            }
            for (int i = 0; i < NY; i++) {
                s_solver.work->Xref(ref_rows[i], k) = (tinytype)r[i];
            }
        }
    }

    tiny_solve(&s_solver);

    /* dU* is the first column of work->u; convert to absolute u. */
    /* u_prev is the last NA-NX elements of the augmented x0 argument.   */
    const double *u_prev_ptr = (u_prev != (const double *)0)
                                   ? u_prev
                                   : x0 + NX; /* embedded in augmented x0 */
    const double u_lb[NU] = BENCH_U_LB;
    const double u_ub[NU] = BENCH_U_UB;
    for (int j = 0; j < NU; j++) {
        double u = (double)u_prev_ptr[j] + (double)s_solver.work->u(j, 0);
        u_out[j] = u < u_lb[j] ? u_lb[j] : (u > u_ub[j] ? u_ub[j] : u);
    }
    return 0;

    (void)u_prev; /* suppress warning when u_prev is used via x0+NX */
}

} /* extern "C" */
