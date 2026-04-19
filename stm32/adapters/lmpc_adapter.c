/*
 * lmpc_adapter.c — solver_adapter.h implementation for lmpc generated code.
 *
 * The lmpc Julia package generates a self-contained C translation unit that
 * exposes a single entry point:
 *
 *   int mpc_compute_control(double *u, double *x, double *r, double *d)
 *
 * where:
 *   u – output  (NU doubles) – the computed absolute control
 *   x – input   (NX doubles) – current plant state
 *   r – input   (NY doubles) – output reference
 *   d – input   (NU doubles) – disturbance (pass zeros for the benchmark)
 *
 * The generated code maintains the previous control internally, so u_prev
 * need not be passed in explicitly.  solver_init() performs one warm call so
 * the lmpc internal state is consistent with the benchmark initial conditions.
 *
 * Return value > 0 indicates a successful solve; ≤ 0 indicates failure.
 */

#include "solver_adapter.h"
#include "problem_data.h"
#include <string.h>

/* Forward-declare the generated function (defined in codegen/lmpc_codegen/*.c). */
extern int mpc_compute_control(double *u, double *x, double *r, double *d);

const char *SOLVER_LABEL = "lmpc";

void solver_init(void)
{
    /* Warm the internal lmpc state with the benchmark initial conditions. */
    double u_tmp[NU];
    double d[NU];
    double x0[NX]   = BENCH_X0_INIT;
    double r0[NY]   = BENCH_R_INIT;
    memset(d, 0, sizeof(d));
    mpc_compute_control(u_tmp, x0, r0, d);
}

int solver_step(const double *x0, const double *r,
                const double *u_prev, double *u_out)
{
    double d[NU];
    memset(d, 0, sizeof(d));
    /* Cast away const: the generated code does not modify the inputs. */
    (void)u_prev;   /* lmpc manages u_prev internally */
    int status = mpc_compute_control(u_out, (double *)x0, (double *)r, d);
    return (status > 0) ? 0 : -1;
}
