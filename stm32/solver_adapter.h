/*
 * solver_adapter.h — Solver-agnostic interface implemented by each adapter.
 *
 * Each solver's adapter file (adapters/<solver>_adapter.c or .cpp) must
 * implement these three symbols.  bench_main.c calls them without knowing
 * which solver is linked in.
 *
 * Dimensions (NX, NU, NY, NA) and benchmark constants are provided by the
 * generated problem_data.h, which is placed in the build directory by the
 * stm32_benchmark.py build command.
 */

#ifndef SOLVER_ADAPTER_H
#define SOLVER_ADAPTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One-time initialisation.  Called by bench_main before the warm-up loop.
 * Implementations may perform ADMM cache setup, workspace zeroing, etc.
 */
void solver_init(void);

/*
 * Single MPC solve step.
 *
 * Parameters (all arrays of doubles, sizes from problem_data.h):
 *   x0     – current plant state           (NX elements)
 *   r      – output reference              (NY elements)
 *   u_prev – previous applied control      (NU elements)
 *   u_out  – optimal absolute control out  (NU elements, written by callee)
 *
 * Returns 0 on success, non-zero on solver failure.
 */
int solver_step(const double *x0, const double *r,
                const double *u_prev, double *u_out);

/*
 * Short human-readable label printed in the CSV output row.
 * Defined as a string literal in each adapter, e.g. "lmpc".
 */
extern const char *SOLVER_LABEL;

#ifdef __cplusplus
}
#endif

#endif /* SOLVER_ADAPTER_H */
