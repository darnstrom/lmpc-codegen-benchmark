/*
 * bench_main.c — STM32F411 MPC benchmark harness.
 *
 * Measures DWT cycle counts for repeated solver_step() calls and prints a
 * single CSV row over USART2 (115 200 Bd, PA2/PA3).
 *
 * Output format (sent once then the MCU halts):
 *
 *   BENCH_START\r\n
 *   <solver>,<n_warmup>,<n_timed>,<sysclk_hz>,<mean>,<max>,<min>,<median>\r\n
 *   BENCH_END\r\n
 *
 * The host script (stm32_benchmark.py run) waits for BENCH_START, reads the
 * data line, then parses it.  Fields after <sysclk_hz> are cycle counts.
 * Divide by sysclk_hz to obtain seconds.
 *
 * Compile-time knobs (override via -D):
 *   BENCH_N_WARMUP  – warm-up iterations before measurement (default 5)
 *   BENCH_N_TIMED   – measured iterations                   (default 50)
 *   BENCH_SYSCLK_HZ – system clock in Hz                    (default 96 000 000)
 */

#include <stdint.h>
#include <string.h>

#include "dwt_timer.h"
#include "uart.h"
#include "solver_adapter.h"
#include "problem_data.h"   /* NX, NU, NY, BENCH_X0_INIT, BENCH_R_INIT, BENCH_U0_INIT */

/* ---- Compile-time defaults ---- */
#ifndef BENCH_N_WARMUP
#  define BENCH_N_WARMUP  5
#endif
#ifndef BENCH_N_TIMED
#  define BENCH_N_TIMED  50
#endif
#ifndef BENCH_SYSCLK_HZ
#  define BENCH_SYSCLK_HZ  96000000UL
#endif

/* ---- Problem data (constant across all timed runs) ---- */
static const double g_x0[NX]    = BENCH_X0_INIT;
static const double g_r[NY]     = BENCH_R_INIT;
static const double g_u_prev[NU]= BENCH_U0_INIT;
static       double g_u_out[NU];

/* ---- Cycle-count storage ---- */
static uint32_t g_cycles[BENCH_N_TIMED];

/* ---- Statistics helpers ---- */

static uint32_t stats_max(const uint32_t *a, int n)
{
    uint32_t m = a[0];
    for (int i = 1; i < n; i++) { if (a[i] > m) m = a[i]; }
    return m;
}

static uint32_t stats_min(const uint32_t *a, int n)
{
    uint32_t m = a[0];
    for (int i = 1; i < n; i++) { if (a[i] < m) m = a[i]; }
    return m;
}

static uint32_t stats_mean(const uint32_t *a, int n)
{
    uint64_t s = 0;
    for (int i = 0; i < n; i++) { s += a[i]; }
    return (uint32_t)(s / (uint64_t)n);
}

/* In-place insertion sort, then return middle element. */
static uint32_t stats_median(uint32_t *a, int n)
{
    for (int i = 1; i < n; i++) {
        uint32_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
    return a[n / 2];
}

/* ---- Entry point ---- */
int main(void)
{
    /* dwt_init() relies on DEMCR which is always accessible. */
    dwt_init();

    /* UART init (clock already configured by startup.c PLL setup). */
    uart_init();

    /* One-time solver initialisation (ADMM cache, static workspace, etc.). */
    solver_init();

    /* Warm-up: let the solver reach a steady cache / branch-predictor state. */
    for (int i = 0; i < BENCH_N_WARMUP; i++) {
        solver_step(g_x0, g_r, g_u_prev, g_u_out);
    }

    /* Timed runs — always the same fixed input so results are reproducible. */
    for (int i = 0; i < BENCH_N_TIMED; i++) {
        uint32_t t0 = dwt_count();
        solver_step(g_x0, g_r, g_u_prev, g_u_out);
        g_cycles[i] = dwt_count() - t0;
    }

    /* Compute statistics. */
    uint32_t cyc_mean   = stats_mean  (g_cycles, BENCH_N_TIMED);
    uint32_t cyc_max    = stats_max   (g_cycles, BENCH_N_TIMED);
    uint32_t cyc_min    = stats_min   (g_cycles, BENCH_N_TIMED);
    uint32_t cyc_median = stats_median(g_cycles, BENCH_N_TIMED); /* sorts in-place */

    /* Print CSV over UART. */
    uart_puts("BENCH_START\r\n");
    uart_puts(SOLVER_LABEL);  uart_putchar(',');
    uart_print_u32(BENCH_N_WARMUP); uart_putchar(',');
    uart_print_u32(BENCH_N_TIMED);  uart_putchar(',');
    uart_print_u32(BENCH_SYSCLK_HZ); uart_putchar(',');
    uart_print_u32(cyc_mean);   uart_putchar(',');
    uart_print_u32(cyc_max);    uart_putchar(',');
    uart_print_u32(cyc_min);    uart_putchar(',');
    uart_print_u32(cyc_median);
    uart_puts("\r\nBENCH_END\r\n");

    /* Halt — blink or just spin. */
    while (1) {
        __asm__ volatile ("wfe");
    }

    return 0;
}
