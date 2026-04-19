/*
 * DWT cycle-counter helpers for Cortex-M4 (STM32F411).
 *
 * Usage:
 *   dwt_init();                   // call once in main (enables the counter)
 *   uint32_t t0 = dwt_count();    // capture start
 *   do_work();
 *   uint32_t cycles = dwt_count() - t0;
 *
 * The counter is 32-bit and wraps at 2^32. At 96 MHz this wraps every ~44 s,
 * which is ample for the benchmark (each solve is well under 1 s).
 */

#ifndef DWT_TIMER_H
#define DWT_TIMER_H

#include <stdint.h>

/* Core Debug / DWT register addresses (ARM DDI 0403 §C1.8) */
#define _DEMCR      (*(volatile uint32_t *)0xE000EDFCu)  /* Debug Exception and Monitor CR */
#define _DWT_CTRL   (*(volatile uint32_t *)0xE0001000u)  /* DWT control register           */
#define _DWT_CYCCNT (*(volatile uint32_t *)0xE0001004u)  /* Cycle count register           */

/* Enable the DWT cycle counter. Call once before any dwt_count() call. */
static inline void dwt_init(void)
{
    _DEMCR      |= (1u << 24);  /* TRCENA: enable DWT and ITM            */
    _DWT_CYCCNT  = 0u;          /* reset counter                         */
    _DWT_CTRL   |= (1u << 0);   /* CYCCNTENA: start counting CPU cycles  */
}

/* Return the current DWT cycle count. */
static inline uint32_t dwt_count(void)
{
    return _DWT_CYCCNT;
}

#endif /* DWT_TIMER_H */
