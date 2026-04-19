/*
 * Minimal UART interface (USART2, 115 200 Bd, 8N1, PA2-TX / PA3-RX).
 * This pin-out matches the ST-LINK virtual COM port on the Nucleo-F411RE.
 */

#ifndef UART_H
#define UART_H

#include <stdint.h>

/* Initialise USART2 for 115 200 Bd, 8N1 (assumes 96 MHz PLL already active). */
void uart_init(void);

/* Transmit a single character (blocking). */
void uart_putchar(char c);

/* Transmit a NUL-terminated string. */
void uart_puts(const char *s);

/* Transmit a 32-bit unsigned integer in decimal without leading zeros. */
void uart_print_u32(uint32_t v);

/* Transmit a 64-bit unsigned integer in decimal. */
void uart_print_u64(uint64_t v);

#endif /* UART_H */
