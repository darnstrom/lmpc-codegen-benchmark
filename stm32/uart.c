/*
 * USART2 driver -- PA2 (TX) / PA3 (RX), 115 200 Bd, 8N1.
 *
 * APB1 clock = 48 MHz (PCLK1 after the /2 prescaler applied in startup.c).
 * BRR  = 48 000 000 / (16 * 115 200) = 26.04 -> mantissa 26, fraction 1
 *        -> BRR register value 0x1A1 (26<<4 | 1)
 * Actual baud ~= 48 000 000 / (16 * 26.0625) = 115 108 Bd  (error < 0.1 %).
 *
 * No interrupts or DMA -- purely polling.  Transmit throughput is ~= 11.5 kB/s
 * which is more than enough for the small CSV lines emitted by bench_main.c.
 */

#include "uart.h"
#include <stdint.h>

/* ---- Minimal register definitions (no CMSIS header required) ---- */

/* RCC */
#define RCC_AHB1ENR (*(volatile uint32_t *)0x40023830u)
#define RCC_APB1ENR (*(volatile uint32_t *)0x40023840u)

/* GPIOA */
#define GPIOA_MODER (*(volatile uint32_t *)0x40020000u)
#define GPIOA_AFRL  (*(volatile uint32_t *)0x40020020u)

/* USART2 (base 0x40004400) */
#define USART2_SR   (*(volatile uint32_t *)0x40004400u)
#define USART2_DR   (*(volatile uint32_t *)0x40004404u)
#define USART2_BRR  (*(volatile uint32_t *)0x40004408u)
#define USART2_CR1  (*(volatile uint32_t *)0x4000440Cu)

/* ---- uart_init ---- */
void uart_init(void)
{
    /* Enable clocks: GPIOA (AHB1 bit 0) and USART2 (APB1 bit 17). */
    RCC_AHB1ENR |= (1u << 0);
    RCC_APB1ENR |= (1u << 17);

    /* PA2 and PA3 -> alternate-function mode (MODER bits [5:4] and [7:6] = 10). */
    GPIOA_MODER &= ~((3u << 4) | (3u << 6));
    GPIOA_MODER |=  ((2u << 4) | (2u << 6));

    /* Select AF7 (USART2) on PA2 (AFRL bits [11:8]) and PA3 (bits [15:12]). */
    GPIOA_AFRL  &= ~((0xFu << 8) | (0xFu << 12));
    GPIOA_AFRL  |=  ((7u   << 8) | (7u   << 12));

    /* Baud-rate divisor for 115 200 Bd at PCLK1 = 48 MHz. */
    USART2_BRR = (26u << 4) | 1u;   /* = 0x1A1 */

    /* Enable USART2: UE | TE | RE. */
    USART2_CR1 = (1u << 13) | (1u << 3) | (1u << 2);
}

/* ---- uart_putchar ---- */
void uart_putchar(char c)
{
    /* Wait until the transmit data register is empty (TXE, bit 7). */
    while (!(USART2_SR & (1u << 7))) {}
    USART2_DR = (uint8_t)c;
}

/* ---- uart_puts ---- */
void uart_puts(const char *s)
{
    while (*s) {
        uart_putchar(*s++);
    }
}

/* ---- uart_print_u32 ---- */
void uart_print_u32(uint32_t v)
{
    char buf[11];   /* max 10 decimal digits + NUL */
    int  i = sizeof(buf) - 1;
    buf[i] = '\0';
    if (v == 0u) {
        uart_putchar('0');
        return;
    }
    while (v > 0u && i > 0) {
        buf[--i] = (char)('0' + (v % 10u));
        v /= 10u;
    }
    uart_puts(buf + i);
}

/* ---- uart_print_u64 ---- */
void uart_print_u64(uint64_t v)
{
    char buf[21];   /* max 20 decimal digits + NUL */
    int  i = sizeof(buf) - 1;
    buf[i] = '\0';
    if (v == 0u) {
        uart_putchar('0');
        return;
    }
    while (v > 0u && i > 0) {
        buf[--i] = (char)('0' + (uint32_t)(v % 10u));
        v /= 10u;
    }
    uart_puts(buf + i);
}
