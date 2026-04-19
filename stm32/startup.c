/*
 * Minimal startup code for STM32F411xE -- no CMSIS device library required.
 *
 * Responsibilities:
 *   1. Configure the PLL for 96 MHz from the 16 MHz HSI oscillator.
 *   2. Copy the .data section from Flash to SRAM.
 *   3. Zero-initialise the .bss section.
 *   4. Call __libc_init_array() so C++ global constructors run (needed by TinyMPC/Eigen).
 *   5. Branch to main().
 *   6. Provide the complete STM32F411 interrupt vector table with weak default handlers.
 *
 * Peripheral pin-out assumed by uart.c (USART2 at 115 200 Bd):
 *   PA2 -> USART2_TX  (connected to ST-LINK VCP on Nucleo-F411RE)
 *   PA3 -> USART2_RX
 */

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Linker-script symbols
 * ---------------------------------------------------------------------- */
extern uint32_t _estack;    /* initial stack pointer (top of RAM)         */
extern uint32_t _sidata;    /* load address of .data in Flash             */
extern uint32_t _sdata;     /* start of .data in RAM                      */
extern uint32_t _edata;     /* end   of .data in RAM                      */
extern uint32_t _sbss;      /* start of .bss                              */
extern uint32_t _ebss;      /* end   of .bss                              */

/* -------------------------------------------------------------------------
 * External symbols
 * ---------------------------------------------------------------------- */
extern void __libc_init_array(void); /* runs C++ constructors              */
int main(void);

/* -------------------------------------------------------------------------
 * Minimal register definitions for clock setup
 * (RCC + Flash interface -- no full CMSIS header needed)
 * ---------------------------------------------------------------------- */
#define RCC_CR      (*(volatile uint32_t *)0x40023800u)
#define RCC_PLLCFGR (*(volatile uint32_t *)0x40023804u)
#define RCC_CFGR    (*(volatile uint32_t *)0x40023808u)
#define FLASH_ACR   (*(volatile uint32_t *)0x40023C00u)

/*
 * clock_init_96mhz()
 *
 * Configure PLL: HSI(16 MHz) -> VCO(192 MHz) -> SYSCLK(96 MHz)
 *   PLLM = 8  -> VCO input = 2 MHz
 *   PLLN = 96 -> VCO output = 192 MHz
 *   PLLP = 2  -> SYSCLK = 96 MHz
 *   PLLQ = 4  -> USB/SDIO clock = 48 MHz
 * Bus prescalers:
 *   AHB  /1 -> HCLK   = 96 MHz
 *   APB1 /2 -> PCLK1  = 48 MHz  (USART2 clock)
 *   APB2 /1 -> PCLK2  = 96 MHz
 */
static void clock_init_96mhz(void)
{
    /* 1. Make sure HSI is on and stable (it is after reset, but be safe). */
    RCC_CR |= (1u << 0);                /* HSION */
    while (!(RCC_CR & (1u << 1))) {}    /* wait HSIRDY */

    /* 2. Raise Flash latency BEFORE increasing clock (RM0383 §3.4.1). */
    /*    96 MHz at VDD 3.3 V requires 3 wait states.                   */
    /*    Also enable instruction/data cache and prefetch buffer.        */
    FLASH_ACR = (3u)        /* LATENCY = 3 WS        */
              | (1u << 8)   /* PRFTEN  prefetch       */
              | (1u << 9)   /* ICEN    icache          */
              | (1u << 10); /* DCEN    dcache          */
    /* Dummy read to ensure the write is flushed before we touch the PLL. */
    (void)FLASH_ACR;

    /* 3. Configure PLL (still sourced from HSI). */
    /*    PLLCFGR: PLLM[5:0]=8, PLLN[14:6]=96, PLLP[17:16]=00 (/2),    */
    /*             PLLSRC[22]=0 (HSI), PLLQ[27:24]=4                    */
    RCC_PLLCFGR = (8u  <<  0)   /* PLLM  */
                | (96u <<  6)   /* PLLN  */
                | (0u  << 16)   /* PLLP=00 -> /2  */
                | (0u  << 22)   /* PLLSRC = HSI  */
                | (4u  << 24);  /* PLLQ  */

    /* 4. Enable PLL and wait for lock. */
    RCC_CR |= (1u << 24);               /* PLLON */
    while (!(RCC_CR & (1u << 25))) {}   /* wait PLLRDY */

    /* 5. Set AHB/APB prescalers, then switch system clock to PLL. */
    /*    CFGR: HPRE=0000 (/1), PPRE1=100 (/2), PPRE2=000 (/1), SW=10  */
    RCC_CFGR = (0u << 4)    /* HPRE  AHB /1        */
             | (4u << 10)   /* PPRE1 APB1 /2        */
             | (0u << 13)   /* PPRE2 APB2 /1        */
             | (2u <<  0);  /* SW = PLL             */

    /* Wait until PLL is selected as system clock (SWS = 10). */
    while (((RCC_CFGR >> 2) & 3u) != 2u) {}
}

/* -------------------------------------------------------------------------
 * Reset handler -- entry point from the vector table
 * ---------------------------------------------------------------------- */
void Reset_Handler(void)
{
    /* Configure clocks first so all peripherals are at the right speed. */
    clock_init_96mhz();

    /* Copy .data initialisation image from Flash to SRAM. */
    {
        uint32_t *src = &_sidata;
        uint32_t *dst = &_sdata;
        while (dst < &_edata) {
            *dst++ = *src++;
        }
    }

    /* Zero-fill .bss. */
    {
        uint32_t *p = &_sbss;
        while (p < &_ebss) {
            *p++ = 0u;
        }
    }

    /* Run C++ global constructors (required for TinyMPC / Eigen). */
    __libc_init_array();

    /* Hand off to application. */
    main();

    /* Should never return; spin to catch runaway execution. */
    while (1) {}
}

/* -------------------------------------------------------------------------
 * Default weak interrupt handler -- infinite loop so a fault is visible.
 * ---------------------------------------------------------------------- */
void Default_Handler(void) { while (1) {} }

#define WEAK_ALIAS(sym) \
    void sym(void) __attribute__((weak, alias("Default_Handler")))

/* Core exception handlers */
WEAK_ALIAS(NMI_Handler);
WEAK_ALIAS(HardFault_Handler);
WEAK_ALIAS(MemManage_Handler);
WEAK_ALIAS(BusFault_Handler);
WEAK_ALIAS(UsageFault_Handler);
WEAK_ALIAS(SVC_Handler);
WEAK_ALIAS(DebugMon_Handler);
WEAK_ALIAS(PendSV_Handler);
WEAK_ALIAS(SysTick_Handler);

/* STM32F411 device interrupt handlers */
WEAK_ALIAS(WWDG_IRQHandler);
WEAK_ALIAS(PVD_IRQHandler);
WEAK_ALIAS(TAMP_STAMP_IRQHandler);
WEAK_ALIAS(RTC_WKUP_IRQHandler);
WEAK_ALIAS(FLASH_IRQHandler);
WEAK_ALIAS(RCC_IRQHandler);
WEAK_ALIAS(EXTI0_IRQHandler);
WEAK_ALIAS(EXTI1_IRQHandler);
WEAK_ALIAS(EXTI2_IRQHandler);
WEAK_ALIAS(EXTI3_IRQHandler);
WEAK_ALIAS(EXTI4_IRQHandler);
WEAK_ALIAS(DMA1_Stream0_IRQHandler);
WEAK_ALIAS(DMA1_Stream1_IRQHandler);
WEAK_ALIAS(DMA1_Stream2_IRQHandler);
WEAK_ALIAS(DMA1_Stream3_IRQHandler);
WEAK_ALIAS(DMA1_Stream4_IRQHandler);
WEAK_ALIAS(DMA1_Stream5_IRQHandler);
WEAK_ALIAS(DMA1_Stream6_IRQHandler);
WEAK_ALIAS(ADC_IRQHandler);
WEAK_ALIAS(EXTI9_5_IRQHandler);
WEAK_ALIAS(TIM1_BRK_TIM9_IRQHandler);
WEAK_ALIAS(TIM1_UP_TIM10_IRQHandler);
WEAK_ALIAS(TIM1_TRG_COM_TIM11_IRQHandler);
WEAK_ALIAS(TIM1_CC_IRQHandler);
WEAK_ALIAS(TIM2_IRQHandler);
WEAK_ALIAS(TIM3_IRQHandler);
WEAK_ALIAS(TIM4_IRQHandler);
WEAK_ALIAS(I2C1_EV_IRQHandler);
WEAK_ALIAS(I2C1_ER_IRQHandler);
WEAK_ALIAS(I2C2_EV_IRQHandler);
WEAK_ALIAS(I2C2_ER_IRQHandler);
WEAK_ALIAS(SPI1_IRQHandler);
WEAK_ALIAS(SPI2_IRQHandler);
WEAK_ALIAS(USART1_IRQHandler);
WEAK_ALIAS(USART2_IRQHandler);
WEAK_ALIAS(EXTI15_10_IRQHandler);
WEAK_ALIAS(RTC_Alarm_IRQHandler);
WEAK_ALIAS(OTG_FS_WKUP_IRQHandler);
WEAK_ALIAS(DMA1_Stream7_IRQHandler);
WEAK_ALIAS(SDIO_IRQHandler);
WEAK_ALIAS(TIM5_IRQHandler);
WEAK_ALIAS(SPI3_IRQHandler);
WEAK_ALIAS(DMA2_Stream0_IRQHandler);
WEAK_ALIAS(DMA2_Stream1_IRQHandler);
WEAK_ALIAS(DMA2_Stream2_IRQHandler);
WEAK_ALIAS(DMA2_Stream3_IRQHandler);
WEAK_ALIAS(DMA2_Stream4_IRQHandler);
WEAK_ALIAS(OTG_FS_IRQHandler);
WEAK_ALIAS(DMA2_Stream5_IRQHandler);
WEAK_ALIAS(DMA2_Stream6_IRQHandler);
WEAK_ALIAS(DMA2_Stream7_IRQHandler);
WEAK_ALIAS(USART6_IRQHandler);
WEAK_ALIAS(I2C3_EV_IRQHandler);
WEAK_ALIAS(I2C3_ER_IRQHandler);
WEAK_ALIAS(FPU_IRQHandler);
WEAK_ALIAS(SPI4_IRQHandler);
WEAK_ALIAS(SPI5_IRQHandler);

/* -------------------------------------------------------------------------
 * Vector table -- must be placed at 0x08000000 (first word in Flash).
 * ---------------------------------------------------------------------- */
typedef void (*irq_handler_t)(void);

__attribute__((section(".isr_vector"), used))
static const irq_handler_t g_pfnVectors[] = {
    /* ---- Core exceptions (indices 0–15) ---- */
    (irq_handler_t)&_estack,       /*  0: Initial stack pointer       */
    Reset_Handler,                  /*  1: Reset                        */
    NMI_Handler,                    /*  2: NMI                          */
    HardFault_Handler,              /*  3: Hard fault                   */
    MemManage_Handler,              /*  4: Memory management            */
    BusFault_Handler,               /*  5: Bus fault                    */
    UsageFault_Handler,             /*  6: Usage fault                  */
    0, 0, 0, 0,                     /*  7–10: Reserved                  */
    SVC_Handler,                    /* 11: SVC                          */
    DebugMon_Handler,               /* 12: Debug monitor                */
    0,                              /* 13: Reserved                     */
    PendSV_Handler,                 /* 14: PendSV                       */
    SysTick_Handler,                /* 15: SysTick                      */
    /* ---- Device interrupts (IRQ0 – IRQ85) ---- */
    WWDG_IRQHandler,                /* IRQ0:  WWDG                      */
    PVD_IRQHandler,                 /* IRQ1:  PVD                       */
    TAMP_STAMP_IRQHandler,          /* IRQ2:  Tamper / time stamp        */
    RTC_WKUP_IRQHandler,            /* IRQ3:  RTC wakeup                */
    FLASH_IRQHandler,               /* IRQ4:  Flash                     */
    RCC_IRQHandler,                 /* IRQ5:  RCC                       */
    EXTI0_IRQHandler,               /* IRQ6:  EXTI line 0               */
    EXTI1_IRQHandler,               /* IRQ7:  EXTI line 1               */
    EXTI2_IRQHandler,               /* IRQ8:  EXTI line 2               */
    EXTI3_IRQHandler,               /* IRQ9:  EXTI line 3               */
    EXTI4_IRQHandler,               /* IRQ10: EXTI line 4               */
    DMA1_Stream0_IRQHandler,        /* IRQ11: DMA1 stream 0             */
    DMA1_Stream1_IRQHandler,        /* IRQ12: DMA1 stream 1             */
    DMA1_Stream2_IRQHandler,        /* IRQ13: DMA1 stream 2             */
    DMA1_Stream3_IRQHandler,        /* IRQ14: DMA1 stream 3             */
    DMA1_Stream4_IRQHandler,        /* IRQ15: DMA1 stream 4             */
    DMA1_Stream5_IRQHandler,        /* IRQ16: DMA1 stream 5             */
    DMA1_Stream6_IRQHandler,        /* IRQ17: DMA1 stream 6             */
    ADC_IRQHandler,                 /* IRQ18: ADC1/2/3                  */
    0, 0, 0, 0,                     /* IRQ19–22: Reserved (no CAN/F411) */
    EXTI9_5_IRQHandler,             /* IRQ23: EXTI lines 5–9            */
    TIM1_BRK_TIM9_IRQHandler,       /* IRQ24: TIM1 break / TIM9         */
    TIM1_UP_TIM10_IRQHandler,       /* IRQ25: TIM1 update / TIM10       */
    TIM1_TRG_COM_TIM11_IRQHandler,  /* IRQ26: TIM1 trigger / TIM11      */
    TIM1_CC_IRQHandler,             /* IRQ27: TIM1 capture/compare      */
    TIM2_IRQHandler,                /* IRQ28: TIM2                      */
    TIM3_IRQHandler,                /* IRQ29: TIM3                      */
    TIM4_IRQHandler,                /* IRQ30: TIM4                      */
    I2C1_EV_IRQHandler,             /* IRQ31: I2C1 event                */
    I2C1_ER_IRQHandler,             /* IRQ32: I2C1 error                */
    I2C2_EV_IRQHandler,             /* IRQ33: I2C2 event                */
    I2C2_ER_IRQHandler,             /* IRQ34: I2C2 error                */
    SPI1_IRQHandler,                /* IRQ35: SPI1                      */
    SPI2_IRQHandler,                /* IRQ36: SPI2                      */
    USART1_IRQHandler,              /* IRQ37: USART1                    */
    USART2_IRQHandler,              /* IRQ38: USART2                    */
    0,                              /* IRQ39: Reserved                  */
    EXTI15_10_IRQHandler,           /* IRQ40: EXTI lines 10–15          */
    RTC_Alarm_IRQHandler,           /* IRQ41: RTC alarms (A/B)          */
    OTG_FS_WKUP_IRQHandler,         /* IRQ42: USB FS wakeup             */
    0, 0, 0, 0,                     /* IRQ43–46: Reserved               */
    DMA1_Stream7_IRQHandler,        /* IRQ47: DMA1 stream 7             */
    0,                              /* IRQ48: Reserved                  */
    SDIO_IRQHandler,                /* IRQ49: SDIO                      */
    TIM5_IRQHandler,                /* IRQ50: TIM5                      */
    SPI3_IRQHandler,                /* IRQ51: SPI3                      */
    0, 0, 0, 0,                     /* IRQ52–55: Reserved               */
    DMA2_Stream0_IRQHandler,        /* IRQ56: DMA2 stream 0             */
    DMA2_Stream1_IRQHandler,        /* IRQ57: DMA2 stream 1             */
    DMA2_Stream2_IRQHandler,        /* IRQ58: DMA2 stream 2             */
    DMA2_Stream3_IRQHandler,        /* IRQ59: DMA2 stream 3             */
    DMA2_Stream4_IRQHandler,        /* IRQ60: DMA2 stream 4             */
    0, 0,                           /* IRQ61–62: Reserved               */
    OTG_FS_IRQHandler,              /* IRQ63: USB FS                    */
    DMA2_Stream5_IRQHandler,        /* IRQ64: DMA2 stream 5             */
    DMA2_Stream6_IRQHandler,        /* IRQ65: DMA2 stream 6             */
    DMA2_Stream7_IRQHandler,        /* IRQ66: DMA2 stream 7             */
    USART6_IRQHandler,              /* IRQ67: USART6                    */
    I2C3_EV_IRQHandler,             /* IRQ68: I2C3 event                */
    I2C3_ER_IRQHandler,             /* IRQ69: I2C3 error                */
    0, 0, 0, 0,                     /* IRQ70–73: Reserved (no OTG-HS)   */
    0, 0, 0, 0,                     /* IRQ74–77: Reserved               */
    0, 0,                           /* IRQ78–79: Reserved               */
    FPU_IRQHandler,                 /* IRQ80: FPU                       */
    0, 0,                           /* IRQ81–82: Reserved               */
    SPI4_IRQHandler,                /* IRQ83: SPI4                      */
    SPI5_IRQHandler,                /* IRQ84: SPI5                      */
};
