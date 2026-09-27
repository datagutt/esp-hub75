// SPDX-FileCopyrightText: 2025 Stuart Parmenter
// SPDX-License-Identifier: MIT
//
// @file gdma_dma.cpp
// @brief ESP32-S3 LCD_CAM + GDMA implementation for HUB75
//
// Uses direct LCD_CAM register access and manual GDMA setup.
// Simplified ring buffer approach with software BCM state tracking.

#include <sdkconfig.h>
#include <esp_idf_version.h>

// Only compile for ESP32-S3
#ifdef CONFIG_IDF_TARGET_ESP32S3

#include "gdma_dma.h"
#include "../../color/color_convert.h"    // For RGB565 scaling utilities
#include "../../panels/scan_patterns.h"   // For scan pattern remapping
#include "../../panels/panel_layout.h"    // For panel layout remapping
#include "../../util/drawing_profiler.h"  // For drawing profiling macros
#include <cassert>                        // NOLINT(readability-simplify-boolean-expr)
#include <cstring>
#include <algorithm>
#include <atomic>
#include <esp_log.h>
#include <esp_rom_gpio.h>
#include <esp_rom_sys.h>
#include <driver/gpio.h>
// gpio_func_sel() requires private GPIO header in ESP-IDF 5.4+
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
#include <esp_private/gpio.h>
#endif
#include <esp_private/gdma.h>
#include <soc/gpio_sig_map.h>
#include <soc/lcd_cam_struct.h>
#include <hal/gpio_hal.h>
#include <hal/gdma_ll.h>
// Header location changed in ESP-IDF 5.0
#if (ESP_IDF_VERSION_MAJOR >= 5)
#include <esp_private/periph_ctrl.h>
#else
#include <driver/periph_ctrl.h>
#endif
#include <esp_heap_caps.h>
#if HUB75_EXTERNAL_FRAMEBUFFERS
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 1, 0)
#include <esp_cache.h>
#else
#include <rom/cache.h>
#endif
#endif

static const char *const TAG = "GdmaDma";

namespace hub75 {

// HUB75 16-bit word layout for LCD_CAM peripheral
// Bit layout: [--|--|OE|LAT|ADDR(5-bit)|R2|G2|B2|R1|G1|B1]
enum HUB75WordBits : uint16_t {
  // RGB data bits
  R1_BIT = 0,  // Upper half red
  G1_BIT = 1,  // Upper half green
  B1_BIT = 2,  // Upper half blue
  R2_BIT = 3,  // Lower half red
  G2_BIT = 4,  // Lower half green
  B2_BIT = 5,  // Lower half blue
  // Bits 6-10: Row address (5-bit field, shifted << 6)
  LAT_BIT = 11,  // Latch signal
  OE_BIT = 12,   // Output Enable (active low)
  // Bits 13-15: Unused
};

// Address field (not individual bits)
constexpr int ADDR_SHIFT = 6;
constexpr uint16_t ADDR_MASK = 0x1F;  // 5-bit address (0-31)

// Combined RGB masks
constexpr uint16_t RGB_UPPER_MASK = (1 << R1_BIT) | (1 << G1_BIT) | (1 << B1_BIT);
constexpr uint16_t RGB_LOWER_MASK = (1 << R2_BIT) | (1 << G2_BIT) | (1 << B2_BIT);
constexpr uint16_t RGB_MASK = RGB_UPPER_MASK | RGB_LOWER_MASK;  // 0x003F

// Bit clear masks
constexpr uint16_t OE_CLEAR_MASK = ~(1 << OE_BIT);

#if HUB75_EXTERNAL_FRAMEBUFFERS
// PSRAM framebuffers start and end on this boundary so that cache write-backs never touch a
// neighbouring allocation. 64 bytes is the largest ESP32-S3 data cache line (16/32/64 are
// configurable) and the largest GDMA external memory block size.
constexpr size_t PSRAM_FB_ALIGNMENT = 64;

// Builds without this component's Kconfig (macro override) get the Kconfig defaults
#ifndef CONFIG_HUB75_EXTERNAL_FRAMEBUFFERS_MAX_PSRAM_MBPS
#define CONFIG_HUB75_EXTERNAL_FRAMEBUFFERS_MAX_PSRAM_MBPS 16
#endif
#ifndef CONFIG_HUB75_EXTERNAL_FRAMEBUFFERS_INTERNAL_PLANES
#define CONFIG_HUB75_EXTERNAL_FRAMEBUFFERS_INTERNAL_PLANES -1
#endif
#endif

GdmaDma::GdmaDma(const Hub75Config &config)
    : PlatformDma(config),
      plane_bytes_(0),
      split_plane_(0),
      row_stride_bytes_(0),
      total_buffer_bytes_(0),
      hot_buffer_bytes_(0),
      dma_chan_(nullptr),
      bit_depth_(resolve_bit_depth(config)),
      lsbMsbTransitionBit_(0),
      actual_clock_hz_(resolve_actual_clock_speed(config.output_clock_speed)),
      panel_width_(config.panel_width),
      panel_height_(config.panel_height),
      layout_rows_(config.layout_rows),
      layout_cols_(config.layout_cols),
      virtual_width_(config.panel_width * config.layout_cols),
      virtual_height_(config.panel_height * config.layout_rows),
      // Use helper function to compute DMA width (doubles for four-scan panels)
      dma_width_(
          get_effective_dma_width(config.scan_wiring, config.panel_width, config.layout_rows, config.layout_cols)),
      scan_wiring_(config.scan_wiring),
      layout_(config.layout),
      needs_scan_remap_(config.scan_wiring != Hub75ScanWiring::STANDARD_TWO_SCAN),
      needs_layout_remap_(config.layout != Hub75PanelLayout::HORIZONTAL),
      rotation_(config.rotation),
      // Use helper function to compute num_rows (halves for four-scan panels)
      num_rows_(get_effective_num_rows(config.scan_wiring, config.panel_height)),
      dma_buffers_{nullptr, nullptr},
      hot_buffers_{nullptr, nullptr},
      row_buffers_{nullptr, nullptr},
      buffer_in_psram_{false, false},
      descriptors_(nullptr),
      front_idx_(0),
      active_idx_(0),
#if HUB75_EXTERNAL_FRAMEBUFFERS
      dirty_row_begin_{UINT16_MAX, UINT16_MAX},
      dirty_row_end_{0, 0},
#endif
      descriptor_count_(0),
      basis_brightness_(config.brightness),  // Use config value (default: 128)
      intensity_(1.0f) {
  // Zero-copy architecture: DMA buffers ARE the display memory
  // Note: For four-scan panels, dma_width_ is doubled and num_rows_ is halved
  // to match the physical shift register layout
}

GdmaDma::~GdmaDma() { GdmaDma::shutdown(); }

bool GdmaDma::init() {
  ESP_LOGI(TAG, "Initializing LCD_CAM peripheral with GDMA...");
  ESP_LOGI(TAG, "Pin config: R1=%d G1=%d B1=%d R2=%d G2=%d B2=%d", config_.pins.r1, config_.pins.g1, config_.pins.b1,
           config_.pins.r2, config_.pins.g2, config_.pins.b2);
  ESP_LOGI(TAG, "Pin config: A=%d B=%d C=%d D=%d E=%d LAT=%d OE=%d CLK=%d", config_.pins.a, config_.pins.b,
           config_.pins.c, config_.pins.d, config_.pins.e, config_.pins.lat, config_.pins.oe, config_.pins.clk);

  // Enable and reset LCD_CAM peripheral
  periph_module_enable(PERIPH_LCD_CAM_MODULE);
  periph_module_reset(PERIPH_LCD_CAM_MODULE);

  // Reset LCD bus
  LCD_CAM.lcd_user.lcd_reset = 1;
  esp_rom_delay_us(1000);

  // Configure LCD clock
  configure_lcd_clock();

  // Configure LCD mode (i8080 16-bit parallel)
  configure_lcd_mode();

  // Configure GPIO routing
  configure_gpio();

  // Allocate GDMA channel
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
  // ESP-IDF 6.0+: simplified config, direction via NULL parameter
  gdma_channel_alloc_config_t dma_alloc_config = {.intr_priority = 0, .flags = {.isr_cache_safe = 0}};
  esp_err_t err = gdma_new_ahb_channel(&dma_alloc_config, &dma_chan_, nullptr);

#elif ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
  // ESP-IDF 5.4 - 5.x: gdma_new_ahb_channel (2-arg)
  gdma_channel_alloc_config_t dma_alloc_config = {.sibling_chan = nullptr,
                                                  .direction = GDMA_CHANNEL_DIRECTION_TX,
                                                  .flags = {.reserve_sibling = 0, .isr_cache_safe = 0}};
  esp_err_t err = gdma_new_ahb_channel(&dma_alloc_config, &dma_chan_);

#elif ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
  // ESP-IDF 5.0 - 5.3: gdma_new_channel with isr_cache_safe flag
  gdma_channel_alloc_config_t dma_alloc_config = {.sibling_chan = nullptr,
                                                  .direction = GDMA_CHANNEL_DIRECTION_TX,
                                                  .flags = {.reserve_sibling = 0, .isr_cache_safe = 0}};
  esp_err_t err = gdma_new_channel(&dma_alloc_config, &dma_chan_);

#else
  // ESP-IDF < 5.0: gdma_new_channel without isr_cache_safe
  gdma_channel_alloc_config_t dma_alloc_config = {
      .sibling_chan = nullptr, .direction = GDMA_CHANNEL_DIRECTION_TX, .flags = {.reserve_sibling = 0}};
  esp_err_t err = gdma_new_channel(&dma_alloc_config, &dma_chan_);
#endif
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to allocate GDMA channel: %s", esp_err_to_name(err));
    return false;
  }

  // Connect GDMA to LCD peripheral
  gdma_connect(dma_chan_, GDMA_MAKE_TRIGGER(GDMA_TRIG_PERIPH_LCD, 0));

  // Configure GDMA strategy
  // owner_check = false: Static descriptors, no dynamic ownership handshaking needed
  // auto_update_desc = false: No descriptor writeback - prevents corruption with infinite ring
  gdma_strategy_config_t strategy_config = {.owner_check = false,
                                            .auto_update_desc = false
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 0, 0)
                                            ,
                                            .eof_till_data_popped = false
#endif
  };
  gdma_apply_strategy(dma_chan_, &strategy_config);

  ESP_LOGI(TAG, "GDMA strategy configured: owner_check=false, auto_update_desc=false");

  // Wait for any pending LCD operations
  while (LCD_CAM.lcd_user.lcd_start)
    ;

  // Post-init cleanup for clean state
  gdma_reset(dma_chan_);
  esp_rom_delay_us(1000);
  LCD_CAM.lcd_user.lcd_dout = 1;         // Enable data out
  LCD_CAM.lcd_user.lcd_update = 1;       // Update registers
  LCD_CAM.lcd_misc.lcd_afifo_reset = 1;  // Reset LCD TX FIFO

  // Note: No EOF callback needed with descriptor-chain approach

  // Register frame callback if already set
  if (frame_callback_) {
    gdma_tx_event_callbacks_t cbs = {};
    cbs.on_trans_eof = GdmaDma::on_trans_eof;
    gdma_register_tx_event_callbacks(dma_chan_, &cbs, this);
  }
  // The descriptor chain encodes all timing via repetition counts

  ESP_LOGI(TAG, "GDMA EOF callback registered successfully");
  ESP_LOGI(TAG, "Panel config: %dx%d pixels, %dx%d layout, virtual: %dx%d", panel_width_, panel_height_, layout_cols_,
           layout_rows_, virtual_width_, virtual_height_);
  ESP_LOGI(TAG, "DMA config: %dx%d (width x rows), four-scan: %s", dma_width_, num_rows_,
           is_four_scan_wiring(scan_wiring_) ? "yes" : "no");

  ESP_LOGI(TAG, "LCD_CAM + GDMA initialized successfully");
  ESP_LOGI(TAG, "Clock: %.2f MHz (requested %u MHz)", actual_clock_hz_ / 1000000.0f,
           (unsigned int) (static_cast<uint32_t>(config_.output_clock_speed) / 1000000));

  // Calculate BCM timing (determines lsbMsbTransitionBit for OE control)
  calculate_bcm_timings();

  // Decide which bit planes go to PSRAM (may lower the clock and redo the BCM timing)
  plan_psram_layout();

  // Allocate per-row bit-plane buffers
  if (!allocate_row_buffers()) {
    return false;
  }

  // Burst and external memory access depend on where the buffers ended up
  if (!configure_dma_transfer()) {
    return false;
  }

  // Adjust LUT for BCM monotonicity (only needed when lsbMsbTransitionBit > 0)
  // With transition=0, BCM weights are always monotonically non-decreasing
#if HUB75_GAMMA_MODE == 1 || HUB75_GAMMA_MODE == 2
  if (lsbMsbTransitionBit_ > 0) {
    int adjusted = adjust_lut_for_bcm(lut_, bit_depth_, lsbMsbTransitionBit_);
    ESP_LOGI(TAG, "Adjusted %d LUT entries for BCM monotonicity (lsbMsbTransitionBit=%d)", adjusted,
             lsbMsbTransitionBit_);
  }
#endif

  // Validate brightness OE configuration safety margins
  if (!validate_brightness_config()) {
    return false;
  }

  // Initialize buffers with blank pixels (control bits only, RGB=0)
  initialize_blank_buffers();

  // Initialize brightness remapping coefficients (quadratic curve)
  init_brightness_coeffs(dma_width_, config_.latch_blanking);

  // Set OE bits for BCM control and brightness
  set_brightness_oe();

  // Build descriptor chain (one descriptor per bit plane)
  if (!build_descriptor_chain()) {
    return false;
  }

  ESP_LOGI(TAG, "Descriptor-chain DMA setup complete");
  return true;
}

HUB75_CONST uint32_t GdmaDma::resolve_actual_clock_speed(Hub75ClockSpeed clock_speed) const {
  // ESP32-S3 LCD_CAM clock derivation:
  //   Output = PLL_F160M / lcd_clkm_div_num
  //   Constraint: lcd_clkm_div_num >= 2
  //
  // We use integer dividers only - no fractional dividers. Fractional dividers
  // cause clock jitter because the hardware alternates between two integer
  // dividers to approximate the fractional value. With pure integer division
  // from the stable 160 MHz PLL, every clock cycle is identical.
  //
  // The resulting frequencies may not be round numbers (e.g., 160/7 = 22.86 MHz),
  // but this is fine - what matters for signal integrity is that each clock
  // period is exactly the same, not that the frequency is a nice decimal.
  //
  // Available speeds: 32 MHz (div=5), 26.67 MHz (div=6), 22.86 MHz (div=7),
  //                   20 MHz (div=8), 17.78 MHz (div=9), 16 MHz (div=10), ...
  uint32_t requested_hz = static_cast<uint32_t>(clock_speed);
  uint32_t divider = (160000000 + requested_hz / 2) / requested_hz;  // Round to nearest
  return 160000000 / std::max(divider, uint32_t{2});
}

void GdmaDma::configure_lcd_clock() {
  // Configure LCD clock from PLL_F160M (160 MHz)
  // actual_clock_hz_ already resolved in constructor
  uint32_t requested_hz = static_cast<uint32_t>(config_.output_clock_speed);
  uint32_t div_num = 160000000 / actual_clock_hz_;

  if (actual_clock_hz_ != requested_hz) {
    ESP_LOGI(TAG, "Clock speed %u Hz rounded to %u Hz (160MHz / %u)", (unsigned int) requested_hz,
             (unsigned int) actual_clock_hz_, (unsigned int) div_num);
  }

  LCD_CAM.lcd_clock.lcd_clk_sel = 3;      // PLL_F160M_CLK (value 3, not 2!)
  LCD_CAM.lcd_clock.lcd_ck_out_edge = 0;  // PCLK low in 1st half cycle
  LCD_CAM.lcd_clock.lcd_ck_idle_edge = config_.clk_phase_inverted ? 1 : 0;
  LCD_CAM.lcd_clock.lcd_clkcnt_n = 1;        // Should never be zero
  LCD_CAM.lcd_clock.lcd_clk_equ_sysclk = 1;  // PCLK = CLK / 1 (simple divisor)
  LCD_CAM.lcd_clock.lcd_clkm_div_num = div_num;
  LCD_CAM.lcd_clock.lcd_clkm_div_a = 1;  // Fractional divider (0/1)
  LCD_CAM.lcd_clock.lcd_clkm_div_b = 0;

  ESP_LOGI(TAG, "LCD clock: PLL_F160M / %u = %.2f MHz", (unsigned int) div_num, actual_clock_hz_ / 1000000.0f);
}

void GdmaDma::configure_lcd_mode() {
  // Configure LCD in i8080 mode, 16-bit parallel, continuous output
  LCD_CAM.lcd_ctrl.lcd_rgb_mode_en = 0;     // i8080 mode (not RGB)
  LCD_CAM.lcd_rgb_yuv.lcd_conv_bypass = 0;  // Disable RGB/YUV converter
  LCD_CAM.lcd_misc.lcd_next_frame_en = 0;   // Do NOT auto-frame
  LCD_CAM.lcd_misc.lcd_bk_en = 1;           // Enable blanking
  LCD_CAM.lcd_misc.lcd_vfk_cyclelen = 0;
  LCD_CAM.lcd_misc.lcd_vbk_cyclelen = 0;

  LCD_CAM.lcd_data_dout_mode.val = 0;      // No data delays
  LCD_CAM.lcd_user.lcd_always_out_en = 1;  // Enable 'always out' mode for arbitrary-length transfers
  LCD_CAM.lcd_user.lcd_8bits_order = 0;    // Do not swap bytes
  LCD_CAM.lcd_user.lcd_bit_order = 0;      // Do not reverse bit order
  LCD_CAM.lcd_user.lcd_2byte_en = 1;       // 16-bit mode
  LCD_CAM.lcd_user.lcd_dout = 1;           // Enable data output

  // CRITICAL: Dummy phases required for DMA to trigger reliably
  LCD_CAM.lcd_user.lcd_dummy = 1;           // Dummy phase(s) @ LCD start
  LCD_CAM.lcd_user.lcd_dummy_cyclelen = 1;  // 1+1 dummy phase
  LCD_CAM.lcd_user.lcd_cmd = 0;             // No command at LCD start

  // Disable start signal
  LCD_CAM.lcd_user.lcd_start = 0;
}

void GdmaDma::configure_gpio() {
  // 16-bit data pins mapping
  int data_pins[16] = {
      config_.pins.r1,   // D0
      config_.pins.g1,   // D1
      config_.pins.b1,   // D2
      config_.pins.r2,   // D3
      config_.pins.g2,   // D4
      config_.pins.b2,   // D5
      config_.pins.a,    // D6
      config_.pins.b,    // D7
      config_.pins.c,    // D8
      config_.pins.d,    // D9
      config_.pins.e,    // D10
      config_.pins.lat,  // D11
      config_.pins.oe,   // D12
      -1,
      -1,
      -1  // D13-D15 unused
  };

  gpio_drive_cap_t drive = (gpio_drive_cap_t) (config_.gpio_drive_strength > 3 ? 3 : config_.gpio_drive_strength);

  // Configure data pins
  for (int i = 0; i < 16; i++) {
    if (data_pins[i] >= 0) {
      esp_rom_gpio_connect_out_signal(data_pins[i], LCD_DATA_OUT0_IDX + i, false, false);
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
      gpio_func_sel((gpio_num_t) data_pins[i], PIN_FUNC_GPIO);
#else
      gpio_hal_iomux_func_sel(GPIO_PIN_MUX_REG[data_pins[i]], PIN_FUNC_GPIO);
#endif
      gpio_set_drive_capability((gpio_num_t) data_pins[i], drive);
    }
  }

  // Configure WR (clock) pin
  if (config_.pins.clk >= 0) {
    esp_rom_gpio_connect_out_signal(config_.pins.clk, LCD_PCLK_IDX, config_.clk_phase_inverted, false);
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
    gpio_func_sel((gpio_num_t) config_.pins.clk, PIN_FUNC_GPIO);
#else
    gpio_hal_iomux_func_sel(GPIO_PIN_MUX_REG[config_.pins.clk], PIN_FUNC_GPIO);
#endif
    gpio_set_drive_capability((gpio_num_t) config_.pins.clk, drive);
  }

  ESP_LOGD(TAG, "GPIO routing configured");
}

uint8_t *GdmaDma::allocate_framebuffer(size_t size, bool *in_psram) {
  *in_psram = false;
#if HUB75_EXTERNAL_FRAMEBUFFERS
  // Each bit plane is one descriptor, and its size must be a whole number of 16-byte blocks
  // (the smallest GDMA external memory block).
  if ((dma_width_ * sizeof(uint16_t)) % 16 != 0) {
    ESP_LOGW(TAG, "DMA width %u is not a multiple of 8 pixels, PSRAM framebuffers unavailable", dma_width_);
  } else {
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
    // The heap maps DMA | SPIRAM to PSRAM and adds any cache line or flash encryption alignment
    constexpr uint32_t caps = MALLOC_CAP_DMA | MALLOC_CAP_SPIRAM;
#else
    // Older heaps have no PSRAM region tagged DMA-capable
    constexpr uint32_t caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
#endif
    auto *buf = (uint8_t *) heap_caps_aligned_calloc(PSRAM_FB_ALIGNMENT, 1, size, caps);
    if (buf) {
      *in_psram = true;
      return buf;
    }
    ESP_LOGW(TAG, "PSRAM allocation of %zu bytes failed, falling back to internal RAM", size);
  }
#endif
  return (uint8_t *) heap_caps_calloc(1, size, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
}

bool GdmaDma::allocate_row_buffers() {
  plane_bytes_ = dma_width_ * sizeof(uint16_t);
  // Each bit plane is sent by one descriptor, whose length field is 12 bits wide
  if (plane_bytes_ > DMA_DESCRIPTOR_BUFFER_MAX_SIZE) {
    ESP_LOGE(TAG, "DMA width %u exceeds the single-descriptor bit plane limit", dma_width_);
    return false;
  }
  const size_t hot_row_stride = plane_bytes_ * (bit_depth_ - split_plane_);
  row_stride_bytes_ = plane_bytes_ * split_plane_;
  total_buffer_bytes_ = num_rows_ * row_stride_bytes_;
  hot_buffer_bytes_ = num_rows_ * hot_row_stride;
#if HUB75_EXTERNAL_FRAMEBUFFERS
  // Pad so the last cache line written back belongs to this buffer
  total_buffer_bytes_ = (total_buffer_bytes_ + PSRAM_FB_ALIGNMENT - 1) & ~(PSRAM_FB_ALIGNMENT - 1);
#endif

  const int buffer_count = config_.double_buffer ? 2 : 1;
  for (int i = 0; i < buffer_count; i++) {
    const char name = static_cast<char>('A' + i);
    bool ok = true;
    if (total_buffer_bytes_ > 0) {
      dma_buffers_[i] = allocate_framebuffer(total_buffer_bytes_, &buffer_in_psram_[i]);
      ok = dma_buffers_[i] != nullptr;
    }
    if (ok && hot_buffer_bytes_ > 0) {
      hot_buffers_[i] = (uint8_t *) heap_caps_calloc(1, hot_buffer_bytes_, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
      ok = hot_buffers_[i] != nullptr;
    }
    if (!ok) {
      ESP_LOGE(TAG, "Failed to allocate buffer %c (%zu + %zu bytes)", name, total_buffer_bytes_, hot_buffer_bytes_);
      heap_caps_free(dma_buffers_[i]);
      heap_caps_free(hot_buffers_[i]);
      dma_buffers_[i] = nullptr;
      hot_buffers_[i] = nullptr;
      buffer_in_psram_[i] = false;
      if (i == 0) {
        return false;
      }
      ESP_LOGW(TAG, "Continuing in single-buffer mode");
      break;
    }

    row_buffers_[i] = new RowBitPlaneBuffer[num_rows_];
    for (int row = 0; row < num_rows_; row++) {
      uint8_t *const low = dma_buffers_[i] ? dma_buffers_[i] + row * row_stride_bytes_ : nullptr;
      row_buffers_[i][row].low = low;
      row_buffers_[i][row].high = hot_buffers_[i] ? hot_buffers_[i] + row * hot_row_stride : low + row_stride_bytes_;
    }

    ESP_LOGI(TAG, "Buffer %c allocated: %d rows × %zu bytes/row = %zu total (%s)", name, num_rows_,
             row_stride_bytes_ + hot_row_stride, num_rows_ * (row_stride_bytes_ + hot_row_stride),
             hot_buffers_[i] ? (buffer_in_psram_[i] ? "low planes PSRAM, high planes internal RAM"
                                                    : "low and high planes internal RAM")
                             : (buffer_in_psram_[i] ? "PSRAM" : "internal RAM"));
  }

  // Single buffer: CPU and DMA share buffer 0. Double buffer: DMA shows 0, CPU draws into 1.
  front_idx_ = 0;
  active_idx_ = is_double_buffered() ? 1 : 0;

  return true;
}

uint8_t GdmaDma::transition_bit_for_clock(uint32_t clock_hz) const {
  // Smallest lsbMsbTransitionBit whose refresh rate meets min_refresh_rate (same search as
  // calculate_bcm_timings())
  const float buffer_time_us = (dma_width_ * 1000000.0f) / clock_hz;
  for (uint8_t tb = 0; tb < bit_depth_; tb++) {
    const float frame_us = calculate_bcm_transmissions(bit_depth_, tb) * buffer_time_us * num_rows_;
    if ((int) (1000000.0f / frame_us) >= (int) config_.min_refresh_rate) {
      return tb;
    }
  }
  return bit_depth_ - 1;
}

// PSRAM layout: the LCD streams output clock x 2 bytes/s with no slack, and each bit plane is
// read once per descriptor that repeats it. The high planes are read up to 2^(depth - tb - 2)
// times per row, the low ones once, so keeping the few most repeated planes in internal RAM
// removes most of the PSRAM traffic. PSRAM demand = clock x 2 x (reads of low planes / all
// reads). Starvation halts the LCD (seen during TLS handshakes at 32 MB/s), hence the limit.
void GdmaDma::plan_psram_layout() {
  split_plane_ = bit_depth_;
#if HUB75_EXTERNAL_FRAMEBUFFERS
  if ((dma_width_ * sizeof(uint16_t)) % 16 != 0) {
    return;  // allocate_framebuffer() keeps such buffers internal anyway
  }
  const uint64_t limit_bps = static_cast<uint64_t>(CONFIG_HUB75_EXTERNAL_FRAMEBUFFERS_MAX_PSRAM_MBPS) * 1000000;
  const int forced_internal = CONFIG_HUB75_EXTERNAL_FRAMEBUFFERS_INTERNAL_PLANES;

  auto reps = [](int bit, int tb) { return bit <= tb ? 1 : 1 << (bit - tb - 1); };
  auto demand_bps = [&](uint32_t clock_hz, int tb, int split) {
    int low_reads = 0;
    for (int bit = 0; bit < split; bit++) {
      low_reads += reps(bit, tb);
    }
    return static_cast<uint64_t>(clock_hz) * 2 * low_reads / calculate_bcm_transmissions(bit_depth_, tb);
  };

  // Lowest clock considered when the forced plane count cannot meet the limit
  constexpr uint32_t MIN_CLOCK_HZ = 8000000;
  uint32_t clock_hz = actual_clock_hz_;
  int tb = lsbMsbTransitionBit_;
  int split = bit_depth_;
  while (true) {
    if (forced_internal >= 0) {
      split = bit_depth_ - std::min<int>(forced_internal, bit_depth_);
    } else {
      // Fewest internal planes that meet the limit (all internal always does)
      for (split = bit_depth_; split > 0 && demand_bps(clock_hz, tb, split) > limit_bps; split--) {
      }
    }
    if (demand_bps(clock_hz, tb, split) <= limit_bps || clock_hz <= MIN_CLOCK_HZ) {
      break;
    }
    const uint32_t divider = 160000000 / clock_hz + 1;
    clock_hz = 160000000 / divider;
    tb = transition_bit_for_clock(clock_hz);
  }

  split_plane_ = split;
  const uint64_t demand = demand_bps(clock_hz, tb, split);
  const size_t internal_bytes = static_cast<size_t>(bit_depth_ - split) * dma_width_ * sizeof(uint16_t) * num_rows_ *
                                (config_.double_buffer ? 2 : 1);
  ESP_LOGI(TAG,
           "PSRAM layout: bit planes 0-%d in PSRAM, %d-%d in internal RAM (%zu bytes); PSRAM stream %.1f MB/s at "
           "%.2f MHz (limit %d MB/s)",
           split - 1, split, bit_depth_ - 1, internal_bytes, demand / 1000000.0f, clock_hz / 1000000.0f,
           CONFIG_HUB75_EXTERNAL_FRAMEBUFFERS_MAX_PSRAM_MBPS);
  if (demand > limit_bps) {
    ESP_LOGW(TAG, "PSRAM stream still above the limit; raise HUB75_EXTERNAL_FRAMEBUFFERS_INTERNAL_PLANES");
  }
  if (clock_hz != actual_clock_hz_) {
    ESP_LOGW(TAG, "Output clock lowered from %.2f MHz to %.2f MHz to limit the PSRAM stream",
             actual_clock_hz_ / 1000000.0f, clock_hz / 1000000.0f);
    actual_clock_hz_ = clock_hz;
    configure_lcd_clock();
    calculate_bcm_timings();
  }
#endif
}

bool GdmaDma::configure_dma_transfer() {
  const bool access_ext_mem = buffer_in_psram_[0] || buffer_in_psram_[1];
  const size_t bytes_per_bitplane = dma_width_ * sizeof(uint16_t);

  // Internal SRAM keeps the 32-byte burst. For PSRAM use the largest external memory block
  // size (16/32/64, capped at 64 by the PSRAM controller) that divides a bit plane, so every
  // descriptor starts and ends on a block boundary.
  size_t burst = 32;
  if (access_ext_mem) {
    burst = (bytes_per_bitplane % 64 == 0) ? 64 : (bytes_per_bitplane % 32 == 0) ? 32 : 16;
  }

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 3, 0)
  gdma_transfer_config_t transfer_config = {};
  transfer_config.max_data_burst_size = burst;
  transfer_config.access_ext_mem = access_ext_mem;
  esp_err_t err = gdma_config_transfer(dma_chan_, &transfer_config);
#else
  gdma_transfer_ability_t ability = {};
  ability.sram_trans_align = 32;
  ability.psram_trans_align = access_ext_mem ? burst : 0;
  esp_err_t err = gdma_set_transfer_ability(dma_chan_, &ability);
#endif
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to configure GDMA transfer: %s", esp_err_to_name(err));
    return false;
  }

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 3, 0)
  // Descriptors point at bit planes, so each bit plane start and size must meet the channel's
  // alignment for the memory it lives in.
  size_t int_align = 1, ext_align = 1;
  gdma_get_alignment_constraints(dma_chan_, &int_align, &ext_align);
  for (int i = 0; i < 2; i++) {
    const uint8_t *const bases[2] = {dma_buffers_[i], hot_buffers_[i]};
    for (int b = 0; b < 2; b++) {
      if (!bases[b]) {
        continue;
      }
      const size_t align = (b == 0 && buffer_in_psram_[i]) ? ext_align : int_align;
      if (((uintptr_t) bases[b] % align) != 0 || (bytes_per_bitplane % align) != 0) {
        ESP_LOGE(TAG, "Buffer %c at %p does not meet the GDMA alignment of %zu bytes", 'A' + i, bases[b], align);
        return false;
      }
    }
  }
#endif

  ESP_LOGI(TAG, "GDMA transfer: burst %zu bytes, external memory access %s", burst, access_ext_mem ? "on" : "off");
  return true;
}

void GdmaDma::start_transfer() {
  if (!dma_chan_ || !descriptors_) {
    ESP_LOGE(TAG, "DMA channel or descriptors not initialized");
    return;
  }

  ESP_LOGI(TAG, "Starting descriptor-chain DMA:");
  ESP_LOGI(TAG, "  Descriptor count: %zu", descriptor_count_);
  ESP_LOGI(TAG, "  Rows: %d, Bits: %d", num_rows_, bit_depth_);

  // Prime LCD registers
  LCD_CAM.lcd_user.lcd_update = 1;
  esp_rom_delay_us(10);

  // Start GDMA transfer from first descriptor in chain (front buffer)
  gdma_start(dma_chan_, (intptr_t) &descriptors_[0]);

  // Delay before starting LCD
  esp_rom_delay_us(100);

  // Start LCD engine (will run continuously via descriptor loop)
  LCD_CAM.lcd_user.lcd_start = 1;

  ESP_LOGI(TAG, "Descriptor-chain DMA transfer started - running continuously");
  log_dma_health();
  transfer_running_ = true;
  start_stall_watchdog();
}

void GdmaDma::stop_transfer() {
  transfer_running_ = false;
  stop_stall_watchdog();
  if (!dma_chan_) {
    return;
  }

  // Disable LCD output
  LCD_CAM.lcd_user.lcd_start = 0;
  LCD_CAM.lcd_user.lcd_update = 1;  // Apply the stop command

  gdma_stop(dma_chan_);

  ESP_LOGI(TAG, "DMA transfer stopped");
}

void GdmaDma::set_frame_callback(Hub75FrameCallback callback, void *arg) {
  PlatformDma::set_frame_callback(callback, arg);

  if (dma_chan_ && callback) {
    gdma_tx_event_callbacks_t cbs = {};
    cbs.on_trans_eof = GdmaDma::on_trans_eof;
    gdma_register_tx_event_callbacks(dma_chan_, &cbs, this);
  }
  // When callback is null: internal pointer is cleared by base class.
  // ISR checks frame_callback_ for null, so it's safe.
  // Full cleanup happens in shutdown() -> gdma_del_channel().
}

bool IRAM_ATTR GdmaDma::on_trans_eof(gdma_channel_handle_t dma_chan, gdma_event_data_t *event_data, void *user_data) {
  auto *dma = static_cast<GdmaDma *>(user_data);
  if (dma && dma->frame_callback_) {
    return dma->frame_callback_(dma->frame_callback_arg_);
  }
  return false;
}

// No EOF callback needed - descriptor chain handles all timing

void GdmaDma::shutdown() {
  GdmaDma::stop_transfer();

  if (dma_chan_) {
    gdma_disconnect(dma_chan_);
    gdma_del_channel(dma_chan_);
    dma_chan_ = nullptr;

    periph_module_disable(PERIPH_LCD_CAM_MODULE);
  }

  if (descriptors_) {
    heap_caps_free(descriptors_);
    descriptors_ = nullptr;
  }

  // Free all allocated resources (using array structure)
  for (int i = 0; i < 2; i++) {
    // Free raw DMA buffers
    if (dma_buffers_[i]) {
      heap_caps_free(dma_buffers_[i]);
      dma_buffers_[i] = nullptr;
    }
    if (hot_buffers_[i]) {
      heap_caps_free(hot_buffers_[i]);
      hot_buffers_[i] = nullptr;
    }

    // Free metadata arrays
    if (row_buffers_[i]) {
      delete[] row_buffers_[i];
      row_buffers_[i] = nullptr;
    }

    buffer_in_psram_[i] = false;
#if HUB75_EXTERNAL_FRAMEBUFFERS
    dirty_row_begin_[i] = UINT16_MAX;
    dirty_row_end_[i] = 0;
#endif
  }

  descriptor_count_ = 0;

  ESP_LOGI(TAG, "Shutdown complete");
}

// ============================================================================
// Brightness Control (Override Base Class)
// ============================================================================

void GdmaDma::set_basis_brightness(uint8_t brightness) {
  basis_brightness_ = brightness;

  if (brightness == 0) {
    ESP_LOGI(TAG, "Brightness set to 0 (display off)");
  } else {
    ESP_LOGD(TAG, "Basis brightness set to %u", (unsigned) brightness);
  }

  // Apply brightness change immediately by updating OE bits in DMA buffers
  set_brightness_oe();
}

void GdmaDma::set_intensity(float intensity) {
  // Clamp to valid range (0.0-1.0)
  if (intensity < 0.0f) {
    intensity = 0.0f;
  } else if (intensity > 1.0f) {
    intensity = 1.0f;
  }

  intensity_ = intensity;

  ESP_LOGD(TAG, "Intensity set to %.2f", intensity);

  // Apply intensity change immediately by updating OE bits in DMA buffers
  set_brightness_oe();
}

void GdmaDma::set_rotation(Hub75Rotation rotation) { rotation_ = rotation; }

// ============================================================================
// Pixel API (Direct DMA Buffer Writes)
// ============================================================================

HUB75_IRAM void GdmaDma::draw_pixels(uint16_t x, uint16_t y, uint16_t w, uint16_t h, const uint8_t *buffer,
                                     Hub75PixelFormat format, Hub75ColorOrder color_order, bool big_endian) {
  // Always write to active buffer (CPU drawing buffer)
  RowBitPlaneBuffer *target_buffers = row_buffers_[active_idx_];

  if (!target_buffers || !buffer) [[unlikely]] {
    return;
  }

  // Calculate rotated dimensions (user-facing coordinates)
  const uint16_t rotated_width = RotationTransform::get_rotated_width(virtual_width_, virtual_height_, rotation_);
  const uint16_t rotated_height = RotationTransform::get_rotated_height(virtual_width_, virtual_height_, rotation_);

  // Bounds check against rotated (user-facing) display size
  if (x >= rotated_width || y >= rotated_height) [[unlikely]] {
    return;
  }

  // Clip to display bounds
  if (x + w > rotated_width) [[unlikely]] {
    w = rotated_width - x;
  }
  if (y + h > rotated_height) [[unlikely]] {
    h = rotated_height - y;
  }

  // Pre-compute pixel stride for pointer arithmetic (avoids multiply per pixel)
  const size_t pixel_stride = (format == Hub75PixelFormat::RGB888)   ? 3
                              : (format == Hub75PixelFormat::RGB565) ? 2
                                                                     : /* RGB888_32 */ 4;

  // Check if we can use identity fast path (no coordinate transforms needed)
  const bool identity_transform = (rotation_ == Hub75Rotation::ROTATE_0) && !needs_layout_remap_ && !needs_scan_remap_;

  // Fused row-pair path: when an identity blit spans the full panel height,
  // source pixels (px, py) and (px, py + num_rows_) land in the SAME DMA word
  // (upper and lower RGB bits), so both halves are merged with a single
  // read-modify-write per bit plane instead of two.
  if (identity_transform && y == 0 && h == virtual_height_ && virtual_height_ == 2 * num_rows_) [[likely]] {
    const uint8_t *upper_ptr = buffer;
    const uint8_t *lower_ptr = buffer + static_cast<size_t>(num_rows_) * w * pixel_stride;

    // Pair cache: frames are dominated by flat color runs, so the merged bit
    // patterns are reused while both halves keep the same raw value.
    uint32_t cached_upper_raw = 0, cached_lower_raw = 0;
    uint16_t pair_patterns[HUB75_MAX_BIT_DEPTH];
    bool pair_cache_valid = false;

    for (uint16_t row = 0; row < num_rows_; row++) {
      const RowBitPlaneBuffer &row_buf = target_buffers[row];
      for (uint16_t dx = 0; dx < w; dx++) {
        const uint16_t px = x + dx;

        uint32_t upper_raw, lower_raw;
        if (format == Hub75PixelFormat::RGB565) {
          upper_raw = *reinterpret_cast<const uint16_t *>(upper_ptr);
          lower_raw = *reinterpret_cast<const uint16_t *>(lower_ptr);
        } else if (format == Hub75PixelFormat::RGB888) {
          upper_raw = upper_ptr[0] | (upper_ptr[1] << 8) | (upper_ptr[2] << 16);
          lower_raw = lower_ptr[0] | (lower_ptr[1] << 8) | (lower_ptr[2] << 16);
        } else {
          upper_raw = *reinterpret_cast<const uint32_t *>(upper_ptr);
          lower_raw = *reinterpret_cast<const uint32_t *>(lower_ptr);
        }

        if (!pair_cache_valid || upper_raw != cached_upper_raw || lower_raw != cached_lower_raw) {
          uint8_t ur = 0, ug = 0, ub = 0, lr = 0, lg = 0, lb = 0;
          extract_rgb888_from_format(upper_ptr, 0, format, color_order, big_endian, ur, ug, ub);
          extract_rgb888_from_format(lower_ptr, 0, format, color_order, big_endian, lr, lg, lb);

          const uint16_t ur_c = lut_[ur], ug_c = lut_[ug], ub_c = lut_[ub];
          const uint16_t lr_c = lut_[lr], lg_c = lut_[lg], lb_c = lut_[lb];

          for (int bit = 0; bit < bit_depth_; bit++) {
            pair_patterns[bit] = (((ur_c >> bit) & 1) << R1_BIT) | (((ug_c >> bit) & 1) << G1_BIT) |
                                 (((ub_c >> bit) & 1) << B1_BIT) | (((lr_c >> bit) & 1) << R2_BIT) |
                                 (((lg_c >> bit) & 1) << G2_BIT) | (((lb_c >> bit) & 1) << B2_BIT);
          }
          cached_upper_raw = upper_raw;
          cached_lower_raw = lower_raw;
          pair_cache_valid = true;
        }
        upper_ptr += pixel_stride;
        lower_ptr += pixel_stride;

        for_each_plane(
            row_buf, [&](int bit, uint16_t *buf) { buf[px] = (buf[px] & (uint16_t) ~RGB_MASK) | pair_patterns[bit]; });
      }
    }
    mark_all_rows_dirty(active_idx_);
    if (!is_double_buffered()) {
      sync_dirty_rows(active_idx_);
    }
    return;
  }

  // Rows touched by the transformed path, for the PSRAM cache write-back
  uint16_t touched_row_begin = UINT16_MAX, touched_row_end = 0;

  // Process each pixel
  const uint8_t *pixel_ptr = buffer;
  for (uint16_t dy = 0; dy < h; dy++) {
    const uint16_t py = y + dy;

    // Hoist row/is_lower computation for identity transform (constant per row)
    uint16_t id_row = 0;
    bool id_is_lower = false;
    const RowBitPlaneBuffer *id_row_buf = nullptr;
    uint16_t id_clear_mask = 0;
    const uint16_t *id_patterns = nullptr;
    if (identity_transform) {
      if (py < num_rows_) {
        id_row = py;
        id_is_lower = false;
      } else {
        id_row = py - num_rows_;
        id_is_lower = true;
      }
      id_row_buf = &target_buffers[id_row];
      id_clear_mask = id_is_lower ? ~RGB_LOWER_MASK : ~RGB_UPPER_MASK;
    }

    for (uint16_t dx = 0; dx < w; dx++) {
      uint16_t px = x + dx;
      uint16_t row;
      const RowBitPlaneBuffer *row_buf;
      uint16_t clear_mask;

      HUB75_PROFILE_BEGIN();

      // Fast path: identity transform — row-level values already computed
      if (identity_transform) {
        row = id_row;
        row_buf = id_row_buf;
        clear_mask = id_clear_mask;
      } else {
        // Full coordinate transformation pipeline
        auto transformed = transform_coordinate(px, py, rotation_, needs_layout_remap_, needs_scan_remap_, layout_,
                                                scan_wiring_, panel_width_, panel_height_, layout_rows_, layout_cols_,
                                                virtual_width_, virtual_height_, dma_width_, num_rows_);
        px = transformed.x;
        row = transformed.row;
        row_buf = &target_buffers[row];
        clear_mask = transformed.is_lower ? ~RGB_LOWER_MASK : ~RGB_UPPER_MASK;
        touched_row_begin = std::min(touched_row_begin, row);
        touched_row_end = std::max(touched_row_end, static_cast<uint16_t>(row + 1));
      }

      HUB75_PROFILE_STAGE(PROFILE_TRANSFORM);

      // Pack raw pixel bytes into uint32_t for fast comparison
      uint32_t current_raw;
      if (format == Hub75PixelFormat::RGB565) {
        current_raw = *reinterpret_cast<const uint16_t *>(pixel_ptr);
      } else if (format == Hub75PixelFormat::RGB888) {
        current_raw = pixel_ptr[0] | (pixel_ptr[1] << 8) | (pixel_ptr[2] << 16);
      } else {
        current_raw = *reinterpret_cast<const uint32_t *>(pixel_ptr);
      }

      // Only recompute patterns if raw pixel changed (skips extraction, LUT, and pattern computation)
      if (current_raw != cached_raw_pixel_) {
        // Extract RGB888 from pixel format (always_inline will inline the switch)
        uint8_t r8 = 0, g8 = 0, b8 = 0;
        extract_rgb888_from_format(pixel_ptr, 0, format, color_order, big_endian, r8, g8, b8);

        HUB75_PROFILE_STAGE(PROFILE_EXTRACT);

        // Apply LUT correction
        const uint16_t r_corrected = lut_[r8];
        const uint16_t g_corrected = lut_[g8];
        const uint16_t b_corrected = lut_[b8];

        HUB75_PROFILE_STAGE(PROFILE_LUT);

        // Pre-compute bit patterns for all bit planes using branchless bit extraction
        for (int bit = 0; bit < bit_depth_; bit++) {
          const uint16_t r_bit = (r_corrected >> bit) & 1;
          const uint16_t g_bit = (g_corrected >> bit) & 1;
          const uint16_t b_bit = (b_corrected >> bit) & 1;
          cached_upper_patterns_[bit] = (r_bit << R1_BIT) | (g_bit << G1_BIT) | (b_bit << B1_BIT);
          cached_lower_patterns_[bit] = (r_bit << R2_BIT) | (g_bit << G2_BIT) | (b_bit << B2_BIT);
        }
        cached_raw_pixel_ = current_raw;
      }
      pixel_ptr += pixel_stride;

      // Select upper or lower cached patterns based on half
      id_patterns = (clear_mask == (uint16_t) ~RGB_LOWER_MASK) ? cached_lower_patterns_ : cached_upper_patterns_;

      // Apply cached patterns to all bit planes (branch-free inner loop)
      for_each_plane(*row_buf, [&](int bit, uint16_t *buf) { buf[px] = (buf[px] & clear_mask) | id_patterns[bit]; });

      HUB75_PROFILE_STAGE(PROFILE_BITPLANE);
      HUB75_PROFILE_PIXEL();
    }
  }
  // Marked after the writes, so a concurrent full write-back cannot drop them
  if (identity_transform) {
    mark_y_span_dirty(active_idx_, y, h);
  } else {
    mark_rows_dirty(active_idx_, touched_row_begin, touched_row_end);
  }
  if (!is_double_buffered()) {
    sync_dirty_rows(active_idx_);
  }
}

void GdmaDma::clear() {
  // Always write to active buffer (CPU drawing buffer)
  RowBitPlaneBuffer *target_buffers = row_buffers_[active_idx_];

  if (!target_buffers) {
    return;
  }

  // Clear RGB bits in all buffers (keep control bits)
  for (int row = 0; row < num_rows_; row++) {
    for_each_plane(target_buffers[row], [&](int, uint16_t *buf) {
      for (uint16_t x = 0; x < dma_width_; x++) {
        // Clear RGB bits but preserve row address, LAT, OE
        buf[x] &= ~RGB_MASK;
      }
    });
  }
  mark_all_rows_dirty(active_idx_);
  if (!is_double_buffered()) {
    sync_dirty_rows(active_idx_);
  }
}

HUB75_IRAM void GdmaDma::fill(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint8_t r, uint8_t g, uint8_t b) {
  // Always write to active buffer (CPU drawing buffer)
  RowBitPlaneBuffer *target_buffers = row_buffers_[active_idx_];

  if (!target_buffers) [[unlikely]] {
    return;
  }

  // Calculate rotated dimensions (user-facing coordinates)
  const uint16_t rotated_width = RotationTransform::get_rotated_width(virtual_width_, virtual_height_, rotation_);
  const uint16_t rotated_height = RotationTransform::get_rotated_height(virtual_width_, virtual_height_, rotation_);

  // Bounds check against rotated (user-facing) display size
  if (x >= rotated_width || y >= rotated_height) [[unlikely]] {
    return;
  }

  // Clip to display bounds
  if (x + w > rotated_width) [[unlikely]] {
    w = rotated_width - x;
  }
  if (y + h > rotated_height) [[unlikely]] {
    h = rotated_height - y;
  }

  // Pre-compute LUT-corrected color values (ONCE for entire fill)
  const uint16_t r_corrected = lut_[r];
  const uint16_t g_corrected = lut_[g];
  const uint16_t b_corrected = lut_[b];

  // Pre-compute bit patterns for all bit planes (ONCE for entire fill)
  // This eliminates per-pixel bit extraction and conditional logic
  uint16_t upper_patterns[HUB75_MAX_BIT_DEPTH];
  uint16_t lower_patterns[HUB75_MAX_BIT_DEPTH];
  for (int bit = 0; bit < bit_depth_; bit++) {
    const uint16_t mask = (1 << bit);
    upper_patterns[bit] = ((r_corrected & mask) ? (1 << R1_BIT) : 0) | ((g_corrected & mask) ? (1 << G1_BIT) : 0) |
                          ((b_corrected & mask) ? (1 << B1_BIT) : 0);
    lower_patterns[bit] = ((r_corrected & mask) ? (1 << R2_BIT) : 0) | ((g_corrected & mask) ? (1 << G2_BIT) : 0) |
                          ((b_corrected & mask) ? (1 << B2_BIT) : 0);
  }

  // Pre-compute values for inner loop
  const bool identity_transform = (rotation_ == Hub75Rotation::ROTATE_0) && !needs_layout_remap_ && !needs_scan_remap_;

  // Rows touched by the transformed path, for the PSRAM cache write-back
  uint16_t touched_row_begin = UINT16_MAX, touched_row_end = 0;

  // Fill loop
  for (uint16_t dy = 0; dy < h; dy++) {
    const uint16_t py = y + dy;

    // Hoist row-level values for identity transform (constant per row)
    uint16_t id_row = 0;
    const RowBitPlaneBuffer *id_row_buf = nullptr;
    uint16_t id_clear_mask = 0;
    const uint16_t *id_patterns = nullptr;
    if (identity_transform) {
      bool id_is_lower;
      if (py < num_rows_) {
        id_row = py;
        id_is_lower = false;
      } else {
        id_row = py - num_rows_;
        id_is_lower = true;
      }
      id_row_buf = &target_buffers[id_row];
      id_clear_mask = id_is_lower ? ~RGB_LOWER_MASK : ~RGB_UPPER_MASK;
      id_patterns = id_is_lower ? lower_patterns : upper_patterns;
    }

    for (uint16_t dx = 0; dx < w; dx++) {
      uint16_t px = x + dx;
      const RowBitPlaneBuffer *row_buf;
      uint16_t clear_mask;
      const uint16_t *patterns;

      // Fast path: identity transform — row-level values already computed
      if (identity_transform) {
        row_buf = id_row_buf;
        clear_mask = id_clear_mask;
        patterns = id_patterns;
      } else {
        // Full coordinate transformation pipeline
        auto transformed = transform_coordinate(px, py, rotation_, needs_layout_remap_, needs_scan_remap_, layout_,
                                                scan_wiring_, panel_width_, panel_height_, layout_rows_, layout_cols_,
                                                virtual_width_, virtual_height_, dma_width_, num_rows_);
        px = transformed.x;
        row_buf = &target_buffers[transformed.row];
        clear_mask = transformed.is_lower ? ~RGB_LOWER_MASK : ~RGB_UPPER_MASK;
        patterns = transformed.is_lower ? lower_patterns : upper_patterns;
        touched_row_begin = std::min(touched_row_begin, static_cast<uint16_t>(transformed.row));
        touched_row_end = std::max(touched_row_end, static_cast<uint16_t>(transformed.row + 1));
      }

      // Update all bit planes (branch-free inner loop)
      for_each_plane(*row_buf, [&](int bit, uint16_t *buf) { buf[px] = (buf[px] & clear_mask) | patterns[bit]; });
    }
  }
  // Marked after the writes, so a concurrent full write-back cannot drop them
  if (identity_transform) {
    mark_y_span_dirty(active_idx_, y, h);
  } else {
    mark_rows_dirty(active_idx_, touched_row_begin, touched_row_end);
  }
  if (!is_double_buffered()) {
    sync_dirty_rows(active_idx_);
  }
}

void GdmaDma::flip_buffer() {
  // Single buffer mode: no-op (both indices point to buffer 0)
  if (!is_double_buffered() || !descriptors_) {
    return;
  }

  // Make the whole back buffer visible to the DMA before it is displayed. Writing back every
  // row rather than only the tracked ones costs a fraction of a millisecond per flip (clean
  // lines are skipped) and cannot miss a write.
#if HUB75_EXTERNAL_FRAMEBUFFERS
  mark_all_rows_dirty(active_idx_);
  const int64_t sync_start_us = esp_timer_get_time();
#endif
  sync_dirty_rows(active_idx_);
#if HUB75_EXTERNAL_FRAMEBUFFERS
  if (!flip_sync_logged_ && buffer_in_psram_[active_idx_]) {
    flip_sync_logged_ = true;
    ESP_LOGI(TAG, "First flip: PSRAM write-back of %zu bytes took %lld us", total_buffer_bytes_,
             (long long) (esp_timer_get_time() - sync_start_us));
  }
#endif

  // Retarget the running chain to the buffer just drawn (no stop/start).
  //
  // Both buffers share one layout, so every descriptor moves by the offset between the two
  // low-plane allocations or the two high-plane allocations, whichever it points into. The rewrite
  // runs in chain order and takes a few tens of microseconds for thousands of descriptors,
  // while the DMA spends several microseconds on each one, so the CPU overtakes the DMA
  // almost immediately: descriptors behind the DMA show the new frame from the next refresh,
  // the ones ahead of it already in this refresh. At worst one refresh is split between the
  // old and the new frame at the row the DMA was on. Flipping right after the frame callback
  // (EOF on the last row by default) puts that split at the top, so in practice it vanishes.
  //
  // The old front buffer becomes the draw buffer on return. The GDMA may still be reading the
  // descriptor it had loaded (plus the one it prefetched), so the first writes can reach at
  // most two bit plane transfers of the outgoing frame.
  const uintptr_t low_front = (uintptr_t) dma_buffers_[front_idx_];
  const uintptr_t low_delta = (uintptr_t) dma_buffers_[active_idx_] - low_front;
  const uintptr_t high_delta = (uintptr_t) hot_buffers_[active_idx_] - (uintptr_t) hot_buffers_[front_idx_];
  for (size_t i = 0; i < descriptor_count_; i++) {
    const uintptr_t addr = (uintptr_t) descriptors_[i].buffer;
    const uintptr_t delta = (addr - low_front < total_buffer_bytes_) ? low_delta : high_delta;
    descriptors_[i].buffer = (void *) (addr + delta);
  }
  // Descriptor stores must land before the caller starts drawing into the old front buffer
  std::atomic_thread_fence(std::memory_order_seq_cst);

  std::swap(front_idx_, active_idx_);
}

// ============================================================================
// Buffer Initialization
// ============================================================================

bool GdmaDma::validate_brightness_config() {
  const uint8_t latch_blanking = config_.latch_blanking;

  // Edge Case 1: Latch blanking must be less than DMA buffer width
  // Prevents underflow in: max_pixels = (dma_width_ - latch_blanking) >> rightshift
  if (latch_blanking >= dma_width_) {
    ESP_LOGE(TAG, "Invalid config: latch_blanking (%u) >= dma_width (%u)", latch_blanking, dma_width_);
    return false;
  }

  // Edge Case 2: DMA buffer width must be large enough for reasonable operation
  // Need space for: data pixels + latch blanking + safety margins
  if (dma_width_ < 8) {
    ESP_LOGE(TAG, "Invalid config: dma_width (%u) too small (minimum 8)", dma_width_);
    return false;
  }

  // Edge Case 3: Verify sufficient headroom for safety margin
  // We need max_pixels >= 2 (at least 1 for display + 1 for safety margin to prevent ghosting)
  // Since we use uniform max_pixels for all bit planes (BCM comes from descriptor repetition),
  // we only need to check once.
  const int max_pixels = dma_width_ - latch_blanking;
  if (max_pixels < 2) {
    ESP_LOGE(TAG,
             "Invalid config: max_pixels=%d (need >=2). "
             "Increase dma_width or decrease latch_blanking.",
             max_pixels);
    ESP_LOGE(TAG, "  Config: dma_width=%u, latch_blanking=%u", dma_width_, latch_blanking);
    return false;
  }

  ESP_LOGI(TAG,
           "Brightness configuration validated: dma_width=%u, latch_blanking=%u, all bit planes have sufficient "
           "headroom",
           dma_width_, latch_blanking);
  return true;
}

void GdmaDma::initialize_buffer_internal(RowBitPlaneBuffer *buffers) {
  if (!buffers) {
    return;
  }

  for (int row = 0; row < num_rows_; row++) {
    uint16_t row_addr = row & ADDR_MASK;

    for (int bit = 0; bit < bit_depth_; bit++) {
      uint16_t *buf = plane_ptr(buffers[row], bit);

      // Row address handling: LSB bit plane uses previous row for LAT settling
      //
      // HUB75 panels need time to process the LAT (latch) signal before the row
      // address changes. LAT transfers data from shift registers to the display
      // buffer. If the address changes too quickly, the panel may latch the
      // previous row's data into the current row's buffer.
      //
      // To provide settling time, bit plane 0 (LSB) is marked with the previous
      // row's address, creating a transition period:
      //
      //   Row N completes → Row N+1 bit 0 transmits (address still = N)
      //                  → Panel finishes latching Row N
      //                  → Row N+1 bit 1-7 transmit (address = N+1)
      //
      // This ensures the panel completes Row N's latch operation before it sees
      // the new address in bit planes 1-7.
      //
      // CRITICAL: Row 0 bit 0 WRAPS AROUND to use last row's address (row 31).
      // This prevents corruption when transitioning from row 31 (last) to row 0 (first).
      // Without wrap-around, the address would change from 31→0 during row 31's LAT settling,
      // causing ghosting on row 0.
      uint16_t addr_for_buffer;
      if (bit == 0) {
        // LSB bit plane uses previous row (wraps row 0 to last row)
        addr_for_buffer = ((row == 0 ? num_rows_ : row) - 1) & ADDR_MASK;
        ESP_LOGD(TAG, "Row %d Bit 0: Using previous row address 0x%02X (current: 0x%02X)", row, addr_for_buffer,
                 row_addr);
      } else {
        // All other bit planes use current row
        addr_for_buffer = row_addr;
      }

      // Fill all pixels with control bits (RGB=0, row address, OE=HIGH)
      for (uint16_t x = 0; x < dma_width_; x++) {
        buf[x] = (addr_for_buffer << ADDR_SHIFT) | (1 << OE_BIT);
      }

      // Set LAT bit on last pixel
      buf[dma_width_ - 1] |= (1 << LAT_BIT);
    }
  }
}

void GdmaDma::initialize_blank_buffers() {
  if (!row_buffers_[0]) {
    ESP_LOGE(TAG, "Row buffers not allocated");
    return;
  }

  ESP_LOGI(TAG, "Initializing blank DMA buffers with control bits...");
  for (int i = 0; i < 2; i++) {
    if (row_buffers_[i]) {
      initialize_buffer_internal(row_buffers_[i]);
      mark_all_rows_dirty(i);
      verify_psram_writeback(i, sync_dirty_rows(i));
    }
  }
  ESP_LOGI(TAG, "Blank buffers initialized");
}

// ============================================================================
// BCM Control via OE Bit Manipulation
// ============================================================================

// Configure OE (Output Enable) bits in DMA buffers to control brightness.
//
// Architecture: BCM timing vs OE duty cycle
// ------------------------------------------
// Binary Code Modulation (BCM) creates color depth by displaying each bit plane
// for a time proportional to its binary weight. This driver achieves BCM timing
// through DMA descriptor repetition:
//   - Bit 7 (MSB): 32 descriptors → displayed 32× longer
//   - Bit 6: 16 descriptors → displayed 16× longer
//   - ...
//   - Bit 0 (LSB): 1 descriptor → displayed 1×
//
// The OE signal controls whether LEDs are actually lit during each bit plane's
// transmission. By keeping OE HIGH (disabled) for part of each transmission,
// we reduce perceived brightness WITHOUT changing BCM ratios.
//
// Key insight: Since BCM ratios come from descriptor repetition, OE duty cycle
// is applied UNIFORMLY to all bit planes. Each bit plane has the same percentage
// of pixels with OE=LOW (enabled). This dims the display while preserving color
// accuracy.
//
// Center-based OE placement
// -------------------------
// The enabled region (OE=LOW) is centered in the buffer rather than left-aligned.
// This provides symmetric blanking margins on both sides, which:
//   1. Keeps the display period away from buffer edges where timing is less stable
//   2. Provides natural separation from the LAT pulse at the end
//   3. Distributes any timing jitter symmetrically
//
void GdmaDma::set_brightness_oe_internal(RowBitPlaneBuffer *buffers, uint8_t brightness) {
  if (!buffers) {
    return;
  }

  const uint8_t latch_blanking = config_.latch_blanking;

  // brightness=0 blanks the display entirely
  if (brightness == 0) {
    for (int row = 0; row < num_rows_; row++) {
      for (int bit = 0; bit < bit_depth_; bit++) {
        uint16_t *buf = plane_ptr(buffers[row], bit);
        for (int x = 0; x < dma_width_; x++) {
          buf[x] |= (1 << OE_BIT);
        }
      }
    }
    return;
  }

  // Remap user brightness through quadratic curve
  //
  // The curve passes through (1, min), (128, 128), (255, 255) ensuring:
  // - Minimum floor for BCM color accuracy at low brightness
  // - Brightness 128 remains the perceptual midpoint (unchanged from pre-floor behavior)
  // - Maximum brightness unchanged
  //
  // See init_brightness_coeffs() for coefficient calculation.
  const int effective_brightness = remap_brightness(brightness);

  for (int row = 0; row < num_rows_; row++) {
    for (int bit = 0; bit < bit_depth_; bit++) {
      uint16_t *buf = plane_ptr(buffers[row], bit);

      // Uniform OE duty cycle: same display_pixels count for all bit planes.
      // BCM ratios come from descriptor repetition, not OE timing.
      const int max_pixels = dma_width_ - latch_blanking;
      int display_pixels = (max_pixels * effective_brightness) >> 8;

      // Edge case fallback for very low brightness
      //
      // Even with the brightness floor, integer truncation can result in display_pixels=0
      // for some configurations. This fallback ensures at least 1 pixel is enabled for
      // the most significant bits, which contribute most to perceived brightness.
      //
      // The threshold increases with brightness: at very low brightness only bit 7 gets
      // the minimum; as brightness increases, more bits naturally exceed 0 anyway.
      //   effective_brightness 1-15:   only bit 7 guaranteed minimum
      //   effective_brightness 16-31:  bits 6-7 guaranteed minimum
      //   effective_brightness 32-47:  bits 5-7 guaranteed minimum, etc.
      const int min_bit_for_display = std::max(0, bit_depth_ - 1 - (effective_brightness >> 4));
      if (effective_brightness > 0 && display_pixels == 0 && bit >= min_bit_for_display) {
        display_pixels = 1;
      }

      // Reserve at least 1 pixel blanking to prevent ghosting at maximum brightness.
      // Without this margin, brightness=255 would enable all pixels including those
      // near the LAT pulse, potentially causing visible artifacts.
      display_pixels = std::min(display_pixels, max_pixels - 1);

      assert(max_pixels >= 2 && "max_pixels < 2: insufficient headroom for safety margin");
      assert(display_pixels >= 0 && "display_pixels underflow");
      assert(display_pixels <= max_pixels - 1 && "display_pixels exceeds safety margin");

      // Center the enabled region in the buffer
      const int x_min = (dma_width_ - display_pixels) / 2;
      const int x_max = (dma_width_ + display_pixels) / 2;

      assert(x_min >= 0 && "x_min underflow");
      assert(x_max <= dma_width_ && "x_max exceeds buffer bounds");
      assert(x_min <= x_max && "x_min > x_max: invalid display region");

      // Apply OE pattern: LOW (enabled) in center, HIGH (blanked) elsewhere
      for (int x = 0; x < dma_width_; x++) {
        if (x >= x_min && x < x_max) {
          buf[x] &= OE_CLEAR_MASK;
        } else {
          buf[x] |= (1 << OE_BIT);
        }
      }

      // Latch blanking: force OE=HIGH around the LAT pulse
      //
      // The LAT (latch) signal on the last pixel transfers shift register data to the
      // display buffer. The panel needs the display blanked during this transition to
      // prevent visible artifacts from partially-latched data. Blanking pixels at both
      // the start and end of the buffer ensures clean transitions regardless of where
      // the centered display region falls.
      const int last_pixel = dma_width_ - 1;

      // Blank the LAT pixel itself (always required)
      buf[last_pixel] |= (1 << OE_BIT);

      // Blank latch_blanking pixels before LAT
      for (int i = 1; i <= latch_blanking && (last_pixel - i) >= 0; i++) {
        buf[last_pixel - i] |= (1 << OE_BIT);
      }

      // Blank latch_blanking pixels at buffer start (for wrap-around from previous row)
      for (int i = 0; i < latch_blanking && i < dma_width_; i++) {
        buf[i] |= (1 << OE_BIT);
      }
    }
  }
}

void GdmaDma::set_brightness_oe() {
  if (!row_buffers_[0]) {
    ESP_LOGE(TAG, "Row buffers not allocated");
    return;
  }

  // Calculate brightness scaling (0-255 maps to 0-255)
  const uint8_t brightness = (uint8_t) ((float) basis_brightness_ * intensity_);

  ESP_LOGD(TAG, "Setting brightness OE: brightness=%u, lsbMsbTransitionBit=%u", brightness, lsbMsbTransitionBit_);

  // Update OE bits in all allocated buffers. Both are rewritten, so both are written back:
  // the back buffer becomes the front one on the next flip.
  for (int i = 0; i < 2; i++) {
    if (row_buffers_[i]) {
      set_brightness_oe_internal(row_buffers_[i], brightness);
      mark_all_rows_dirty(i);
      sync_dirty_rows(i);
    }
  }

  ESP_LOGD(TAG, "Brightness OE configuration complete");
}

bool GdmaDma::build_descriptor_chain() {
  // Calculate total descriptors needed WITH BCM repetitions
  // For bits <= lsbMsbTransitionBit: 1 descriptor each (base timing)
  // For bits > lsbMsbTransitionBit: 2^(bit - lsbMsbTransitionBit - 1) descriptors each
  descriptor_count_ = 0;
  for (int row = 0; row < num_rows_; row++) {
    for (int bit = 0; bit < bit_depth_; bit++) {
      if (bit <= lsbMsbTransitionBit_) {
        descriptor_count_ += 1;  // Base timing
      } else {
        descriptor_count_ += (1 << (bit - lsbMsbTransitionBit_ - 1));  // BCM repetitions
      }
    }
  }

  // Safety check to prevent division by zero
  if (num_rows_ == 0) {
    ESP_LOGE(TAG, "Invalid configuration: num_rows_ is 0");
    return false;
  }

  size_t descriptors_per_row = descriptor_count_ / num_rows_;
  size_t total_descriptor_bytes = sizeof(dma_descriptor_t) * descriptor_count_;

  ESP_LOGI(TAG, "Building BCM descriptor chain: %zu descriptors (%zu per row) for %d rows × %d bits", descriptor_count_,
           descriptors_per_row, num_rows_, bit_depth_);
  ESP_LOGI(TAG, "  BCM via descriptor repetition (lsbMsbTransitionBit=%d)", lsbMsbTransitionBit_);
  ESP_LOGI(TAG, "  Allocating %zu bytes for the descriptor chain", total_descriptor_bytes);

  // Free existing descriptors if already allocated (prevent leak on retry)
  if (descriptors_) {
    heap_caps_free(descriptors_);
    descriptors_ = nullptr;
  }

  // GDMA only fetches descriptors from internal RAM. calloc keeps unused control bits zero.
  descriptors_ = (dma_descriptor_t *) heap_caps_calloc(1, total_descriptor_bytes, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
  if (!descriptors_) {
    ESP_LOGE(TAG, "Failed to allocate %zu descriptors (%zu bytes) in DMA memory", descriptor_count_,
             total_descriptor_bytes);
    return false;
  }

  // The chain starts on the front buffer; flip_buffer() retargets it to the other one
  const RowBitPlaneBuffer *const buffers = row_buffers_[front_idx_];
  const size_t bytes_per_bitplane = dma_width_ * sizeof(uint16_t);

  // Link descriptors with BCM repetitions
  size_t desc_idx = 0;
  const int configured_sync_row = config_.frame_sync_row;
  const int sync_row =
      (configured_sync_row >= 0 && configured_sync_row < num_rows_) ? configured_sync_row : (num_rows_ - 1);
  for (int row = 0; row < num_rows_; row++) {
    for (int bit = 0; bit < bit_depth_; bit++) {
      uint8_t *const bit_buffer = reinterpret_cast<uint8_t *>(plane_ptr(buffers[row], bit));

      // Calculate number of descriptor repetitions for this bit plane
      const int repetitions = (bit <= lsbMsbTransitionBit_) ? 1  // Base timing for LSBs
                                                            : (1 << (bit - lsbMsbTransitionBit_ - 1));  // BCM weighting

      // Create 'repetitions' descriptors, all pointing to the SAME buffer
      // This achieves BCM timing via temporal repetition
      for (int rep = 0; rep < repetitions; rep++) {
        dma_descriptor_t *const desc = &descriptors_[desc_idx];
        desc->dw0.owner = DMA_DESCRIPTOR_BUFFER_OWNER_DMA;
        desc->dw0.suc_eof = 0;  // EOF only on last descriptor
        desc->dw0.size = bytes_per_bitplane;
        desc->dw0.length = bytes_per_bitplane;
        desc->buffer = bit_buffer;  // Same buffer for all repetitions

        // Link to next descriptor
        if (desc_idx < descriptor_count_ - 1) {
          desc->next = &descriptors_[desc_idx + 1];
        }

        desc_idx++;
      }
    }

    // Keep exactly one EOF per refresh. On a single-buffer display this may
    // sit just after a latency-sensitive region, leaving the remaining rows
    // as a safe window to update that region before the chain wraps.
    if (row == sync_row) {
      descriptors_[desc_idx - 1].dw0.suc_eof = 1;
    }
  }

  // Last descriptor loops back to first (continuous refresh)
  descriptors_[descriptor_count_ - 1].next = &descriptors_[0];

  ESP_LOGI(TAG, "BCM descriptor chain built: %zu descriptors in continuous loop%s", descriptor_count_,
           is_double_buffered() ? " (shared by both buffers)" : "");

  return true;
}

// ============================================================================
// BCM Timing Calculation (Platform-Specific)
// ============================================================================

// Calculate number of transmissions per row for BCM timing
// Must match the actual descriptor allocation in build_descriptor_chain():
//   bits <= transition: 1 descriptor each
//   bits > transition: 2^(bit - transition - 1) descriptors each
HUB75_CONST constexpr int GdmaDma::calculate_bcm_transmissions(int bit_depth, int lsb_msb_transition) {
  int transmissions = lsb_msb_transition + 1;  // Bits 0 to transition: 1 each

  // Add BCM repetitions for bits above transition
  for (int i = lsb_msb_transition + 1; i < bit_depth; ++i) {
    transmissions += (1 << (i - lsb_msb_transition - 1));
  }

  return transmissions;
}

void GdmaDma::calculate_bcm_timings() {
  // Calculate buffer transmission time
  // Buffer contains dma_width_ pixels with LAT on last pixel
  // Latch blanking is handled via OE bits, not extra pixels
  const uint16_t buffer_pixels = dma_width_;  // LAT is on last pixel, not extra
  const float buffer_time_us = (buffer_pixels * 1000000.0f) / actual_clock_hz_;

  ESP_LOGI(TAG, "Buffer transmission time: %.2f µs (%u pixels @ %lu Hz)", buffer_time_us, (unsigned) buffer_pixels,
           (unsigned long) actual_clock_hz_);

  // Target refresh rate from config
  const uint32_t target_hz = config_.min_refresh_rate;

  // Calculate optimal lsbMsbTransitionBit to achieve target refresh rate
  lsbMsbTransitionBit_ = 0;
  int actual_hz = 0;

  while (true) {
    // Calculate transmissions per row with current transition bit
    const int transmissions = GdmaDma::calculate_bcm_transmissions(bit_depth_, lsbMsbTransitionBit_);

    // Calculate refresh rate
    const float time_per_row_us = transmissions * buffer_time_us;
    const float time_per_frame_us = time_per_row_us * num_rows_;
    actual_hz = (int) (1000000.0f / time_per_frame_us);

    ESP_LOGD(TAG, "Testing lsbMsbTransitionBit=%d: %d transmissions/row, %d Hz", lsbMsbTransitionBit_, transmissions,
             actual_hz);

    if (actual_hz >= target_hz) [[likely]]
      break;

    if (lsbMsbTransitionBit_ < bit_depth_ - 1) [[likely]] {
      lsbMsbTransitionBit_++;
    } else {
      ESP_LOGW(TAG, "Cannot achieve target %lu Hz, max is %d Hz", (unsigned long) target_hz, actual_hz);
      break;
    }
  }

  ESP_LOGI(TAG, "lsbMsbTransitionBit=%d achieves %d Hz (target %lu Hz)", lsbMsbTransitionBit_, actual_hz,
           (unsigned long) target_hz);

  if (lsbMsbTransitionBit_ > 0) {
    ESP_LOGW(TAG,
             "Using lsbMsbTransitionBit=%d, lower %d bits show once "
             "(reduced color depth for speed)",
             lsbMsbTransitionBit_, lsbMsbTransitionBit_ + 1);
  }

  ESP_LOGI(TAG, "BCM timing calculated (lsbMsbTransitionBit used by set_brightness_oe for OE control)");
}

void GdmaDma::mark_y_span_dirty(int buffer_idx, uint16_t y, uint16_t h) {
  if (h == 0) {
    return;
  }
  // Identity transform: display row py lives in DMA row py (upper half) or py - num_rows_
  // (lower half). A span crossing the halves touches rows from 0 up to num_rows_.
  const uint16_t last_y = y + h - 1;
  if (last_y < num_rows_) {
    mark_rows_dirty(buffer_idx, y, last_y + 1);
  } else if (y >= num_rows_) {
    mark_rows_dirty(buffer_idx, y - num_rows_, last_y - num_rows_ + 1);
  } else {
    mark_all_rows_dirty(buffer_idx);
  }
}

esp_err_t GdmaDma::sync_dirty_rows(int buffer_idx) {
#if HUB75_EXTERNAL_FRAMEBUFFERS
  const uint16_t begin = dirty_row_begin_[buffer_idx];
  const uint16_t end = dirty_row_end_[buffer_idx];
  dirty_row_begin_[buffer_idx] = UINT16_MAX;
  dirty_row_end_[buffer_idx] = 0;
  if (!buffer_in_psram_[buffer_idx] || begin >= end) {
    return ESP_OK;
  }

  // Widen the row range to whole cache lines. The allocation is aligned and padded to
  // PSRAM_FB_ALIGNMENT, so the widened range never leaves this buffer.
  const size_t start = (begin * row_stride_bytes_) & ~(PSRAM_FB_ALIGNMENT - 1);
  const size_t stop =
      std::min(total_buffer_bytes_, (end * row_stride_bytes_ + PSRAM_FB_ALIGNMENT - 1) & ~(PSRAM_FB_ALIGNMENT - 1));
  uint8_t *const addr = dma_buffers_[buffer_idx] + start;
  const size_t size = stop - start;

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 2, 0)
  const esp_err_t err = esp_cache_msync(addr, size, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
#elif ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 1, 0)
  // C2M is the only (implicit) direction before IDF 5.2
  const esp_err_t err = esp_cache_msync(addr, size, 0);
#else
  const esp_err_t err = Cache_WriteBack_Addr((uint32_t) addr, size) == 0 ? ESP_OK : ESP_FAIL;
#endif
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "PSRAM cache write-back failed: %s", esp_err_to_name(err));
  }
  return err;
#else
  return ESP_OK;
#endif
}

void GdmaDma::verify_psram_writeback(int buffer_idx, esp_err_t sync_err) {
#if HUB75_EXTERNAL_FRAMEBUFFERS && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 2, 0)
  if (!buffer_in_psram_[buffer_idx] || split_plane_ == 0) {
    return;
  }
  // Drop the (clean, just written back) cache lines of row 0 bit plane 0 and read them again,
  // so the values below come from PSRAM, which is what the GDMA sees.
  uint16_t *const plane = reinterpret_cast<uint16_t *>(dma_buffers_[buffer_idx]);
  const esp_err_t inv_err = esp_cache_msync(plane, plane_bytes_, ESP_CACHE_MSYNC_FLAG_DIR_M2C);

  // Row 0, bit plane 0 carries the previous (last) row address, OE high, LAT on the last pixel
  const uint16_t addr_bits = ((num_rows_ - 1) & ADDR_MASK) << ADDR_SHIFT;
  const uint16_t expect_first = addr_bits | (1 << OE_BIT);
  const uint16_t expect_last = expect_first | (1 << LAT_BIT);
  const uint16_t first = plane[0];
  const uint16_t last = plane[dma_width_ - 1];
  const bool ok = sync_err == ESP_OK && inv_err == ESP_OK && first == expect_first && last == expect_last;
  ESP_LOGI(TAG,
           "Buffer %c PSRAM write-back check %s: msync=%s invalidate=%s first=0x%04x/0x%04x last=0x%04x/0x%04x (%p)",
           'A' + buffer_idx, ok ? "OK" : "FAILED", esp_err_to_name(sync_err), esp_err_to_name(inv_err), first,
           expect_first, last, expect_last, plane);
#endif
}

void GdmaDma::log_dma_health() {
  // One-time check shortly after start: the LCD must still be running, the GDMA must be walking
  // the chain, and its TX FIFO must not have underrun (PSRAM starvation).
  int chan_id = 0;
  if (gdma_get_group_channel_id(dma_chan_, nullptr, &chan_id) != ESP_OK) {
    return;
  }
  esp_rom_delay_us(2000);
  const uint32_t desc_a = GDMA.channel[chan_id].out.dscr;
  esp_rom_delay_us(1000);
  const uint32_t desc_b = GDMA.channel[chan_id].out.dscr;
  const uint32_t raw = gdma_ll_tx_get_interrupt_status(&GDMA, chan_id, true);
  const bool lcd_running = LCD_CAM.lcd_user.lcd_start;
  const bool underrun = raw & (GDMA_LL_EVENT_TX_L1_FIFO_UDF | GDMA_LL_EVENT_TX_L3_FIFO_UDF);
  const bool desc_error = raw & GDMA_LL_EVENT_TX_DESC_ERROR;
  const bool healthy = lcd_running && desc_a != desc_b && !underrun && !desc_error;
  const bool psram = buffer_in_psram_[0] || buffer_in_psram_[1];

  if (healthy && !psram) {
    ESP_LOGD(TAG, "DMA health OK: lcd running, desc 0x%08lx -> 0x%08lx", (unsigned long) desc_a,
             (unsigned long) desc_b);
    return;
  }
  ESP_LOGI(TAG,
           "DMA health %s @ %.2f MHz (%s): lcd_start=%d desc 0x%08lx -> 0x%08lx (%s) gdma_raw=0x%02lx "
           "fifo_underrun=%d desc_error=%d",
           healthy ? "OK" : "FAILED", actual_clock_hz_ / 1000000.0f, psram ? "PSRAM" : "internal RAM", lcd_running,
           (unsigned long) desc_a, (unsigned long) desc_b, desc_a != desc_b ? "moving" : "stuck", (unsigned long) raw,
           underrun, desc_error);
}

void GdmaDma::start_stall_watchdog() {
#if HUB75_EXTERNAL_FRAMEBUFFERS
  if (stall_timer_ || (!buffer_in_psram_[0] && !buffer_in_psram_[1])) {
    return;
  }
  esp_timer_create_args_t args = {};
  args.callback = &GdmaDma::stall_watchdog_cb;
  args.arg = this;
  args.dispatch_method = ESP_TIMER_TASK;
  args.name = "hub75_stall";
  if (esp_timer_create(&args, &stall_timer_) != ESP_OK) {
    stall_timer_ = nullptr;
    ESP_LOGW(TAG, "Could not create the DMA stall watchdog");
    return;
  }
  esp_timer_start_periodic(stall_timer_, 100 * 1000);
#endif
}

void GdmaDma::stop_stall_watchdog() {
  if (stall_timer_) {
    esp_timer_stop(stall_timer_);
    esp_timer_delete(stall_timer_);
    stall_timer_ = nullptr;
  }
}

void GdmaDma::stall_watchdog_cb(void *arg) {
  auto *self = static_cast<GdmaDma *>(arg);
  int chan_id = 0;
  if (!self->transfer_running_ || !self->dma_chan_ ||
      gdma_get_group_channel_id(self->dma_chan_, nullptr, &chan_id) != ESP_OK) {
    return;
  }
  // A running chain moves to a new descriptor every dma_width_ clocks; allow three of them
  const uint32_t desc_time_us = (self->dma_width_ * 1000000u + self->actual_clock_hz_ - 1) / self->actual_clock_hz_;
  const uint32_t desc_a = GDMA.channel[chan_id].out.dscr;
  esp_rom_delay_us(3 * desc_time_us + 2);
  const uint32_t desc_b = GDMA.channel[chan_id].out.dscr;
  if (desc_a != desc_b || !self->transfer_running_) {
    return;
  }

  const uint32_t raw = gdma_ll_tx_get_interrupt_status(&GDMA, chan_id, true);
  const bool lcd_running = LCD_CAM.lcd_user.lcd_start;
  self->stall_restarts_++;
  // Rate limited: the first few, then every 100th
  if (self->stall_restarts_ <= 5 || self->stall_restarts_ % 100 == 0) {
    ESP_LOGW(TAG, "DMA stalled (PSRAM starvation?) at desc 0x%08lx, lcd_start=%d gdma_raw=0x%02lx; restart #%lu",
             (unsigned long) desc_a, lcd_running, (unsigned long) raw, (unsigned long) self->stall_restarts_);
  }
  self->restart_transfer();
}

void GdmaDma::restart_transfer() {
  LCD_CAM.lcd_user.lcd_start = 0;
  LCD_CAM.lcd_user.lcd_update = 1;
  gdma_stop(dma_chan_);
  gdma_reset(dma_chan_);
  LCD_CAM.lcd_misc.lcd_afifo_reset = 1;

  LCD_CAM.lcd_user.lcd_update = 1;
  esp_rom_delay_us(10);
  gdma_start(dma_chan_, (intptr_t) &descriptors_[0]);
  esp_rom_delay_us(100);
  LCD_CAM.lcd_user.lcd_start = 1;
}

// ============================================================================
// Compile-Time Validation (ESP-IDF 5.x only - requires consteval/GCC 9+)
// ============================================================================

#if ESP_IDF_VERSION_MAJOR >= 5
namespace {

// Validate BCM calculations match actual descriptor allocation
// Formula: (transition + 1) base + sum of 2^(i - transition - 1) for i > transition
consteval bool test_bcm_12bit_transition0() {
  // 12-bit depth, transition=0: 1 + (1+2+4+8+16+32+64+128+256+512+1024) = 1 + 2047 = 2048
  constexpr int transmissions = GdmaDma::calculate_bcm_transmissions(12, 0);
  return transmissions == 2048;
}

consteval bool test_bcm_10bit_transition0() {
  // 10-bit depth, transition=0: 1 + (1+2+4+8+16+32+64+128+256) = 1 + 511 = 512
  constexpr int transmissions = GdmaDma::calculate_bcm_transmissions(10, 0);
  return transmissions == 512;
}

consteval bool test_bcm_8bit_transition0() {
  // 8-bit depth, transition=0: 1 + (1+2+4+8+16+32+64) = 1 + 127 = 128
  constexpr int transmissions = GdmaDma::calculate_bcm_transmissions(8, 0);
  return transmissions == 128;
}

consteval bool test_bcm_8bit_transition1() {
  // 8-bit, transition=1: 2 + (1+2+4+8+16+32) = 2 + 63 = 65
  constexpr int transmissions = GdmaDma::calculate_bcm_transmissions(8, 1);
  return transmissions == 65;
}

consteval bool test_bcm_8bit_transition2() {
  // 8-bit, transition=2: 3 + (1+2+4+8+16) = 3 + 31 = 34
  constexpr int transmissions = GdmaDma::calculate_bcm_transmissions(8, 2);
  return transmissions == 34;
}

// Static assertions
static_assert(test_bcm_12bit_transition0(), "BCM: 12-bit/transition=0 should produce 2048 transmissions");
static_assert(test_bcm_10bit_transition0(), "BCM: 10-bit/transition=0 should produce 512 transmissions");
static_assert(test_bcm_8bit_transition0(), "BCM: 8-bit/transition=0 should produce 128 transmissions");
static_assert(test_bcm_8bit_transition1(), "BCM: 8-bit/transition=1 should produce 65 transmissions");
static_assert(test_bcm_8bit_transition2(), "BCM: 8-bit/transition=2 should produce 34 transmissions");

}  // namespace
#endif  // ESP_IDF_VERSION_MAJOR >= 5

}  // namespace hub75

#endif  // CONFIG_IDF_TARGET_ESP32S3
