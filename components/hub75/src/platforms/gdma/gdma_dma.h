// SPDX-FileCopyrightText: 2025 Stuart Parmenter
// SPDX-License-Identifier: MIT
//
// @file gdma_dma.h
// @brief ESP32-S3 LCD_CAM peripheral + GDMA for HUB75
//
// Uses ESP32-S3's LCD_CAM peripheral with direct register access
// and manual GDMA setup for continuous data transfer.

#pragma once

#include "hub75_types.h"
#include "hub75_config.h"
#include "hub75_internal.h"  // For Hub75FramebufferFormat
#include "../platform_dma.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <variant>
#include <esp_private/gdma.h>
#include <esp_timer.h>
#include <hal/dma_types.h>

namespace hub75 {

// Forward declaration
class Framebuffer;

/**
 * @brief ESP32-S3 GDMA + LCD_CAM implementation for HUB75
 */
class GdmaDma : public PlatformDma {
 public:
  GdmaDma(const Hub75Config &config);
  ~GdmaDma();

  /**
   * @brief Initialize LCD_CAM peripheral with GDMA
   */
  bool init() override;

  /**
   * @brief Shutdown LCD_CAM and free GDMA resources
   */
  void shutdown() override;

  /**
   * @brief Start DMA transfer (starts once, runs continuously)
   */
  void start_transfer() override;

  /**
   * @brief Stop DMA transfer
   */
  void stop_transfer() override;

  void set_frame_callback(Hub75FrameCallback callback, void *arg) override;

  static bool IRAM_ATTR on_trans_eof(gdma_channel_handle_t dma_chan, gdma_event_data_t *event_data, void *user_data);

  /**
   * @brief Set basis brightness (override base class)
   */
  void set_basis_brightness(uint8_t brightness) override;

  /**
   * @brief Set intensity (override base class)
   */
  void set_intensity(float intensity) override;

  /**
   * @brief Set display rotation (override base class)
   */
  void set_rotation(Hub75Rotation rotation) override;

  /**
   * @brief Resolve clock speed to achievable frequency (160 MHz / N)
   */
  HUB75_CONST uint32_t resolve_actual_clock_speed(Hub75ClockSpeed clock_speed) const;

  // ============================================================================
  // Pixel API (Direct DMA Buffer Writes)
  // ============================================================================

  /**
   * @brief Draw pixels from buffer (bulk operation, writes directly to DMA buffers)
   */
  void draw_pixels(uint16_t x, uint16_t y, uint16_t w, uint16_t h, const uint8_t *buffer, Hub75PixelFormat format,
                   Hub75ColorOrder color_order, bool big_endian) override;

  /**
   * @brief Clear all pixels to black
   */
  void clear() override;

  /**
   * @brief Fill a rectangular region with a solid color
   */
  void fill(uint16_t x, uint16_t y, uint16_t w, uint16_t h, uint8_t r, uint8_t g, uint8_t b) override;

  /**
   * @brief Swap front and back buffers (double buffer mode only)
   */
  void flip_buffer() override;

  /**
   * @brief Get GDMA channel handle for external callback registration
   * @return GDMA channel handle, or NULL if not initialized
   */
  gdma_channel_handle_t getGdmaChannel() const { return dma_chan_; }

  // ============================================================================
  // Static Helper Functions (Public for compile-time validation)
  // ============================================================================

  /**
   * @brief Calculate BCM transmissions per row for given bit depth and transition bit
   */
  static constexpr int calculate_bcm_transmissions(int bit_depth, int lsb_msb_transition);

  // Bit planes of one row. Planes [0, split_plane_) are contiguous at `low`, planes
  // [split_plane_, bit_depth_) contiguous at `high`. Without a split, `high` is simply where
  // `low` ends, so a row is one contiguous [bit0][bit1]...[bitN] block.
  struct RowBitPlaneBuffer {
    uint8_t *low;
    uint8_t *high;
  };

 private:
  void configure_lcd_clock();
  void configure_lcd_mode();
  void configure_gpio();

  // Calls fn(bit, plane) for every bit plane of a row, in bit order
  template<typename Fn>
  __attribute__((always_inline)) inline void for_each_plane(const RowBitPlaneBuffer &row, Fn &&fn) const {
    uint8_t *plane = row.low;
    int bit = 0;
    for (; bit < split_plane_; bit++, plane += plane_bytes_) {
      fn(bit, reinterpret_cast<uint16_t *>(plane));
    }
    plane = row.high;
    for (; bit < bit_depth_; bit++, plane += plane_bytes_) {
      fn(bit, reinterpret_cast<uint16_t *>(plane));
    }
  }
  uint16_t *plane_ptr(const RowBitPlaneBuffer &row, int bit) const {
    return reinterpret_cast<uint16_t *>(bit < split_plane_ ? row.low + bit * plane_bytes_
                                                           : row.high + (bit - split_plane_) * plane_bytes_);
  }

  // Buffer management
  void plan_psram_layout();
  uint8_t transition_bit_for_clock(uint32_t clock_hz) const;
  bool allocate_row_buffers();
  uint8_t *allocate_framebuffer(size_t size, bool *in_psram);
  bool configure_dma_transfer();
  bool validate_brightness_config();  // Validate safety margins for brightness OE configuration
  void initialize_blank_buffers();    // Initialize DMA buffers with control bits only
  void initialize_buffer_internal(RowBitPlaneBuffer *buffers);                      // Helper: initialize one buffer set
  void set_brightness_oe();                                                         // Set OE bits for BCM control
  void set_brightness_oe_internal(RowBitPlaneBuffer *buffers, uint8_t brightness);  // Helper: set OE for one buffer
  bool build_descriptor_chain();

  // BCM timing calculation (calculates lsbMsbTransitionBit for OE control)
  void calculate_bcm_timings();

  // True once buffer B exists; the driver falls back to single buffering when it cannot be allocated.
  bool is_double_buffered() const { return row_buffers_[1] != nullptr; }

  // PSRAM framebuffers are written through the CPU data cache, while GDMA reads PSRAM directly.
  // Writers record the rows they touched and the dirty range is written back to PSRAM in one
  // esp_cache_msync() before the DMA can see it: at the end of each draw call in single-buffer
  // mode, at flip_buffer() in double-buffer mode. All of this compiles away without
  // HUB75_EXTERNAL_FRAMEBUFFERS.
  void mark_rows_dirty(int buffer_idx, uint16_t first_row, uint16_t end_row) {
#if HUB75_EXTERNAL_FRAMEBUFFERS
    dirty_row_begin_[buffer_idx] = std::min(dirty_row_begin_[buffer_idx], first_row);
    dirty_row_end_[buffer_idx] = std::max(dirty_row_end_[buffer_idx], end_row);
#endif
  }
  void mark_all_rows_dirty(int buffer_idx) { mark_rows_dirty(buffer_idx, 0, num_rows_); }
  void mark_y_span_dirty(int buffer_idx, uint16_t y, uint16_t h);  // Identity transform only
  esp_err_t sync_dirty_rows(int buffer_idx);

  void verify_psram_writeback(int buffer_idx, esp_err_t sync_err);
  void log_dma_health();

  // When PSRAM cannot feed the GDMA in time the transfer can halt and the scan freezes on one
  // row. A periodic check restarts the transfer when the GDMA stops advancing. Only armed for
  // PSRAM framebuffers.
  void start_stall_watchdog();
  void stop_stall_watchdog();
  static void stall_watchdog_cb(void *arg);
  void restart_transfer();
  esp_timer_handle_t stall_timer_ = nullptr;
  std::atomic<bool> transfer_running_{false};  // Watchdog may restart only while set
  uint32_t stall_restarts_ = 0;
  bool flip_sync_logged_ = false;

  size_t plane_bytes_;         // Bytes per bit plane (dma_width_ 16-bit words)
  uint8_t split_plane_;        // Planes below this live in dma_buffers_, the rest in hot_buffers_
  size_t row_stride_bytes_;    // Bytes per row in dma_buffers_ (planes below split_plane_)
  size_t total_buffer_bytes_;  // Bytes per dma_buffers_ allocation (padded to the PSRAM alignment when in PSRAM)
  size_t hot_buffer_bytes_;    // Bytes per hot_buffers_ allocation (0 without a split)

  gdma_channel_handle_t dma_chan_;
  const uint8_t bit_depth_;      // Bit depth from config (6, 7, 8, 10, or 12)
  uint8_t lsbMsbTransitionBit_;  // BCM optimization threshold (calculated at init)
  uint32_t actual_clock_hz_;     // Achieved clock after rounding (lowered for PSRAM framebuffers)

  // Panel configuration (immutable, cached from config)
  const uint16_t panel_width_;
  const uint16_t panel_height_;
  const uint16_t layout_rows_;
  const uint16_t layout_cols_;
  const uint16_t virtual_width_;   // Visual display width: panel_width * layout_cols
  const uint16_t virtual_height_;  // Visual display height: panel_height * layout_rows
  const uint16_t dma_width_;       // DMA buffer width: panel_width * layout_rows * layout_cols (row-major chaining)

  // Coordinate transformation (immutable, cached from config)
  const Hub75ScanWiring scan_wiring_;
  const Hub75PanelLayout layout_;

  // Optimization flags (immutable, for branch prediction)
  const bool needs_scan_remap_;
  const bool needs_layout_remap_;

  // Display rotation (mutable, can change at runtime)
  Hub75Rotation rotation_;

  const uint16_t num_rows_;  // Computed: panel_height / 2

  // Double buffering: Array + index architecture
  // [0] = buffer A (always allocated), [1] = buffer B (nullptr if single-buffer mode)
  uint8_t *dma_buffers_[2];            // Low bit planes (PSRAM with HUB75_EXTERNAL_FRAMEBUFFERS), or all of them
  uint8_t *hot_buffers_[2];            // High (most repeated) bit planes in internal RAM, or nullptr
  RowBitPlaneBuffer *row_buffers_[2];  // Metadata arrays pointing into dma_buffers_
  bool buffer_in_psram_[2];            // Allocation landed in PSRAM (needs cache write-back before DMA sees it)

  // One descriptor chain serves both buffers and always points at buffers[front_idx_].
  // flip_buffer() retargets it instead of keeping a second chain, which would cost another
  // descriptor_count_ * 12 bytes of internal RAM (GDMA cannot fetch descriptors from PSRAM).
  dma_descriptor_t *descriptors_;

  int front_idx_;   // DMA displays buffers[front_idx_]
  int active_idx_;  // CPU draws to buffers[active_idx_]

#if HUB75_EXTERNAL_FRAMEBUFFERS
  // Dirty row range [begin, end) per buffer; begin >= end means clean.
  uint16_t dirty_row_begin_[2];
  uint16_t dirty_row_end_[2];
#endif

  size_t descriptor_count_;  // Number of descriptors in the chain

  // Brightness control (implementation of base class interface)
  uint8_t basis_brightness_;  // 1-255
  float intensity_;           // 0.0-1.0

  // Pixel pattern cache for consecutive identical pixels optimization
  // Persists across draw_pixels() calls to benefit single-pixel APIs
  uint32_t cached_raw_pixel_ = 0;  // Raw source bytes packed into uint32_t
  // Sized for the maximum depth; only the first bit_depth_ entries are used.
  uint16_t cached_upper_patterns_[HUB75_MAX_BIT_DEPTH] = {0};
  uint16_t cached_lower_patterns_[HUB75_MAX_BIT_DEPTH] = {0};
};

}  // namespace hub75
