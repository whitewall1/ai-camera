#pragma once
#include <stdint.h>

// 启动 LVGL UI 交互线程
// drm_map_ptr: 底层 DRM 引擎提供的 ARGB8888 虚拟内存首地址
// width, height: 屏幕分辨率 (1024x600)
void lvgl_worker_func(void* drm_map_ptr, uint32_t width, uint32_t height);