#pragma once
#include <cstdint>
#include <iostream>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h> // 新增：为了使用 DRM_FORMAT_ARGB8888 宏

class DisplayEngine {
public:
    DisplayEngine();
    ~DisplayEngine();

    bool init();

    // 接口 1：给视频层 (RGA) 用的底层显存，返回 DMA FD
    int create_dumb_buffer_fd(uint32_t* out_fb_id);
    bool present_fb(uint32_t fb_id);

    // 接口 2 (新增！)：给 UI 层 (LVGL) 用的透明显存，直接返回 CPU 可写的虚拟内存指针
    void* create_overlay_plane(uint32_t* out_fb_id);

private:
    int drm_fd_;
    uint32_t crtc_id_;
    uint32_t crtc_index_; // 新增：DRM 寻找 Plane 时需要用到索引，而不是 ID
    uint32_t connector_id_;
    uint32_t overlay_plane_id_; // 新增：记录我们抢到的透明图层 ID
    drmModeModeInfo display_mode_;
    
    void cleanup();
};