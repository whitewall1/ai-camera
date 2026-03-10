#include "display_engine.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>

DisplayEngine::DisplayEngine() : drm_fd_(-1), crtc_id_(0), connector_id_(0) {}

DisplayEngine::~DisplayEngine() {
    cleanup();
}
int DisplayEngine::create_dumb_buffer_fd(uint32_t* out_fb_id) {
    if (drm_fd_ < 0 || crtc_id_ == 0) return -1;

    // 1. 申请 1024x600 的显存
    drm_mode_create_dumb create_req = {};
    create_req.width = display_mode_.hdisplay;
    create_req.height = display_mode_.vdisplay;
    create_req.bpp = 32;
    if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) return -1;

    // 2. 包装成 DRM 认识的 Framebuffer (深度24, bpp32 = XRGB8888)
    uint32_t fb_id;
    if (drmModeAddFB(drm_fd_, display_mode_.hdisplay, display_mode_.vdisplay, 
                     24, 32, create_req.pitch, create_req.handle, &fb_id) < 0) return -1;

    // 3. 核心奇迹：将这块 DRM 显存的硬件 handle 转换成 Linux 标准的 DMA-BUF fd
    int dma_fd = -1;
    if (drmPrimeHandleToFD(drm_fd_, create_req.handle, 0, &dma_fd) < 0) return -1;

    *out_fb_id = fb_id;
    return dma_fd; // 把这把钥匙交给 RGA
}

bool DisplayEngine::present_fb(uint32_t fb_id) {
    // 触发硬件 VOP2 翻页送显
    return (drmModeSetCrtc(drm_fd_, crtc_id_, fb_id, 0, 0, &connector_id_, 1, &display_mode_) == 0);
}
void DisplayEngine::cleanup() {
    if (drm_fd_ >= 0) {
        close(drm_fd_);
        drm_fd_ = -1;
    }
}

bool DisplayEngine::init() {
    drm_fd_ = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (drm_fd_ < 0) {
        std::cerr << "DisplayEngine: Failed to open /dev/dri/card0" << std::endl;
        return false;
    }

    drmModeRes *res = drmModeGetResources(drm_fd_);
    if (!res) return false;

    drmModeConnector *conn = nullptr;
    // 1. 寻找 MIPI 屏幕
    for (int i = 0; i < res->count_connectors; i++) {
        conn = drmModeGetConnector(drm_fd_, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            connector_id_ = conn->connector_id;
            display_mode_ = conn->modes[0];
            break;
        }
        drmModeFreeConnector(conn);
        conn = nullptr;
    }

    if (!conn) {
        drmModeFreeResources(res);
        return false;
    }

    // 2. 动态匹配 CRTC
    drmModeEncoder *enc = nullptr;
    if (conn->encoder_id) {
        enc = drmModeGetEncoder(drm_fd_, conn->encoder_id);
        if (enc && enc->crtc_id) crtc_id_ = enc->crtc_id;
        if (enc) drmModeFreeEncoder(enc);
    }

    if (!crtc_id_) {
        for (int i = 0; i < conn->count_encoders; i++) {
            enc = drmModeGetEncoder(drm_fd_, conn->encoders[i]);
            if (enc) {
                for (int j = 0; j < res->count_crtcs; j++) {
                    if (enc->possible_crtcs & (1 << j)) {
                        crtc_id_ = res->crtcs[j];
                        crtc_index_ = j;
                        break;
                    }
                }
                drmModeFreeEncoder(enc);
                if (crtc_id_) break;
            }
        }
    }

    drmModeFreeConnector(conn);
    drmModeFreeResources(res);

    return crtc_id_ != 0;
}


void* DisplayEngine::create_overlay_plane(uint32_t* out_fb_id) {
    if (drm_fd_ < 0 || crtc_id_ == 0) return nullptr;

    uint32_t width = display_mode_.hdisplay;
    uint32_t height = display_mode_.vdisplay;

    // 1. 申请 Dumb Buffer (注意 bpp=32, 给 ARGB 留足空间)
    drm_mode_create_dumb create_req = {};
    create_req.width = width;
    create_req.height = height;
    create_req.bpp = 32;
    if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) {
        std::cerr << "[ERROR] Overlay 显存申请失败！" << std::endl;
        return nullptr;
    }

    // 2. 注册为 Framebuffer，【核心机制：强制指定格式为 ARGB8888】
    uint32_t handles[4] = {create_req.handle, 0, 0, 0};
    uint32_t pitches[4] = {create_req.pitch, 0, 0, 0};
    uint32_t offsets[4] = {0, 0, 0, 0};
    
    // 使用 drmModeAddFB2，因为它允许我们精确指定 FourCC 像素格式
    if (drmModeAddFB2(drm_fd_, width, height, DRM_FORMAT_ARGB8888, 
                      handles, pitches, offsets, out_fb_id, 0) < 0) {
        std::cerr << "[ERROR] Overlay Framebuffer 注册失败！" << std::endl;
        return nullptr;
    }

    // 3. 映射到虚拟内存 (因为 LVGL 是靠 CPU 渲染 UI 的，它需要虚拟地址指针)
    drm_mode_map_dumb map_req = {};
    map_req.handle = create_req.handle;
    drmIoctl(drm_fd_, DRM_IOCTL_MODE_MAP_DUMB, &map_req);
    
    void* map_ptr = mmap(0, create_req.size, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd_, map_req.offset);
    if (map_ptr == MAP_FAILED) return nullptr;

    // 4. 把这块显存全部刷成全透明 (Alpha = 0)
    memset(map_ptr, 0, create_req.size);

    // 5. 寻找并绑定硬件 Overlay Plane
    drmModePlaneRes *plane_res = drmModeGetPlaneResources(drm_fd_);
    bool plane_found = false;

    for (uint32_t i = 0; i < plane_res->count_planes; i++) {
        drmModePlane *plane = drmModeGetPlane(drm_fd_, plane_res->planes[i]);
        // 检查该图层是否能连到我们的屏幕 (CRTC)
        if (plane->possible_crtcs & (1 << crtc_index_)) {
            // 检查该图层是否支持 ARGB8888
            for (uint32_t j = 0; j < plane->count_formats; j++) {
                if (plane->formats[j] == DRM_FORMAT_ARGB8888) {
                    overlay_plane_id_ = plane->plane_id;
                    plane_found = true;
                    break;
                }
            }
        }
        drmModeFreePlane(plane);
        if (plane_found) break;
    }
    drmModeFreePlaneResources(plane_res);

    if (!plane_found) {
        std::cerr << "[ERROR] 找不到支持 ARGB8888 的硬件图层！" << std::endl;
        return nullptr;
    }

    // 6. 见证奇迹的时刻：把这块透明显存强行拍到屏幕的最顶层
    // 注意：DRM API 中，源图像的坐标必须左移 16 位 (Q16.16 格式)
    if (drmModeSetPlane(drm_fd_, overlay_plane_id_, crtc_id_, *out_fb_id, 0,
                        0, 0, width, height, 
                        0, 0, width << 16, height << 16) < 0) {
        std::cerr << "[ERROR] 顶层 Overlay 绑定失败！" << std::endl;
        return nullptr;
    }

    std::cout << "[INFO] 透明 UI 硬件图层开辟成功！Plane ID: " << overlay_plane_id_ << std::endl;
    return map_ptr; // 把这块画布的指针交出去
}
