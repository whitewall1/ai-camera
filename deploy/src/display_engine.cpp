#include "display_engine.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>
// 在 display_worker.cpp 顶部定义全局变量
uint32_t g_drm_fb_ids[2] = {0};
void* g_drm_map_ptrs[2] = {nullptr};
int      g_drm_fd = -1;
uint32_t g_overlay_plane_id = 0;
uint32_t g_crtc_id = 0;
uint32_t g_screen_width = 1024;
uint32_t g_screen_height = 600;
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
    g_drm_fd=drm_fd_;
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
    g_crtc_id=crtc_id_;
    drmModeFreeConnector(conn);
    drmModeFreeResources(res);

    return crtc_id_ != 0;
}


// 注意：假设你已经在全局或头文件中定义了这两个数组
// extern uint32_t g_drm_fb_ids[2];
// extern void* g_drm_map_ptrs[2];

bool DisplayEngine::create_overlay_plane() { // 返回值可以改成 bool，因为指针已经是全局的了
    if (drm_fd_ < 0 || crtc_id_ == 0) return false;

    uint32_t width = display_mode_.hdisplay;
    uint32_t height = display_mode_.vdisplay;

    // --- 核心改造：循环申请两块 Dumb Buffer ---
    for (int i = 0; i < 2; i++) {
        // 1. 申请 Dumb Buffer (bpp=32, 给 ARGB 留足空间)
        drm_mode_create_dumb create_req = {};
        create_req.width = width;
        create_req.height = height;
        create_req.bpp = 32;
        if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create_req) < 0) {
            std::cerr << "[ERROR] Overlay 显存 " << i << " 申请失败！" << std::endl;
            return false;
        }

        // 2. 注册为 Framebuffer，强制指定格式为 ARGB8888
        uint32_t handles[4] = {create_req.handle, 0, 0, 0};
        uint32_t pitches[4] = {create_req.pitch, 0, 0, 0};
        uint32_t offsets[4] = {0, 0, 0, 0};
        
        if (drmModeAddFB2(drm_fd_, width, height, DRM_FORMAT_ARGB8888, 
                          handles, pitches, offsets, &g_drm_fb_ids[i], 0) < 0) {
            std::cerr << "[ERROR] Overlay FB " << i << " 注册失败！" << std::endl;
            return false;
        }

        // 3. 向内核索要 Mmap 的 offset 偏移量（极其关键！）
        drm_mode_map_dumb map_req = {};
        map_req.handle = create_req.handle;
        if (drmIoctl(drm_fd_, DRM_IOCTL_MODE_MAP_DUMB, &map_req) < 0) {
            std::cerr << "[ERROR] 获取 Mmap Offset 失败！" << std::endl;
            return false;
        }

        // 4. 映射到虚拟内存
        g_drm_map_ptrs[i] = mmap(0, create_req.size, PROT_READ | PROT_WRITE, 
                                 MAP_SHARED, drm_fd_, map_req.offset);
        if (g_drm_map_ptrs[i] == MAP_FAILED) {
            std::cerr << "[ERROR] 显存 Mmap 失败！" << std::endl;
            return false;
        }

        // 5. 将这块显存全部刷成全透明 (Alpha = 0)
        memset(g_drm_map_ptrs[i], 0, create_req.size);
    }

    // --- 寻找并绑定硬件 Overlay Plane ---
    drmModePlaneRes *plane_res = drmModeGetPlaneResources(drm_fd_);
    bool plane_found = false;

    for (uint32_t i = 0; i < plane_res->count_planes; i++) {
        drmModePlane *plane = drmModeGetPlane(drm_fd_, plane_res->planes[i]);
        if (plane->possible_crtcs & (1 << crtc_index_)) {
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
        return false;
    }

    // 6. 初始状态：把第 0 块显存 (g_drm_fb_ids[0]) 拍到屏幕上
    if (drmModeSetPlane(drm_fd_, overlay_plane_id_, crtc_id_, g_drm_fb_ids[0], 0,
                        0, 0, width, height, 
                        0, 0, width << 16, height << 16) < 0) {
        std::cerr << "[ERROR] 顶层 Overlay 绑定失败！" << std::endl;
        return false;
    }
    g_overlay_plane_id=overlay_plane_id_;
    
    std::cout << "[INFO] 双缓冲透明 UI 硬件图层开辟成功！" << std::endl;
    return true; 
}
