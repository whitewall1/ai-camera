#include "display_worker.hpp"
#include "display_engine.hpp" // 引入底层的 DRM 引擎工具
#include "globals.hpp"        // 引入 CS, camera_mutex, latest_buf_index, keep_running
#include "utils.hpp"          // 引入 bind_thread_to_cpus
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include "lvgl_worker.hpp"
// RGA 相关头文件
#include "im2d.h"
#include "rga.h"
void display_worker_func() {
    bind_thread_to_cpus(2, 3); // 绑定到小核，把大核全留给模型
    //std::cout << "[INFO] DRM 显示线程已启动，开启 RGA 硬件直通..." << std::endl;

    DisplayEngine display;
    if (!display.init()) {
        std::cerr << "[ERROR] DRM 初始化失败，退出显示线程。" << std::endl;
        return;
    }

    // 向系统申请一块屏幕专属的物理显存
    uint32_t display_fb_id = 0;
    int display_dma_fd = display.create_dumb_buffer_fd(&display_fb_id);
    if (display_dma_fd < 0) {
        std::cerr << "[ERROR] 获取 DRM 显存 FD 失败。" << std::endl;
        return;
    }
     display.present_fb(display_fb_id); 
    //std::cout << "[INFO] 物理图层绑定完毕，开启零拷贝自动刷新！" << std::endl;
    // 预先算好 RGA 的源和目标分辨率矩形
    im_rect src_rect = {0, 0, IMG_WIDTH, IMG_HEIGHT}; 
    im_rect dst_rect = {0, 0, 1024, 600}; // 你的 MIPI 屏幕真实分辨率
    
    // 构造空参数以满足 RGA API 要求
    im_rect empty_rect = {0, 0, 0, 0};
    rga_buffer_t empty_pat;
    memset(&empty_pat, 0, sizeof(empty_pat));
    uint32_t overlay_fb_id = 0;
    void* overlay_map_ptr = display.create_overlay_plane(&overlay_fb_id);

    if (!overlay_map_ptr) {
        std::cerr << "[ERROR] 无法开辟透明 UI 图层，LVGL 启动失败！" << std::endl;
    } else {
        // 把这块隐形玻璃的画笔 (overlay_map_ptr) 交给 LVGL 线程
        std::thread lvgl_thread(lvgl_worker_func, overlay_map_ptr, 1024, 600);
        lvgl_thread.detach(); // 让 UI 线程在后台独立运转
    }
    while (keep_running) {
        int process_fd = -1;
        // 极速获取最新的一帧
        {
            std::lock_guard<std::mutex> lock(camera_mutex);
            if (latest_buf_index != -1) {
                process_fd = CS.buffers[latest_buf_index].fd;
            }
        }

        if (process_fd != -1) {
            // RGA 零拷贝核心战役：将 V4L2 的 NV12 转换并缩放，直接写入 DRM 显存 (BGRA_8888 对应 XRGB 内存布局)
            rga_buffer_t src = wrapbuffer_fd(process_fd, IMG_WIDTH, IMG_HEIGHT, RK_FORMAT_YCbCr_420_SP);
            rga_buffer_t dst = wrapbuffer_fd(display_dma_fd, 1024, 600, RK_FORMAT_BGRA_8888);
            
            // 补齐 7 个参数
            IM_STATUS rga_stat = improcess(src, dst, empty_pat, src_rect, dst_rect, empty_rect, 0);
            
           
        }
        
        // 限制刷新率为 60fps 左右，防止吃满总线
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    std::cout << "[INFO] 显示线程已退出。" << std::endl;
}