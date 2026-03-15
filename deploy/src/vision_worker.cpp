#include "vision_worker.hpp"
#include "globals.hpp" // 获取 CS, camera_mutex, vision_mutex, global_img_embed, is_vision_ready 等
#include "utils.hpp"   // 获取 bind_thread_to_cpus

#include <iostream>
#include <thread>
#include <chrono>
#include <sys/ioctl.h>
#include <linux/videodev2.h>

// RGA 相关头文件
#include "im2d.h"
#include "rga.h"
cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color) {
    int width = img.cols;
    int height = img.rows;

    // If the width and height are equal, return to the original image directly
    if (width == height) {
        return img.clone();
    }

    // Calculate the new size and create a new image
    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    // Calculate the image paste position
    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    // Paste the original image into the center of the new image
    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}


void vision_worker_func(rknn_app_context_t* app_ctx) {
    // 建议绑定到 NPU 所在的相同集群，或者留给系统的调度器
    bind_thread_to_cpus(4, 5); 
    std::cout << "[INFO] 视觉处理线程已启动，开始后台特征提取..." << std::endl;
// ---------------------------------------------------------
    // 优化 1：把所有【常量】和【内存分配】移出 while 循环！
    // ---------------------------------------------------------
    size_t n_image_tokens = app_ctx->model_image_token;
    size_t image_embed_len = app_ctx->model_embed_size;
    int rkllm_image_embed_len = n_image_tokens * image_embed_len;
    
    size_t image_width = app_ctx->model_width;   // 392
    size_t image_height = app_ctx->model_height; // 392
    int aligned_w = (image_width + 15) & (~15);  // 400

    // 提前算好缩放和 ROI，只算一次
    int max_dim = std::max(IMG_WIDTH, IMG_HEIGHT);
    float scale = (float)image_width / max_dim; 
    int scaled_w = IMG_WIDTH * scale;
    int scaled_h = IMG_HEIGHT * scale;
    int dx = (image_width - scaled_w) / 2;
    int dy = (image_height - scaled_h) / 2;

    im_rect src_rect = {0, 0, IMG_WIDTH, IMG_HEIGHT}; 
    im_rect dst_rect = {dx, dy, scaled_w, scaled_h};  
    rga_buffer_t empty_pat;
    memset(&empty_pat, 0, sizeof(empty_pat));
    im_rect empty_rect = {0, 0, 0, 0};

    // 提前分配好所有的物理/虚拟内存块 (整个线程生命周期只 new 这一次)
   
    std::vector<uint8_t> rga_buf(aligned_w * image_height * 3);
    
    // 预先包好 Mat 头，不分配新内存
    cv::Mat rga_mat_wrapper(image_height, aligned_w, CV_8UC3, rga_buf.data());
    // 预分配一块连续紧凑的内存，给 NPU 喂数据用
    cv::Mat packed_mat(image_height, image_width, CV_8UC3);
    cv::Mat last_packed_mat; // 用来保存上一帧的原始像素，用于比对
    // 在 while 循环外面定义
    auto last_npu_time = std::chrono::steady_clock::now();
 
    while (vision_thread_running && keep_running) {
        if (is_llm_generating) {
            // 如果 LLM 正在疯狂输出文字，我们就休眠，把算力和内存带宽全让给它！
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue; 
        }
        int process_id = -1;
        int process_fd = -1;

        // 1. 从摄像头获取最新的帧
        {
            std::lock_guard<std::mutex> lock(camera_mutex);
            if (latest_buf_index != -1) {
                process_id = latest_buf_index;
                process_fd = CS.buffers[latest_buf_index].fd;
                CS.buffers[latest_buf_index].in_use_by_llm = true; // 锁定这帧画面
            }
        }

        if (process_id != -1) {
            memset(rga_buf.data(), 127, aligned_w * image_height * 3);


            rga_buffer_t src = wrapbuffer_fd(process_fd, IMG_WIDTH, IMG_HEIGHT, RK_FORMAT_YCbCr_420_SP);
            rga_buffer_t dst = wrapbuffer_virtualaddr((void*)rga_buf.data(), image_width, image_height, RK_FORMAT_RGB_888, (int)aligned_w, (int)image_height);
           
            
            // 执行 RGA
            IM_STATUS rga_stat = improcess(src, dst, empty_pat, src_rect, dst_rect, empty_rect, 0);
            if (rga_stat != IM_STATUS_SUCCESS) {
                printf("[ERROR] RGA improcess failed: %s\n", imStrError(rga_stat));
            }
           rga_mat_wrapper(cv::Rect(0, 0, image_width, image_height)).copyTo(packed_mat);
           
           
            // 释放这帧摄像头 Buffer (重要！)
            {
                std::lock_guard<std::mutex> lock(camera_mutex);
                CS.buffers[process_id].in_use_by_llm = false;
                struct v4l2_buffer qbuf;
                struct v4l2_plane qplanes[1];
                memset(&qbuf, 0, sizeof(qbuf));
                memset(qplanes, 0, sizeof(qplanes));
                qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                qbuf.memory = V4L2_MEMORY_MMAP;
                qbuf.index = process_id;
                qbuf.length = 1;
                qbuf.m.planes = qplanes;
                
                if(last_held_index == process_id) last_held_index = -1;
                ioctl(CS.fd, VIDIOC_QBUF, &qbuf);
            }
            bool should_run_npu = false;
            std::string trigger_reason = "";

            auto now = std::chrono::steady_clock::now();
            auto seconds_since_last_npu = std::chrono::duration_cast<std::chrono::seconds>(now - last_npu_time).count();
            
            if (!last_packed_mat.empty()) {
                // 方案 A：像素阈值占比
                cv::Mat diff, gray_diff, thresh;
                // 1. 求绝对差值
                cv::absdiff(packed_mat, last_packed_mat, diff);
                // 2. 转灰度图，降低计算量
                cv::cvtColor(diff, gray_diff, cv::COLOR_BGR2GRAY);
                // 3. 核心：二值化过滤噪点！像素差值大于 30（满分255）才认为是真变化，否则设为 0
                cv::threshold(gray_diff, thresh, 30, 255, cv::THRESH_BINARY);

                // 4. 统计真正发生了变化的像素点数量
                int changed_pixels = cv::countNonZero(thresh);
                float changed_ratio = (float)changed_pixels / (image_width * image_height);

                // 如果画面有超过 3% 的区域发生实质性变化（捕捉人脸微表情、手势）
                if (changed_ratio > 0.03f) {
                    should_run_npu = true;
                    trigger_reason = "局部动作触发 (变化率: " + std::to_string(changed_ratio * 100) + "%)";
                } 
                // 方案 C：心跳兜底机制
                else if (seconds_since_last_npu >= 3) {
                    should_run_npu = true;
                    trigger_reason = "心跳兜底触发 (已超 3 秒)";
                }
            } else {
                // 第一帧绝对要跑
                should_run_npu = true;
                trigger_reason = "系统初始化首帧";
            }
           
            if (should_run_npu) {
                std::cout << "\n[VISION] 唤醒 NPU! 原因: " << trigger_reason  << std::endl;

                // ... 执行 run_imgenc (耗时 3.5s) ...
                // ... 更新 global_img_embed 和 is_vision_ready = true ...
                int ret = run_imgenc(app_ctx, packed_mat.data);
                if (ret != 0) {
                    printf("[ERROR] run_imgenc fail! ret=%d\n", ret);
                }
                // 3. 把算好的特征放入“保险箱”
                {
                    std::lock_guard<std::mutex> lock(vision_mutex);
                    
                    is_vision_ready = true;
                }
                // 重置心跳计时器和上一帧画面
                last_npu_time = std::chrono::steady_clock::now();
                packed_mat.copyTo(last_packed_mat);
            } else {
                // 画面极其静止，且不到兜底时间，继续省电
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            
            
        } else {
            // 如果没抓到图，稍微等一下
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    std::cout << "[INFO] 视觉处理线程已退出。" << std::endl;
}