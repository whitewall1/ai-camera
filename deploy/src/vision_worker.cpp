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
    std::vector<float> local_img_vec(rkllm_image_embed_len);
    std::vector<uint8_t> rga_buf(aligned_w * image_height * 3);
    
    // 预先包好 Mat 头，不分配新内存
    cv::Mat rga_mat_wrapper(image_height, aligned_w, CV_8UC3, rga_buf.data());
    // 预分配一块连续紧凑的内存，给 NPU 喂数据用
    cv::Mat packed_mat(image_height, image_width, CV_8UC3);

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
            // // ==================== 探针 1：保存 RGA 处理前（摄像头原始 NV12） ====================
            // // 你的 init_camera 里已经把 DMA-BUF 映射到了虚拟内存 CS.buffers[i].start
            // // NV12 格式在内存中占用的大小是 height * width * 1.5 (即 height * 3 / 2)
            // cv::Mat yuv_mat(IMG_HEIGHT * 3 / 2, IMG_WIDTH, CV_8UC1, CS.buffers[process_id].start);
            // cv::Mat src_bgr;
            // // 将 NV12 转为 OpenCV 认识的 BGR 格式用于正常保存和预览
            // cv::cvtColor(yuv_mat, src_bgr, cv::COLOR_YUV2BGR_NV12);

            // auto now = std::chrono::system_clock::now();
            // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            // std::time_t t = std::chrono::system_clock::to_time_t(now);
            // std::tm tm = *std::localtime(&t);
            // char time_buf[64];
            // std::strftime(time_buf, sizeof(time_buf), "%H-%M-%S", &tm);
            // std::string time_str = std::string(time_buf) + "_" + std::to_string(ms.count());

            // std::string before_name = "debug_before_" + time_str + ".jpg";
            // cv::imwrite(before_name, src_bgr);
            // // ==============================================================================
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

            // 2. 执行 NPU 视觉编码
            int ret = run_imgenc(app_ctx, packed_mat.data, local_img_vec.data());
            if (ret != 0) {
                printf("[ERROR] run_imgenc fail! ret=%d\n", ret);
            }

            // 3. 把算好的特征放入“保险箱”
            {
                std::lock_guard<std::mutex> lock(vision_mutex);
                global_img_embed = local_img_vec; // 拷贝给全局变量
                is_vision_ready = true;
            }
        } else {
            // 如果没抓到图，稍微等一下
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    std::cout << "[INFO] 视觉处理线程已退出。" << std::endl;
}