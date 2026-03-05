// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "image_enc.h"
#include "rkllm.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <thread>  // 提供多线程支持
#include <mutex>   // 提供互斥锁，防止读写冲突
#include <atomic>  // 提供原子变量，用于安全地控制线程退出
#include <csignal>
#include "im2d.h"
#include "rga.h"
#include <queue>
#include <condition_variable>
#define VIDEO_NODE "/dev/video16"
#define IMG_WIDTH  1920
#define IMG_HEIGHT 1080
using namespace std;
LLMHandle llmHandle = nullptr;
// -------------------- 新增：DMA-BUF 与多线程控制 --------------------
#define BUFFER_COUNT 8 // 申请 4 个 Buffer，保证流水线不卡死
struct CameraBuffer {
    int index;           // Buffer 序号
    int fd;              // 导出的 DMA-BUF 文件描述符
    void* start;         // 依然保留虚拟地址映射（以备不时之需或调试）
    size_t length;
    bool in_use_by_llm;  // 核心标志位：是否正在被大模型占用
};

struct CameraState {
    int fd;
    CameraBuffer buffers[BUFFER_COUNT];
};

struct CameraState CS;

std::mutex camera_mutex;              // 互斥锁
int latest_buf_index = -1;            // 共享变量：永远存放最新一帧的 index（不再是 cv::Mat）
int last_held_index=-1;
std::atomic<bool> keep_running{true}; // 线程开关
std::thread* camera_thread = nullptr;
// -------------------- 新增：视觉特征共享区 --------------------
std::mutex vision_mutex;                  // 保护视觉特征的互斥锁
std::vector<float> global_img_embed;      // 全局特征向量（存放 run_imgenc 的结果）
bool is_vision_ready = false;             // 标志位：第一帧是否已经处理完毕
std::atomic<bool> vision_thread_running{true}; // 视觉线程的开关
std::thread* vision_thread = nullptr;     // 视觉线程指针
std::atomic<bool> is_llm_generating{false}; // 新增：标记 LLM 是否正在说话
// --------------------------------------------------------------
// -------------------- 新增：任务队列与通信组件 --------------------
std::queue<std::string> prompt_queue;     // 存放用户输入的任务队列
std::mutex queue_mutex;                   // 保护队列的互斥锁
std::condition_variable queue_cv;         // 条件变量：用于唤醒 LLM 线程
std::thread* llm_thread = nullptr;        // LLM 推理线程指针
// ------------------------------------------------------------------
//-----------------------------------------multi_thread------------------------------------------------

bool bind_thread_to_cpus(int start_core, int end_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    for (int i = start_core; i <= end_core; ++i) {
        CPU_SET(i, &cpuset);
    }

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    if (rc != 0) {
        std::cerr << "[Warning] Error calling pthread_setaffinity_np: " << rc << std::endl;
        return false;
    }
    
    std::cout << "[INFO] 成功将线程绑定到 CPU " << start_core << " ~ " << end_core << std::endl;
    return true;
}
//-----------------------------------------camera--------------------------------





bool init_camera() {
    CS.fd = open(VIDEO_NODE, O_RDWR); // 直接存入 CS.fd
    if (CS.fd < 0) {
        perror("[ERROR] Failed to open video node");
        return false;
    }

    // 2. Set Format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = IMG_WIDTH;
    fmt.fmt.pix_mp.height = IMG_HEIGHT;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;

    if (ioctl(CS.fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("[ERROR] VIDIOC_S_FMT failed");
        close(CS.fd);
        return false;
    }

    
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = BUFFER_COUNT; // 【改动1】从 1 改成 BUFFER_COUNT (也就是 4)
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(CS.fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("[ERROR] VIDIOC_REQBUFS failed");
        close(CS.fd);
        return false;
    }
    // 4. Query and Map Buffer
    // 4. 遍历这 4 个 Buffer：查询属性、导出 FD、映射内存、交还给底层
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        struct v4l2_plane planes[1];
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.length = 1;
        buf.m.planes = planes;

        // 查询第 i 个 Buffer 的信息
        if (ioctl(CS.fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("[ERROR] VIDIOC_QUERYBUF failed");
            return false;
        }

        // 【改动2：最核心的一步！】导出 DMA-BUF 文件描述符 (FD)
        struct v4l2_exportbuffer expbuf;
        memset(&expbuf, 0, sizeof(expbuf));
        expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        expbuf.index = i;
        expbuf.plane = 0; // NV12 往往只有一个 Plane
        
        if (ioctl(CS.fd, VIDIOC_EXPBUF, &expbuf) == 0) {
            // 成功拿到了宝贵的硬件 FD，存入我们的全局结构体中！
            CS.buffers[i].fd = expbuf.fd;
            std::cout << "[INFO] 成功导出 Buffer " << i << " 的 DMA FD: " << expbuf.fd << std::endl;
        } else {
            perror("[ERROR] VIDIOC_EXPBUF failed");
            return false;
        }

        // 记录其他信息
        CS.buffers[i].index = i;
        CS.buffers[i].length = buf.m.planes[0].length;
        CS.buffers[i].in_use_by_llm = false; // 初始状态：大模型没有在用它

        // 虽然我们要用 FD，但也同时把它映射成虚拟地址，方便后续调试
        CS.buffers[i].start = mmap(NULL, buf.m.planes[0].length, PROT_READ | PROT_WRITE, 
                                   MAP_SHARED, CS.fd, buf.m.planes[0].m.mem_offset);

        // QBUF: 把空 Buffer 正式塞回给底层驱动，让它准备拍照装填数据
        if (ioctl(CS.fd, VIDIOC_QBUF, &buf) < 0) {
            perror("[ERROR] VIDIOC_QBUF failed");
            return false;
        }
    }
    

    // 5. Stream on
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(CS.fd, VIDIOC_STREAMON, &type) < 0) {
        perror("[ERROR] VIDIOC_STREAMON failed");
        return false; 
    }

    std::cout << "[INFO] Camera initialized successfully!" << std::endl;
    return true; // 修复：别忘了成功时返回 true
}
// 后台抓图线程函数 (生产者)
void camera_thread_func() {
      int rc = pthread_setname_np(pthread_self(), "camera"); // <= 15 chars
    if (rc != 0) {
        std::cerr << "pthread_setname_np failed: " << std::strerror(rc) << "\n";
    }
    bind_thread_to_cpus(0, 3);
    std::cout << "[INFO] 后台抓图线程已启动..." << std::endl;
    
    while (keep_running) {
        
        // 1. DQBUF (从底层硬件拿到装满画面的 buffer)
        struct v4l2_plane planes[1];
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        buf.length = 1;
        buf.m.planes = planes;
        if (ioctl(CS.fd, VIDIOC_DQBUF, &buf) < 0) {
            // 如果没拿到，稍微等一下继续尝试，防止死循环占满 CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue; 
        }
        int new_index=buf.index;
        // 2. 加锁，更新全局图片
        {
            std::lock_guard<std::mutex> lock(camera_mutex); 
            if(last_held_index!=-1&&!CS.buffers[last_held_index].in_use_by_llm){
                struct v4l2_plane qplanes[1];
                struct v4l2_buffer qbuf;
                memset(&qbuf, 0, sizeof(qbuf));
                memset(qplanes, 0, sizeof(qplanes));
                qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                qbuf.memory = V4L2_MEMORY_MMAP;
                qbuf.index=last_held_index;
                qbuf.length = 1;
                qbuf.m.planes = qplanes;
                if (ioctl(CS.fd, VIDIOC_QBUF, &qbuf) < 0) {
                    // 如果没拿到，稍微等一下继续尝试，防止死循环占满 CPU
                    perror("[ERROR] VIDIOC_QBUF failed in thread");
                }
            }
            last_held_index=new_index;
            latest_buf_index=new_index;
        }
       
    }
    std::cout << "[INFO] 后台抓图线程已安全退出。" << std::endl;
}
void release_camera() {
    // 1. 停止视频流
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(CS.fd, VIDIOC_STREAMOFF, &type);

    // 2. 遍历释放我们申请的 4 个 Buffer
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        // 解除虚拟地址映射
        if (CS.buffers[i].start != NULL && CS.buffers[i].start != MAP_FAILED) {
            munmap(CS.buffers[i].start, CS.buffers[i].length);
        }
        
        // 【极其重要】关闭导出的 DMA-BUF 文件描述符！防止内存和句柄泄露！
        if (CS.buffers[i].fd > 0) {
            close(CS.buffers[i].fd);
        }
    }

    // 3. 关闭摄像头设备节点
    if (CS.fd > 0) {
        close(CS.fd);
    }
    
    std::cout << "[INFO] Camera resources and DMA-BUFs released safely." << std::endl;
}
//-----------------------------------------camera-end----------------------------
// 视觉处理线程：在后台默默把最新的画面转成 LLM 特征
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
            // ==================== 探针 1：保存 RGA 处理前（摄像头原始 NV12） ====================
            // 你的 init_camera 里已经把 DMA-BUF 映射到了虚拟内存 CS.buffers[i].start
            // NV12 格式在内存中占用的大小是 height * width * 1.5 (即 height * 3 / 2)
            cv::Mat yuv_mat(IMG_HEIGHT * 3 / 2, IMG_WIDTH, CV_8UC1, CS.buffers[process_id].start);
            cv::Mat src_bgr;
            // 将 NV12 转为 OpenCV 认识的 BGR 格式用于正常保存和预览
            cv::cvtColor(yuv_mat, src_bgr, cv::COLOR_YUV2BGR_NV12);

            auto now = std::chrono::system_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::time_t t = std::chrono::system_clock::to_time_t(now);
            std::tm tm = *std::localtime(&t);
            char time_buf[64];
            std::strftime(time_buf, sizeof(time_buf), "%H-%M-%S", &tm);
            std::string time_str = std::string(time_buf) + "_" + std::to_string(ms.count());

            std::string before_name = "debug_before_" + time_str + ".jpg";
            cv::imwrite(before_name, src_bgr);
            // ==============================================================================
            memset(rga_buf.data(), 127, aligned_w * image_height * 3);


            rga_buffer_t src = wrapbuffer_fd(process_fd, IMG_WIDTH, IMG_HEIGHT, RK_FORMAT_YCbCr_420_SP);
            rga_buffer_t dst = wrapbuffer_virtualaddr((void*)rga_buf.data(), image_width, image_height, RK_FORMAT_RGB_888, aligned_w, image_height);
           

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

void llm_worker_func(rknn_app_context_t* app_ctx) {
    // 强制绑定到 A76 大核 (例如 6, 7)，榨干大核算力
    bind_thread_to_cpus(6, 7);
    std::cout << "[INFO] LLM 推理线程已启动，进入深度睡眠等待任务..." << std::endl;

    size_t n_image_tokens = app_ctx->model_image_token;
    size_t image_width = app_ctx->model_width;
    size_t image_height = app_ctx->model_height;

    // 线程内维护一份自己的特征向量副本来防止竞争
    std::vector<float> local_img_vec;

    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;

    while (keep_running) {
        std::string current_prompt;

        // 1. 等待任务：无锁时绝对睡眠，0% CPU 占用
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // 铃铛没响，且队列为空时，死死睡住
            queue_cv.wait(lock, []{ return !prompt_queue.empty() || !keep_running; });
            
            if (!keep_running && prompt_queue.empty()) {
                break; // 收到退出信号，安全结束线程
            }

            // 拿到任务，出队
            current_prompt = prompt_queue.front();
            prompt_queue.pop();
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        // 2. 解析任务并准备输入
        if (current_prompt.find("<image>") != std::string::npos) {
            auto t_feat_start = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(vision_mutex);
                if (!is_vision_ready) {
                    std::cout << "[WARNING] 后台视觉特征尚未初始化，跳过此次多模态推理！" << std::endl;
                    std::cout << "\nuser: " << std::flush;
                    continue; 
                }
                local_img_vec = global_img_embed; // 极速深拷贝（微秒级）
            }
            auto feat_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_feat_start).count();
            std::cout << " -> 从后台获取最新视觉特征耗时: " << feat_cost << " us" << std::endl;

            rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
            rkllm_input.multimodal_input.prompt = (char*)current_prompt.c_str();
            rkllm_input.multimodal_input.image_embed = local_img_vec.data();
            rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
            rkllm_input.multimodal_input.n_image = 1;
            rkllm_input.multimodal_input.image_height = image_height;
            rkllm_input.multimodal_input.image_width = image_width;
        } else {
            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.prompt_input = (char*)current_prompt.c_str();
        }
        rkllm_input.role = "user";

        // 3. 执行核心推理
        printf("robot: ");
        
        is_llm_generating = true; // 【极其关键】抢占 NPU，让视觉线程挂起闭嘴
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
        is_llm_generating = false; // 推理结束，松开刹车，视觉线程恢复常态感知

        auto t_end = std::chrono::high_resolution_clock::now();
        auto cost_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        std::cout << "\n[性能打点] 推理总耗时: " << cost_time_ms << " ms" << std::endl;
        
        // 打印出新的交互提示符，因为主线程可能早就等在那里了
        std::cout << "\nuser: " << std::flush; 
    }
    std::cout << "[INFO] LLM 推理线程已退出。" << std::endl;
}
void exit_handler(int signal)
{
    std::cout << "\n[INFO] 接收到退出信号 (比如 Ctrl+C)，准备清理现场..." << std::endl;

    // 1. 告诉后台抓图线程：别抓了，准备下班！
    keep_running = false; 
    
    // 2. 等待后台线程安全结束当前循环
    if (camera_thread && camera_thread->joinable()) {
        camera_thread->join(); // 阻塞在这里，直到线程函数执行完毕
        delete camera_thread;
        camera_thread = nullptr;
    }

    // 3. 释放摄像头 (极其重要，防止设备被锁死)
    release_camera();

    // 4. 释放大模型
    if (llmHandle != nullptr)
    {
        std::cout << "[INFO] 释放大模型资源..." << std::endl;
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
    }

    std::cout << "[INFO] 清理完毕，程序体面退出。" << std::endl;
    exit(signal);
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state)
{

    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        printf("\\run error\n");
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        printf("%s", result->text);
        // for(int i=0; i<result->num; i++)
        // {
        //     printf("%d token_id: %d logprob: %f\n", i, result->tokens[i].id, result->tokens[i].logprob);
        // }
    }
    return 0;
}

// Expand the image into a square and fill it with the specified background color
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

int main(int argc, char** argv)
{
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                << " image_path encoder_model_path llm_model_path max_new_tokens max_context_len rknn_core_num "
                << "[img_start] [img_end] [img_content]\n";
        return -1;
    }
    if(!init_camera()){
        return -1;
    }
    //std::signal(SIGINT, exit_handler);
    // --- 新增：启动后台抓图线程 ---
    bind_thread_to_cpus(4, 7);
    std::cout << "[INFO] 锁定 CPU 为最高性能模式 (performance)..." << std::endl;
    system("echo performance | tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor > /dev/null");
    camera_thread = new std::thread(camera_thread_func);
    // 等待一下，确保后台线程能抓到第一张图，避免主线程跑太快拿到空图
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const char * image_path = argv[1];
    const char * encoder_model_path = argv[2];

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[3];
    param.top_k = 1;
    param.max_new_tokens = std::atoi(argv[4]);
    param.max_context_len = std::atoi(argv[5]);
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;

    param.img_start   = "<|vision_start|>";
    param.img_end     = "<|vision_end|>";
    param.img_content = "<|image_pad|>";

    if (argc == 7) {
        std::cerr << "[Warning] Using default img_start/img_end/img_content: "
                << param.img_start << " , "
                << param.img_end << " , "
                << param.img_content
                << ". Please customize these values according to your model, "
                << "otherwise the output may be incorrect.\n";
    }

    if (argc > 7) param.img_start   = argv[7];
    if (argc > 8) param.img_end     = argv[8];
    if (argc > 9) param.img_content = argv[9];

    int ret;
    std::chrono::high_resolution_clock::time_point t_start_us = std::chrono::high_resolution_clock::now();

    ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }
    std::chrono::high_resolution_clock::time_point t_load_end_us = std::chrono::high_resolution_clock::now();

    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: LLM Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    t_start_us = std::chrono::high_resolution_clock::now();

    const int core_num = atoi(argv[6]);
    ret = init_imgenc(encoder_model_path, &rknn_app_ctx, core_num);
    if (ret != 0) {
        printf("init_imgenc fail! ret=%d model_path=%s\n", ret, encoder_model_path);
        return -1;
    }
    // -------------------- 新增：启动视觉处理线程 --------------------
    vision_thread = new std::thread(vision_worker_func, &rknn_app_ctx);
    llm_thread = new std::thread(llm_worker_func, &rknn_app_ctx);
    t_load_end_us = std::chrono::high_resolution_clock::now();

    load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: ImgEnc Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);


    size_t n_image_tokens = rknn_app_ctx.model_image_token;
    size_t image_embed_len = rknn_app_ctx.model_embed_size;
    int rkllm_image_embed_len = n_image_tokens * image_embed_len;
    //float img_vec[rkllm_image_embed_len];
    std::vector<float> img_vec(rkllm_image_embed_len);
    
    
    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));

    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;
    // rkllm_set_chat_template(llmHandle, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");

    vector<string> pre_input;
    pre_input.push_back("<image>What is in the image?");
    pre_input.push_back("<image>这幅画面里展示了什么样的场景?");
    cout << "\n**********************可输入以下问题对应序号获取回答/或自定义输入********************\n"
         << endl;
    for (int i = 0; i < (int)pre_input.size(); i++)
    {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "\n*************************************************************************\n"
         << endl;

    while(true) {
        std::string input_str;
        std::getline(std::cin, input_str);
        
        if (input_str == "exit") {
            break; // 触发退出流程
        }
        if (input_str == "clear") {
            ret = rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
            if (ret != 0) printf("clear kv cache failed!\n");
            printf("user: ");
            continue;
        }

        // 处理快捷数字输入
        for (int i = 0; i < (int)pre_input.size(); i++) {
            if (input_str == to_string(i)) {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }

        // 将任务推入队列，并敲响铃铛唤醒 LLM 线程！
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            prompt_queue.push(input_str);
        }
        queue_cv.notify_one(); 

        // 注意：这里不需要再打 "user: " 了，因为大模型还没说完，打印标志位的事情交给 llm_worker 去做。
    }

    
    // --- 新增：安全关闭所有后台线程 ---
    
    keep_running = false; 
    vision_thread_running = false; 
    queue_cv.notify_all(); 

    // ---------------------------------------------------------
    // 第二步：等待所有打工人（子线程）安全下班
    // ---------------------------------------------------------
    if (camera_thread && camera_thread->joinable()) {
        camera_thread->join(); 
        delete camera_thread;
        camera_thread = nullptr;
        std::cout << "[清理] 摄像头抓图线程已回收。" << std::endl;
    }

    if (vision_thread && vision_thread->joinable()) {
        vision_thread->join(); 
        delete vision_thread;
        vision_thread = nullptr;
        std::cout << "[清理] 视觉预热线程已回收。" << std::endl;
    }

    if (llm_thread && llm_thread->joinable()) {
        llm_thread->join(); 
        delete llm_thread;
        llm_thread = nullptr;
        std::cout << "[清理] LLM 推理线程已回收。" << std::endl;
    }

    // ---------------------------------------------------------
    // 第三步：人走空了，可以安全地拆厂房（释放底层驱动资源）
    // ---------------------------------------------------------
    std::cout << "[INFO] 恢复 CPU 动态调频模式 (schedutil)..." << std::endl;
    system("echo schedutil | tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor > /dev/null");
    
    // 把 release_imgenc 移到了这里！此时 vision_thread 已经死透了，绝对安全
    ret = release_imgenc(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_imgenc fail! ret=%d\n", ret);
    }
    
    rkllm_destroy(llmHandle);
    release_camera();

    return 0;
}