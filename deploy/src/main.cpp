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
#define VIDEO_NODE "/dev/video9"
#define IMG_WIDTH  1920
#define IMG_HEIGHT 1080
using namespace std;
LLMHandle llmHandle = nullptr;
//-----------------------------------------multi_thread------------------------------------------------
// 多线程全局控制变量
std::mutex camera_mutex;              // 互斥锁：保证同一时间只有一个人能碰图片
cv::Mat latest_nv12_frame;            // 共享内存：永远存放最新的一帧原始 NV12 图片
std::atomic<bool> keep_running{true}; // 线程开关：当它变成 false 时，抓图线程就会乖乖退出
std::thread* camera_thread = nullptr; // 指向我们后台抓图线程的指针
//-----------------------------------------camera--------------------------------
struct CameraState {
    int fd;
    struct v4l2_buffer buf;
    struct v4l2_plane planes[1]; // 修复：必须存在全局结构体里，否则会变成悬空指针
    void* buffer_start;
    unsigned int buffer_length;  // 修复：记录长度，方便 munmap 时使用
};

struct CameraState CS;

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

    // 3. Request Buffers
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(CS.fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("[ERROR] VIDIOC_REQBUFS failed");
        close(CS.fd);
        return false;
    }

    // 4. Query and Map Buffer
    memset(&CS.buf, 0, sizeof(CS.buf));
    memset(CS.planes, 0, sizeof(CS.planes)); // 初始化结构体里的 planes
    CS.buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    CS.buf.memory = V4L2_MEMORY_MMAP;
    CS.buf.index = 0;
    CS.buf.length = 1; 
    CS.buf.m.planes = CS.planes; // 安全绑定！

    if (ioctl(CS.fd, VIDIOC_QUERYBUF, &CS.buf) < 0) {
        perror("[ERROR] VIDIOC_QUERYBUF failed");
        close(CS.fd);
        return false;
    }

    CS.buffer_length = CS.buf.m.planes[0].length;
    CS.buffer_start = mmap(NULL, CS.buffer_length, PROT_READ | PROT_WRITE, 
                           MAP_SHARED, CS.fd, CS.buf.m.planes[0].m.mem_offset);
                           
    if (CS.buffer_start == MAP_FAILED) {
        perror("[ERROR] mmap failed");
        close(CS.fd);
        return false;
    }

    // 重点：开流(STREAMON)之前，先要把缓存(QBUF)交给底层，让它有地方存第一张图！
    if (ioctl(CS.fd, VIDIOC_QBUF, &CS.buf) < 0) {
        perror("[ERROR] VIDIOC_QBUF failed");
        return false;
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

// 修复：返回值改为 cv::Mat
// cv::Mat get_camera_frame() {
//     // 1. 出队 (DQBUF) -> 拿到硬件填满的图像
//     if (ioctl(CS.fd, VIDIOC_DQBUF, &CS.buf) < 0) {
//         perror("[ERROR] VIDIOC_DQBUF failed");
//         return cv::Mat(); // 返回空图片
//     }

//     // 2. 图像转换
//     cv::Mat nv12_mat(IMG_HEIGHT * 3 / 2, IMG_WIDTH, CV_8UC1, CS.buffer_start);
//     cv::Mat bgr_mat;
//     cv::cvtColor(nv12_mat, bgr_mat, cv::COLOR_YUV2BGR_NV12); 

//     // 3. 入队 (QBUF) -> 转换完后，赶紧把空内存还给硬件，让它去拍下一张！
//     if (ioctl(CS.fd, VIDIOC_QBUF, &CS.buf) < 0) {
//         perror("[ERROR] VIDIOC_QBUF (re-queue) failed");
//         // 不 return 错，因为至少这帧我们拿到了
//     }

//     return bgr_mat;
// }
// 后台抓图线程函数 (生产者)
void camera_thread_func() {
    std::cout << "[INFO] 后台抓图线程已启动..." << std::endl;
    while (keep_running) {
        // 1. DQBUF (从底层硬件拿到装满画面的 buffer)
        if (ioctl(CS.fd, VIDIOC_DQBUF, &CS.buf) < 0) {
            // 如果没拿到，稍微等一下继续尝试，防止死循环占满 CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue; 
        }

        // 2. 加锁，更新全局图片
        {
            // lock_guard 极其好用：大括号开始时自动加锁，大括号结束时自动解锁！
            std::lock_guard<std::mutex> lock(camera_mutex); 
            
            // 创建一个指向 V4L2 内存的 Mat 头
            cv::Mat temp(IMG_HEIGHT * 3 / 2, IMG_WIDTH, CV_8UC1, CS.buffer_start);
            
            // 重点：必须用 clone() 进行深拷贝！
            // 因为下一步我们就要把 V4L2 的内存还回去了，如果不深拷贝，数据会被覆盖。
            latest_nv12_frame = temp.clone(); 
        }

        // 3. QBUF (把空出来的 buffer 赶紧还给底层，让它去拍下一张)
        if (ioctl(CS.fd, VIDIOC_QBUF, &CS.buf) < 0) {
            perror("[ERROR] VIDIOC_QBUF failed in thread");
        }
    }
    std::cout << "[INFO] 后台抓图线程已安全退出。" << std::endl;
}
void release_camera() {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(CS.fd, VIDIOC_STREAMOFF, &type);
    munmap(CS.buffer_start, CS.buffer_length); // 使用存下来的 length
    close(CS.fd);
    std::cout << "[INFO] Camera released." << std::endl;
}
//-----------------------------------------camera-end----------------------------



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
    t_load_end_us = std::chrono::high_resolution_clock::now();

    load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: ImgEnc Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    // The image is read in BGR format
    // cv::Mat img = cv::imread(image_path);
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // // Expand the image into a square and fill it with the specified background color (According the modeling_minicpmv.py)
    // cv::Scalar background_color(127.5, 127.5, 127.5);
    // cv::Mat square_img = expand2square(img, background_color);

    // Resize the image
    // size_t image_width = rknn_app_ctx.model_width;
    // size_t image_height = rknn_app_ctx.model_height;
    // cv::Mat resized_img;
    // cv::Size new_size(image_width, image_height);
    // cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

    size_t n_image_tokens = rknn_app_ctx.model_image_token;
    size_t image_embed_len = rknn_app_ctx.model_embed_size;
    int rkllm_image_embed_len = n_image_tokens * image_embed_len;
    //float img_vec[rkllm_image_embed_len];
    std::vector<float> img_vec(rkllm_image_embed_len);
    // ret = run_imgenc(&rknn_app_ctx, resized_img.data, img_vec);
    // if (ret != 0) {
    //     printf("run_imgenc fail! ret=%d\n", ret);
    // }
    
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
        printf("\n");
        printf("user: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit")
        {
            break;
        }
        if (input_str == "clear")
        {
            ret = rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
            if (ret != 0)
            {
                printf("clear kv cache failed!\n");
            }
            continue;
        }
        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }
        auto t_start = std::chrono::high_resolution_clock::now();
        if (input_str.find("<image>") == std::string::npos) 
        {
            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.role = "user";
            rkllm_input.prompt_input = (char*)input_str.c_str();
        } else {
            // std::cout << "[INFO] 侦测到 <image> 标签，正在抓取最新摄像头画面..." << std::endl;

            // std::cout << "--> DEBUG 1: 准备调用 get_camera_frame" << std::endl;
            // cv::Mat img = get_camera_frame();
            // std::cout << "--> DEBUG 2: get_camera_frame 调用结束" << std::endl;

            // if (img.empty()) {
            //     std::cout << "[ERROR] 抓图失败，请检查摄像头！" << std::endl;
            //     continue; 
            // }

            // std::cout << "--> DEBUG 3: 准备 cvtColor" << std::endl;
            // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            std::cout << "[INFO] 侦测到 <image> 标签，正在从共享内存获取最新画面..." << std::endl;
            
            cv::Mat nv12_img;
            {
                // 1. 加锁：去全局变量里拿图，防止抓图线程此时正好在写入导致图像撕裂
                std::lock_guard<std::mutex> lock(camera_mutex);
                if (latest_nv12_frame.empty()) {
                    std::cout << "[WARNING] 后台还没抓到图，请稍后再试！" << std::endl;
                    continue;
                }
                // 2. 深拷贝出来给主线程用。因为都在内存里，这步极快！
                nv12_img = latest_nv12_frame.clone(); 
            } // 大括号结束，锁自动释放！此时后台线程又可以继续疯狂抓图了。

          
        //     std::cout << "--> DEBUG 3: 启动 RGA 硬件加速 (色彩转换+填充+缩放)" << std::endl;
            
        //    size_t image_width = rknn_app_ctx.model_width;
        //     size_t image_height = rknn_app_ctx.model_height;

        //     // 1. 申请一块最终送给模型的内存 (RGB888格式)
        //     std::vector<uint8_t> rgb_buf(image_width * image_height * 3);
            
        //     // 2. 模拟 expand2square 的背景色：直接把这块内存全刷成灰色 (127)
        //     memset(rgb_buf.data(), 127, image_width * image_height * 3);

        //     // 3. 计算原图按比例缩放后，应该贴在目标灰图的哪个位置 (保持画面不被拉伸变形)
        //     int max_dim = std::max(IMG_WIDTH, IMG_HEIGHT);
        //     float scale = (float)image_width / max_dim; 
            
        //     int scaled_w = IMG_WIDTH * scale;
        //     int scaled_h = IMG_HEIGHT * scale;
        //     int dx = (image_width - scaled_w) / 2; // X轴偏移量 (居中)
        //     int dy = (image_height - scaled_h) / 2; // Y轴偏移量 (居中)

        //     // 4. 配置 RGA 的输入和输出 Buffer
        //     // 源图：刚刚抓到的 NV12，分辨率是 IMG_WIDTH x IMG_HEIGHT
        //     rga_buffer_t src = wrapbuffer_virtualaddr((void*)nv12_img.data, IMG_WIDTH, IMG_HEIGHT, RK_FORMAT_YCbCr_420_SP);
        //     // 目标图：我们刚申请的 RGB 数组，分辨率是 image_width x image_height
        //     rga_buffer_t dst = wrapbuffer_virtualaddr((void*)rgb_buf.data(), image_width, image_height, RK_FORMAT_RGB_888);

        //     // 5. 设置处理区域 (ROI)
        //     im_rect src_rect = {0, 0, IMG_WIDTH, IMG_HEIGHT}; // 截取原图的全部
        //     im_rect dst_rect = {dx, dy, scaled_w, scaled_h};  // 贴到目标图的居中位置

        //     // 6. 一键呼叫硬件！(improcess 会瞬间完成格式转换和缩放粘贴)
        //     rga_buffer_t empty_pat = {0};
        //     im_rect empty_rect = {0, 0, 0, 0};
        //     IM_STATUS rga_stat = improcess(src, dst, empty_pat, src_rect, dst_rect, empty_rect, 0);
        //     if (rga_stat != IM_STATUS_SUCCESS) {
        //         printf("[ERROR] RGA improcess failed: %s\n", imStrError(rga_stat));
        //     }

        //     std::cout << "--> DEBUG 6: 准备 run_imgenc" << std::endl;
        //     // 直接把 RGA 处理好的 rgb_buf 喂给特征提取模型
        //     ret = run_imgenc(&rknn_app_ctx, rgb_buf.data(), img_vec.data());

        //     std::cout << "--> DEBUG 7: run_imgenc 结束" << std::endl;
        //     if (ret != 0) {
        //         printf("run_imgenc fail! ret=%d\n", ret);
        //     }
        //     rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        //     rkllm_input.role = "user";
        //     rkllm_input.multimodal_input.prompt = (char*)input_str.c_str();
        //     rkllm_input.multimodal_input.image_embed = img_vec.data();
        //     rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
        //     rkllm_input.multimodal_input.n_image = 1;
        //     rkllm_input.multimodal_input.image_height = image_height;
        //     rkllm_input.multimodal_input.image_width = image_width;
            std::cout << "--> DEBUG 3: 启动 RGA 硬件加速 (处理 Stride 对齐)" << std::endl;
            
            size_t image_width = rknn_app_ctx.model_width;   // 392
            size_t image_height = rknn_app_ctx.model_height; // 392

            // 1. 计算 16 字节对齐的 width stride
            int aligned_w = (image_width + 15) & (~15); // 392 -> 400

            // 2. 申请对齐后的内存缓冲区
            std::vector<uint8_t> rga_buf(aligned_w * image_height * 3);
            memset(rga_buf.data(), 127, aligned_w * image_height * 3);

            // 3. 计算缩放参数与 ROI 偏移
            int max_dim = std::max(IMG_WIDTH, IMG_HEIGHT);
            float scale = (float)image_width / max_dim; 
            int scaled_w = IMG_WIDTH * scale;
            int scaled_h = IMG_HEIGHT * scale;
            int dx = (image_width - scaled_w) / 2;
            int dy = (image_height - scaled_h) / 2;

            // 4. 构建 RGA 内存描述符 (使用完整参数列表，显式指定 dst 的 wstride 为 aligned_w)
            rga_buffer_t src = wrapbuffer_virtualaddr((void*)nv12_img.data, IMG_WIDTH, IMG_HEIGHT, RK_FORMAT_YCbCr_420_SP);
            // 参数列表: vir_addr, width, height, format, wstride, hstride
            rga_buffer_t dst = wrapbuffer_virtualaddr((void*)rga_buf.data(), image_width, image_height, RK_FORMAT_RGB_888, aligned_w, image_height);

            im_rect src_rect = {0, 0, IMG_WIDTH, IMG_HEIGHT}; 
            im_rect dst_rect = {dx, dy, scaled_w, scaled_h};  

            rga_buffer_t empty_pat;
            memset(&empty_pat, 0, sizeof(empty_pat));
            im_rect empty_rect = {0, 0, 0, 0};
            auto t_rga_start = std::chrono::high_resolution_clock::now();
            // 执行 RGA 硬件加速
            IM_STATUS rga_stat = improcess(src, dst, empty_pat, src_rect, dst_rect, empty_rect, 0);
            if (rga_stat != IM_STATUS_SUCCESS) {
                printf("[ERROR] RGA improcess failed: %s\n", imStrError(rga_stat));
            }

            // 5. 内存重排 (Memory Repacking)
            // rga_buf 当前步长为 400，存在无效 Padding。
            // 映射为 Mat 并裁切 392x392 区域，clone() 会强制分配一块紧凑连续的内存。
            cv::Mat rga_mat(image_height, aligned_w, CV_8UC3, rga_buf.data());
            cv::Mat packed_mat = rga_mat(cv::Rect(0, 0, image_width, image_height)).clone();
            auto t_rga_end = std::chrono::high_resolution_clock::now();
            std::cout << "--> DEBUG 6: 准备 run_imgenc" << std::endl;
            // 传入去除了 Padding 的连续内存指针 packed_mat.data
            ret = run_imgenc(&rknn_app_ctx, packed_mat.data, img_vec.data());
            auto t_npu_end = std::chrono::high_resolution_clock::now();
            if (ret != 0) {
                printf("run_imgenc fail! ret=%d\n", ret);
            }
            
            rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
            rkllm_input.role = "user";
            rkllm_input.multimodal_input.prompt = (char*)input_str.c_str();
            rkllm_input.multimodal_input.image_embed = img_vec.data();
            rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
            rkllm_input.multimodal_input.n_image = 1;
            rkllm_input.multimodal_input.image_height = image_height;
            rkllm_input.multimodal_input.image_width = image_width;
            auto rga_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_rga_end - t_rga_start).count();
            auto npu_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_npu_end - t_rga_end).count();
            
            std::cout << "\n[异构流水线打点]" << std::endl;
            std::cout << " -> RGA 预处理耗时: " << rga_cost << " ms" << std::endl;
            std::cout << " -> NPU 视觉编码耗时: " << npu_cost << " ms" << std::endl;
        }
        printf("robot: ");
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
        auto t_end = std::chrono::high_resolution_clock::now();
        auto cost_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        std::cout << "\n[性能打点] 从接收指令到生成完毕总耗时: " << cost_time_ms << " ms"<<std::endl;
    }

    ret = release_imgenc(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_imgenc fail! ret=%d\n", ret);
    }
    // --- 新增：安全关闭后台线程 ---
    keep_running = false; // 告诉后台线程循环该结束了
    if (camera_thread && camera_thread->joinable()) {
        camera_thread->join(); // 等待后台线程安全结束最后一轮循环
        delete camera_thread;
        camera_thread = nullptr;
    }
    rkllm_destroy(llmHandle);
    release_camera();

    return 0;
}