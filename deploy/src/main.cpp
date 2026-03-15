#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <csignal>
#include <chrono>

// === 引入我们自己切分好的所有模块 ===
#include "globals.hpp"
#include "utils.hpp"
#include "camera_hal.hpp"
#include "vision_worker.hpp"
#include "llm_worker.hpp"
#include "display_worker.hpp"

// 依然需要这两个官方头文件来初始化大模型
#include "image_enc.h"
#include "rkllm.h"

using namespace std;

// --- 线程与上下文资源 (供 exit_handler 统一回收) ---
static std::thread* camera_thread = nullptr;
static std::thread* vision_thread = nullptr;
static std::thread* llm_thread = nullptr;
static std::thread* display_thread = nullptr;
static rknn_app_context_t rknn_app_ctx;

// --- 优雅退出机制 ---
void exit_handler(int signal) {
    std::cout << "\n[INFO] 接收到退出信号，准备清理现场..." << std::endl;

    // 1. 敲响退堂鼓，唤醒所有睡眠的线程
    keep_running = false; 
    vision_thread_running = false; 
    queue_cv.notify_all(); 

    // 2. 回收所有打工人线程
    if (camera_thread && camera_thread->joinable()) { camera_thread->join(); delete camera_thread; }
    if (vision_thread && vision_thread->joinable()) { vision_thread->join(); delete vision_thread; }
    if (llm_thread && llm_thread->joinable()) { llm_thread->join(); delete llm_thread; }
    if (display_thread && display_thread->joinable()) { display_thread->join(); delete display_thread; }

    // 3. 恢复系统 CPU 调度
    system("echo schedutil | tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor > /dev/null");

    // 4. 释放底层硬件资源
    release_imgenc(&rknn_app_ctx);
    if (llmHandle != nullptr) {
        rkllm_destroy(llmHandle);
        llmHandle = nullptr;
    }
    release_camera();

    std::cout << "[INFO] 清理完毕，程序体面退出。" << std::endl;
    exit(signal);
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " image_path encoder_model_path llm_model_path max_new_tokens max_context_len rknn_core_num "
                  << "[img_start] [img_end] [img_content]\n";
        return -1;
    }

    // 注册 Ctrl+C 信号捕获
    std::signal(SIGINT, exit_handler);

    // ==========================================
    // 1. 初始化底层外设并拉起 Camera 线程
    // ==========================================
    if (!init_camera()) { return -1; }
    
    // 锁定 CPU 为最高性能模式
    system("echo performance | tee /sys/devices/system/cpu/cpufreq/policy*/scaling_governor > /dev/null");
    camera_thread = new std::thread(camera_thread_func);
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 等待第一帧抓取

    // ==========================================
    // 2. 初始化大模型 (LLM)
    // ==========================================
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[3];
    param.top_k = 1;
    param.max_new_tokens = std::atoi(argv[4]);
    param.max_context_len = std::atoi(argv[5]);
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;
    param.img_start   = (argc > 7) ? argv[7] : "<|vision_start|>";
    param.img_end     = (argc > 8) ? argv[8] : "<|vision_end|>";
    param.img_content = (argc > 9) ? argv[9] : "<|image_pad|>";

    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
    //param.core_mask = RKLLM_NPU_CORE_1_2;
    if (rkllm_init(&llmHandle, &param, callback) != 0) {
        printf("[ERROR] rkllm init failed\n");
        exit_handler(-1);
    }
    printf("[INFO] LLM Model loaded.\n");

    // ==========================================
    // 3. 初始化视觉特征编码器 (RKNN)
    // ==========================================
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    const int core_num = atoi(argv[6]);
    if (init_imgenc(argv[2], &rknn_app_ctx, core_num) != 0) {
        printf("[ERROR] init_imgenc fail! model_path=%s\n", argv[2]);
        exit_handler(-1);
    }
    printf("[INFO] ImgEnc Model loaded.\n");

    // ==========================================
    // 4. 派发后续核心业务线程
    // ==========================================
    vision_thread = new std::thread(vision_worker_func, &rknn_app_ctx);
    llm_thread = new std::thread(llm_worker_func, &rknn_app_ctx);
    display_thread = new std::thread(display_worker_func);

    // ==========================================
    // 5. 主循环：交互与任务调度
    // ==========================================
    vector<string> pre_input = {
        "<image>What is in the image?",
        "<image>这幅画面里展示了什么样的场景?"
    };
    cout << "\n********************** 快捷指令测试 ********************\n";
    for (int i = 0; i < (int)pre_input.size(); i++) {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "********************************************************\n\n";
    
    while (keep_running) {
        std::string input_str;
        std::getline(std::cin, input_str);
        
        if (input_str == "exit") break;
        
        if (input_str == "clear") {
            if (rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr) != 0) {
                printf("clear kv cache failed!\n");
            }
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

        // 把任务塞入队列，并敲响铃铛唤醒 LLM 大核
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            prompt_queue.push(input_str);
        }
        queue_cv.notify_one(); 
    }

    // 正常通过输入 exit 退出时的清理操作
    exit_handler(0);
    return 0;
}