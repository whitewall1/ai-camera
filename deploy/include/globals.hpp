#pragma once
#include <cstddef>
#include <mutex>
#include <atomic>
#include <queue>
#include <vector>
#include <string>
#include <condition_variable>
#include "rkllm.h" // 必须包含这个，因为用到了 LLMHandle

// ==========================================
// 1. 宏定义 (全员可见的常量)
// ==========================================
#define VIDEO_NODE "/dev/video16"
#define IMG_WIDTH  1920
#define IMG_HEIGHT 1080
#define BUFFER_COUNT 8 

// ==========================================
// 2. 类型图纸 (结构体定义，不占内存)
// ==========================================
struct CameraBuffer {
    int index;           
    int fd;              
    void* start;         
    size_t length;
    bool in_use_by_llm;  
};

struct CameraState {
    int fd;
    CameraBuffer buffers[BUFFER_COUNT];
};

// ==========================================
// 3. 全局变量“声明” (告诉编译器它们存在)
// ==========================================

// --- 核心硬件句柄 ---
extern LLMHandle llmHandle;
extern struct CameraState CS;

// --- 系统控制 ---
extern std::atomic<bool> keep_running;

// --- 摄像头共享区 ---
extern std::mutex camera_mutex;
extern int latest_buf_index;
extern int last_held_index;

// --- 视觉特征共享区 ---
extern std::mutex vision_mutex;
extern std::vector<float> global_img_embed;
extern bool is_vision_ready;
extern std::atomic<bool> is_llm_generating;
extern std::atomic<bool> vision_thread_running;
// --- LLM 任务队列 ---
extern std::queue<std::string> prompt_queue;
extern std::mutex queue_mutex;
extern std::condition_variable queue_cv;

// 用于 LLM 和 LVGL 之间的跨线程文本通讯
extern std::string current_llm_response;
extern std::mutex llm_response_mutex;
