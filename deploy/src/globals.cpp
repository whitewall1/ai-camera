#include "globals.hpp"

// ==========================================
// 全局变量“物理实例化” (真正占用内存的地方，只能有一份)
// ==========================================

// --- 核心硬件句柄 ---
LLMHandle llmHandle = nullptr;
struct CameraState CS = {}; // 初始化为空

// --- 系统控制 ---
std::atomic<bool> keep_running{true};

// --- 摄像头共享区 ---
std::mutex camera_mutex;
int latest_buf_index = -1;
int last_held_index = -1;

// --- 视觉特征共享区 ---
std::mutex vision_mutex;
std::vector<float> global_img_embed;
bool is_vision_ready = false;
std::atomic<bool> is_llm_generating{false};
std::atomic<bool> vision_thread_running{true};
// --- LLM 任务队列 ---
std::queue<std::string> prompt_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;

std::string current_llm_response = "";
std::mutex llm_response_mutex;