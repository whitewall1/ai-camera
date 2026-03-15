#include "llm_worker.hpp"
#include "globals.hpp" // 获取 prompt_queue, vision_mutex, global_img_embed 等
#include "utils.hpp"   // 获取 bind_thread_to_cpus

#include <iostream>
#include <string>
#include <chrono>
#include <cstring>
int generated_token_count=0;
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
        generated_token_count=0;
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
            //auto t_feat_start = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(vision_mutex);
                if (!is_vision_ready) {
                    std::cout << "[WARNING] 后台视觉特征尚未初始化，跳过此次多模态推理！" << std::endl;
                    std::cout << "\nuser: " << std::flush;
                    continue; 
                }
                local_img_vec = global_img_embed; // 极速深拷贝（微秒级）
            }
            //auto feat_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t_feat_start).count();
            //std::cout << " -> 从后台获取最新视觉特征耗时: " << feat_cost << " us" << std::endl;

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
        float tps = (generated_token_count * 1000.0f) / cost_time_ms;
std::cout << "[METRIC] LLM 生成速度: " << tps << " tokens/s (耗时: " << cost_time_ms << "ms)" << std::endl;
        // 打印出新的交互提示符，因为主线程可能早就等在那里了
        std::cout << "\nuser: " << std::flush; 
    }
    std::cout << "[INFO] LLM 推理线程已退出。" << std::endl;
}
// 注意参数类型和内部判断宏的改变
int callback(RKLLMResult *result, void *userdata, LLMCallState state) {
    if (state == RKLLM_RUN_NORMAL) {
        generated_token_count++;
        // 【核心修复】：增加严格的空指针防御，防止 basic_string::append 崩溃
        if (result != nullptr && result->text != nullptr) {
            std::lock_guard<std::mutex> lock(llm_response_mutex);
            current_llm_response += result->text;
            
            printf("%s", result->text);
            fflush(stdout);
        }
    } else if (state == RKLLM_RUN_FINISH) {
        printf("\n");
    } else if (state == RKLLM_RUN_ERROR) {
        printf("\n[ERROR] LLM run error\n");
    }
    
    return 0; 
}