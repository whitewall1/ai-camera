#pragma once
#include <cstddef>
#include "rkllm.h"
#include "image_enc.h" // 这里面定义了 rknn_app_context_t

// 大模型推理线程主循环
void llm_worker_func(rknn_app_context_t* app_ctx);

// 大模型输出流式回调函数
int callback(RKLLMResult *result, void *userdata, LLMCallState state);