#pragma once

#include "image_enc.h" // 必须包含，因为用到了 rknn_app_context_t
#include <opencv2/opencv.hpp>

// 视觉预处理与 NPU 特征提取的后台线程主循环
void vision_worker_func(rknn_app_context_t* app_ctx);

// 图像扩展辅助函数 (如果仅内部使用，其实可以不在此声明)
cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color);