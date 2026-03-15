#include "rknn_api.h"

#ifndef _RKNN_IMAGE_ENC_H_
#define _RKNN_IMAGE_ENC_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    int model_image_token;
    int model_embed_size;
    // 在你的 rknn_app_context_t 结构体里加上这一行：
    rknn_tensor_mem* zero_copy_embed_mem;
} rknn_app_context_t;

int init_imgenc(const char* model_path, rknn_app_context_t* app_ctx, const int core_num);

int release_imgenc(rknn_app_context_t* app_ctx);

//int run_imgenc(rknn_app_context_t* app_ctx, void* img_data, float* out_result);
int run_imgenc(rknn_app_context_t* app_ctx, void* img_data);
#ifdef __cplusplus
}
#endif

#endif //_RKNN_IMAGE_ENC_H_