#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "image_enc.h"

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

int init_imgenc(const char* model_path, rknn_app_context_t* app_ctx, const int core_num)
{
    int ret;
    int model_len = 0;
    rknn_context ctx = 0;
    int is_crypt = 0;

    ret = rknn_init(&ctx, (void*)model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    printf("===the core num is %d===\n", core_num);
     //在此设置多核推理
    if (core_num == 2) {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
    } else if (core_num == 3) {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
    } else {
        ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);
    }
    //rknn_set_core_mask(ctx, RKNN_NPU_CORE_0);
    if (ret < 0) {
        printf("rknn_set_core_mask fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }
    // Set to context
    // app_ctx->model_image_token = output_attrs[0].dims[1];
    // app_ctx->model_embed_size = output_attrs[0].dims[2];
        // 增加维度判断逻辑，兼容 Qwen2-VL 的 2 维输出 [196, 1536] 和 其他 3/4 维输出
    if (output_attrs[0].n_dims == 2) {
        app_ctx->model_image_token = output_attrs[0].dims[0];
        app_ctx->model_embed_size  = output_attrs[0].dims[1];
    } else {
        app_ctx->model_image_token = output_attrs[0].dims[1];
        app_ctx->model_embed_size  = output_attrs[0].dims[2];
    }
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        // printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
    uint32_t output_size = app_ctx->model_image_token * app_ctx->model_embed_size * sizeof(float);

    // 【新增】：向系统申请一块 NPU 和 CPU 都能直接访问的连续物理内存
    app_ctx->zero_copy_embed_mem = rknn_create_mem(ctx, output_size);
    if (app_ctx->zero_copy_embed_mem == NULL) {
        printf("[ERROR] rknn_create_mem 申请零拷贝物理内存失败！\n");
        return -1;
    }
    printf("[INFO] 成功分配零拷贝物理内存，大小: %u 字节\n", output_size);

    return 0;
    return 0;
}

int release_imgenc(rknn_app_context_t* app_ctx)
{
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}
// 注意：去掉了原来的 float* out_result 参数，因为不需要外部接盘了
int run_imgenc(rknn_app_context_t* app_ctx, void* img_data) 
{
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[1];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // 1. 设置输入 (这部分保持原样)
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    inputs[0].buf   = img_data;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // 2. 运行 NPU 计算
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // 3. 【核心改造：零拷贝输出】
    outputs[0].want_float = 1;      // 依然需要底层自动反量化为 float
    outputs[0].is_prealloc = 1;     // 【关键】告诉驱动：我已经准备好盘子了，别给我重新分配！
    outputs[0].index = 0;
    outputs[0].buf = app_ctx->zero_copy_embed_mem->virt_addr; // 指向我们申请的物理内存的虚拟映射地址
    outputs[0].size = app_ctx->zero_copy_embed_mem->size;

    // 获取输出：此时 NPU 会直接把 float 数据填入 app_ctx->zero_copy_embed_mem
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    // 【重要】：删除了 memcpy！删除了 rknn_outputs_release！
    // 因为这块内存是我们自己管理的全局物理内存，不需要释放，也不需要拷贝！

    return ret;
}
// int run_imgenc(rknn_app_context_t* app_ctx, void* img_data, float* out_result)
// {
//     int ret;
//     rknn_input inputs[1];
//     rknn_output outputs[1];

//     memset(inputs, 0, sizeof(inputs));
//     memset(outputs, 0, sizeof(outputs));

//     // Set Input Data
//     inputs[0].index = 0;
//     inputs[0].type  = RKNN_TENSOR_UINT8;
//     inputs[0].fmt   = RKNN_TENSOR_NHWC;
//     inputs[0].size  = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
//     inputs[0].buf   = img_data;

//     ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
//     if (ret < 0) {
//         printf("rknn_input_set fail! ret=%d\n", ret);
//         return -1;
//     }

//     // Run
//     ret = rknn_run(app_ctx->rknn_ctx, nullptr);
//     if (ret < 0) {
//         printf("rknn_run fail! ret=%d\n", ret);
//         return -1;
//     }

//     // Get Output
//     outputs[0].want_float = 1;
//     ret = rknn_outputs_get(app_ctx->rknn_ctx, 1, outputs, NULL);
//     if (ret < 0) {
//         printf("rknn_outputs_get fail! ret=%d\n", ret);
//         goto out;
//     }

//     // Post Process
//     memcpy(out_result, outputs[0].buf, outputs[0].size);

//     // Remeber to release rknn output
//     rknn_outputs_release(app_ctx->rknn_ctx, 1, outputs);

// out:

//     return ret;
// }