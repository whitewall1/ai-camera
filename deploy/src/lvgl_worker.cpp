#include "lvgl_worker.hpp"
#include "globals.hpp" 
#include "utils.hpp"   
#include "lvgl/lvgl.h" 

#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <linux/input.h>
#include <fcntl.h>
#include <unistd.h>

// 声明我们刚刚生成的离线中文字库
LV_FONT_DECLARE(lv_font_cn_16);

static void* g_drm_map_ptr = nullptr;
static uint32_t g_screen_width = 1024;
static int evdev_fd = -1;
static int32_t last_x = 0;
static int32_t last_y = 0;
static lv_indev_state_t last_state = LV_INDEV_STATE_REL;

// ==========================================================
// 触摸屏与显存底层驱动
// ==========================================================
static void touch_init() {
    evdev_fd = open("/dev/input/event1", O_RDONLY | O_NONBLOCK);
}

static void touch_read_cb(lv_indev_drv_t * indev_drv, lv_indev_data_t * data) {
    if (evdev_fd < 0) return;
    struct input_event in;
    while (read(evdev_fd, &in, sizeof(struct input_event)) > 0) {
        if (in.type == EV_ABS) {
            if (in.code == ABS_MT_POSITION_X || in.code == ABS_X) last_x = in.value;
            else if (in.code == ABS_MT_POSITION_Y || in.code == ABS_Y) last_y = in.value;
        } else if (in.type == EV_KEY) {
            if (in.code == BTN_TOUCH) {
                last_state = (in.value == 1) ? LV_INDEV_STATE_PR : LV_INDEV_STATE_REL;
            }
        }
    }
    data->point.x = last_x;
    data->point.y = last_y;
    data->state = last_state;
}

static void drm_disp_flush_cb(lv_disp_drv_t * disp_drv, const lv_area_t * area, lv_color_t * color_p) {
    if (!g_drm_map_ptr) { lv_disp_flush_ready(disp_drv); return; }
    int32_t x1 = area->x1, y1 = area->y1, x2 = area->x2, y2 = area->y2;
    uint32_t* drm_fb = static_cast<uint32_t*>(g_drm_map_ptr);
    uint32_t* lvgl_buf = reinterpret_cast<uint32_t*>(color_p);
    uint32_t copy_width = (x2 - x1 + 1);
    for (int32_t y = y1; y <= y2; y++) {
        memcpy(&drm_fb[y * g_screen_width + x1], &lvgl_buf[(y - y1) * copy_width], copy_width * sizeof(uint32_t));
    }
    lv_disp_flush_ready(disp_drv);
}

// ==========================================================
// 业务逻辑：按钮点击与流式文本定时器
// ==========================================================
static void llm_text_update_cb(lv_timer_t * timer) {
    lv_obj_t * label = (lv_obj_t *)timer->user_data;
    std::lock_guard<std::mutex> lock(llm_response_mutex);
    if (!current_llm_response.empty()) {
        lv_label_set_text(label, current_llm_response.c_str());
    }
}

static void btn_event_cb(lv_event_t * e) {
    if (lv_event_get_code(e) == LV_EVENT_CLICKED) {
        lv_obj_t * label = (lv_obj_t *)lv_event_get_user_data(e);
        
        // 1. 刷新 UI 状态
        {
            std::lock_guard<std::mutex> lock(llm_response_mutex);
            current_llm_response = "NPU 正在提取视觉特征，请稍候...\n";
            lv_label_set_text(label, current_llm_response.c_str());
        }

        // 2. 纯视觉检测推断：将中文 Prompt 推入 LLM 队列
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            prompt_queue.push("<image>详细描述这幅视频流画面里展示了什么样的场景?");
        }
        queue_cv.notify_one(); 
    }
}

// ==========================================================
// LVGL 核心工作线程
// ==========================================================
void lvgl_worker_func(void* drm_map_ptr, uint32_t width, uint32_t height) {
    bind_thread_to_cpus(2, 3);
    g_drm_map_ptr = drm_map_ptr;
    g_screen_width = width;

    lv_init();
    touch_init();

    const uint32_t buf_size = width * height / 10;
    static lv_color_t* buf_1 = (lv_color_t*)malloc(buf_size * sizeof(lv_color_t));
    static lv_disp_draw_buf_t draw_buf;
    lv_disp_draw_buf_init(&draw_buf, buf_1, NULL, buf_size);

    static lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);
    disp_drv.draw_buf = &draw_buf;
    disp_drv.flush_cb = drm_disp_flush_cb;
    disp_drv.hor_res = width; disp_drv.ver_res = height;
    disp_drv.screen_transp = 1; 
    lv_disp_drv_register(&disp_drv);

    static lv_indev_drv_t indev_drv;
    lv_indev_drv_init(&indev_drv);
    indev_drv.type = LV_INDEV_TYPE_POINTER;
    indev_drv.read_cb = touch_read_cb;
    lv_indev_drv_register(&indev_drv);

    lv_obj_set_style_bg_opa(lv_scr_act(), 0, 0);

    // --- 构建流式文本输出的半透明面板 ---
    lv_obj_t * result_panel = lv_obj_create(lv_scr_act());
    lv_obj_set_size(result_panel, 800, 250);
    lv_obj_align(result_panel, LV_ALIGN_TOP_MID, 0, 40);
    lv_obj_set_style_bg_color(result_panel, lv_color_hex(0x000000), 0);
    lv_obj_set_style_bg_opa(result_panel, LV_OPA_70, 0); 
    lv_obj_set_style_border_width(result_panel, 2, 0);
    lv_obj_set_style_border_color(result_panel, lv_color_hex(0x00A8FF), 0); 

    lv_obj_t * result_label = lv_label_create(result_panel);
    lv_obj_set_width(result_label, 760);
    lv_label_set_long_mode(result_label, LV_LABEL_LONG_WRAP); 
    
    // 强制挂载中文字库
    lv_obj_set_style_text_font(result_label, &lv_font_cn_16, 0); 
    lv_label_set_text(result_label, "系统就绪。点击下方按钮开始视频流检测。");
    
    lv_obj_set_style_text_color(result_label, lv_color_hex(0xFFFFFF), 0);
    lv_obj_align(result_label, LV_ALIGN_TOP_LEFT, 0, 0);

    // 启动 LVGL 硬件定时器 (50ms 刷新率)
    lv_timer_create(llm_text_update_cb, 50, result_label);

    // --- 构建一键检测交互按钮 ---
    lv_obj_t * btn = lv_btn_create(lv_scr_act());
    lv_obj_set_size(btn, 240, 60);
    lv_obj_align(btn, LV_ALIGN_BOTTOM_MID, 0, -40);
    lv_obj_add_event_cb(btn, btn_event_cb, LV_EVENT_ALL, result_label);

    lv_obj_t * label = lv_label_create(btn);
    // 按钮文本挂载中文字库
    lv_obj_set_style_text_font(label, &lv_font_cn_16, 0);
    lv_label_set_text(label, "开始视觉检测");
    lv_obj_center(label);

    while (keep_running) {
        lv_tick_inc(5); 
        lv_timer_handler(); 
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    if(evdev_fd >= 0) close(evdev_fd);
    free(buf_1);
}