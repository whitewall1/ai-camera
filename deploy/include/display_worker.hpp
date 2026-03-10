#pragma once

// 负责将摄像头数据通过 RGA 零拷贝推送到 DRM 屏幕的后台线程
void display_worker_func();