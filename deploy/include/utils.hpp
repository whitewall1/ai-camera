#pragma once

// 将当前调用该函数的线程，绑定到指定的 CPU 核心区间 [start_core, end_core]
bool bind_thread_to_cpus(int start_core, int end_core);