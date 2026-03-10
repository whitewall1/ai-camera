#include "utils.hpp"

#include <iostream>
#include <pthread.h>
#include <sched.h> // 提供 cpu_set_t, CPU_ZERO, CPU_SET 等底层调度宏
bool bind_thread_to_cpus(int start_core, int end_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    for (int i = start_core; i <= end_core; ++i) {
        CPU_SET(i, &cpuset);
    }

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    if (rc != 0) {
        std::cerr << "[Warning] Error calling pthread_setaffinity_np: " << rc << std::endl;
        return false;
    }
    
    //std::cout << "[INFO] 成功将线程绑定到 CPU " << start_core << " ~ " << end_core << std::endl;
    return true;
}
