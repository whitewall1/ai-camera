#include "camera_hal.hpp"
#include "globals.hpp" // 获取 CS, BUFFER_COUNT, camera_mutex 等全局变量图纸与声明
#include "utils.hpp"   // 获取 bind_thread_to_cpus 函数

#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <thread>
#include <chrono>

// ==========================================================
// 下面直接无缝粘贴你刚才发给我的那三段核心代码：
// bool init_camera() { ... }
// void camera_thread_func() { ... }
// void release_camera() { ... }
// ==========================================================
bool init_camera() {
    CS.fd = open(VIDEO_NODE, O_RDWR); // 直接存入 CS.fd
    if (CS.fd < 0) {
        perror("[ERROR] Failed to open video node");
        return false;
    }

    // 2. Set Format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = IMG_WIDTH;
    fmt.fmt.pix_mp.height = IMG_HEIGHT;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_ANY;

    if (ioctl(CS.fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("[ERROR] VIDIOC_S_FMT failed");
        close(CS.fd);
        return false;
    }

    
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = BUFFER_COUNT; // 【改动1】从 1 改成 BUFFER_COUNT (也就是 4)
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(CS.fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("[ERROR] VIDIOC_REQBUFS failed");
        close(CS.fd);
        return false;
    }
    // 4. Query and Map Buffer
    // 4. 遍历这 4 个 Buffer：查询属性、导出 FD、映射内存、交还给底层
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        struct v4l2_plane planes[1];
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.length = 1;
        buf.m.planes = planes;

        // 查询第 i 个 Buffer 的信息
        if (ioctl(CS.fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("[ERROR] VIDIOC_QUERYBUF failed");
            return false;
        }

        // 【改动2：最核心的一步！】导出 DMA-BUF 文件描述符 (FD)
        struct v4l2_exportbuffer expbuf;
        memset(&expbuf, 0, sizeof(expbuf));
        expbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        expbuf.index = i;
        expbuf.plane = 0; // NV12 往往只有一个 Plane
        
        if (ioctl(CS.fd, VIDIOC_EXPBUF, &expbuf) == 0) {
            // 成功拿到了宝贵的硬件 FD，存入我们的全局结构体中！
            CS.buffers[i].fd = expbuf.fd;
            //std::cout << "[INFO] 成功导出 Buffer " << i << " 的 DMA FD: " << expbuf.fd << std::endl;
        } else {
            perror("[ERROR] VIDIOC_EXPBUF failed");
            return false;
        }

        // 记录其他信息
        CS.buffers[i].index = i;
        CS.buffers[i].length = buf.m.planes[0].length;
        CS.buffers[i].in_use_by_llm = false; // 初始状态：大模型没有在用它

        // 虽然我们要用 FD，但也同时把它映射成虚拟地址，方便后续调试
        CS.buffers[i].start = mmap(NULL, buf.m.planes[0].length, PROT_READ | PROT_WRITE, 
                                   MAP_SHARED, CS.fd, buf.m.planes[0].m.mem_offset);

        // QBUF: 把空 Buffer 正式塞回给底层驱动，让它准备拍照装填数据
        if (ioctl(CS.fd, VIDIOC_QBUF, &buf) < 0) {
            perror("[ERROR] VIDIOC_QBUF failed");
            return false;
        }
    }
    

    // 5. Stream on
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(CS.fd, VIDIOC_STREAMON, &type) < 0) {
        perror("[ERROR] VIDIOC_STREAMON failed");
        return false; 
    }

    //std::cout << "[INFO] Camera initialized successfully!" << std::endl;
    return true; // 修复：别忘了成功时返回 true
}
// 后台抓图线程函数 (生产者)
void camera_thread_func() {
      int rc = pthread_setname_np(pthread_self(), "camera"); // <= 15 chars
    if (rc != 0) {
        std::cerr << "pthread_setname_np failed: " << std::strerror(rc) << "\n";
    }
    bind_thread_to_cpus(0, 3);
    std::cout << "[INFO] 后台抓图线程已启动..." << std::endl;
    
    while (keep_running) {
        
        // 1. DQBUF (从底层硬件拿到装满画面的 buffer)
        struct v4l2_plane planes[1];
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        buf.length = 1;
        buf.m.planes = planes;
        if (ioctl(CS.fd, VIDIOC_DQBUF, &buf) < 0) {
            // 如果没拿到，稍微等一下继续尝试，防止死循环占满 CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue; 
        }
        int new_index=buf.index;
        // 2. 加锁，更新全局图片
        {
            std::lock_guard<std::mutex> lock(camera_mutex); 
            if(last_held_index!=-1&&!CS.buffers[last_held_index].in_use_by_llm){
                struct v4l2_plane qplanes[1];
                struct v4l2_buffer qbuf;
                memset(&qbuf, 0, sizeof(qbuf));
                memset(qplanes, 0, sizeof(qplanes));
                qbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
                qbuf.memory = V4L2_MEMORY_MMAP;
                qbuf.index=last_held_index;
                qbuf.length = 1;
                qbuf.m.planes = qplanes;
                if (ioctl(CS.fd, VIDIOC_QBUF, &qbuf) < 0) {
                    // 如果没拿到，稍微等一下继续尝试，防止死循环占满 CPU
                    perror("[ERROR] VIDIOC_QBUF failed in thread");
                }
            }
            last_held_index=new_index;
            latest_buf_index=new_index;
        }
       
    }
    std::cout << "[INFO] 后台抓图线程已安全退出。" << std::endl;
}
void release_camera() {
    // 1. 停止视频流
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(CS.fd, VIDIOC_STREAMOFF, &type);

    // 2. 遍历释放我们申请的 4 个 Buffer
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        // 解除虚拟地址映射
        if (CS.buffers[i].start != NULL && CS.buffers[i].start != MAP_FAILED) {
            munmap(CS.buffers[i].start, CS.buffers[i].length);
        }
        
        // 【极其重要】关闭导出的 DMA-BUF 文件描述符！防止内存和句柄泄露！
        if (CS.buffers[i].fd > 0) {
            close(CS.buffers[i].fd);
        }
    }

    // 3. 关闭摄像头设备节点
    if (CS.fd > 0) {
        close(CS.fd);
    }
    
    std::cout << "[INFO] Camera resources and DMA-BUFs released safely." << std::endl;
}