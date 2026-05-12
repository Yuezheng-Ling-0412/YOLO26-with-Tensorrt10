#ifndef _trt_infer_H
#define _trt_infer_H

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <chrono>

#include "cuda_preprocessor.hpp"
#include "trt_engine.hpp"
#include "postprocessor.hpp"

#define MAX_DET_NUM 64     // 最大检测目标数
#define CONF_THRESHOLD 0.4 // 置信度阈值
#define NUM_CLASSES 80     // 目标种类

class trt_infer
{
public:
    cv::Mat img;
    int infer();
    trt_infer(char *engine_path, int src_w, int src_h);
    ~trt_infer();

private:
    int net_w = 640; // 目标网络输入宽度
    int net_h = 640; // 目标网络输入高度
    CudaPreprocessor preprocessor;
    TrtEngine engine;
    cudaStream_t stream = nullptr;
    PostProcessor postprocessor;
    std::vector<Detection> dets;
    uint8_t *d_input = nullptr; // 固定输入显存
    int input_size = 0;         // 输入显存大小
};

trt_infer::trt_infer(char *engine_path, int src_w, int src_h)
{
    // 初始化预处理器
    preprocessor.init(src_w, src_h, net_w, net_h);

    // 初始化推理引擎
    engine.loadEngine(engine_path);

    // 初始化cuda流
    cudaStreamCreate(&stream);

    // 初始化后处理器
    postprocessor.init(CONF_THRESHOLD, NUM_CLASSES, MAX_DET_NUM);

    // 分配输入缓冲区
    input_size = src_w * src_h * 3 * sizeof(uint8_t);
    cudaMalloc(&d_input, input_size);
}

trt_infer::~trt_infer()
{
    if (d_input)
        cudaFree(d_input);
    cudaStreamDestroy(stream);
    engine.destroy();
    preprocessor.destroy();
}

int trt_infer::infer()
{
    // 拷贝图像到gpu
    cudaMemcpyAsync(d_input, img.data, input_size, cudaMemcpyHostToDevice, stream);

    // 异步执行预处理
    float *input_dev = preprocessor.process(d_input, stream);

    // 异步推理
    engine.infer(input_dev, stream);

    // 复制推理结果到主机
    float *output_host = nullptr;
    size_t output_bytes = static_cast<size_t>(engine.outputChannels()) * engine.outputBoxes() * sizeof(float);
    cudaMallocHost(&output_host, output_bytes);
    cudaMemcpyAsync(output_host, engine.outputDevice(),
                    output_bytes, cudaMemcpyDeviceToHost, stream);

    // 同步cuda流
    cudaStreamSynchronize(stream);

    const auto &lb = preprocessor.letterbox();
    int num_dets = postprocessor.postprocess(output_host, dets,
                                             lb.scale, lb.pad_x, lb.pad_y, lb.src_w, lb.src_h);

    for (const auto &det : dets)
    {
        cv::rectangle(img, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(255, 204, 102, 0), 2);
    }

    // 释放资源
    cudaFreeHost(output_host);

    return 0;
}

#endif

