#include "trt_engine.hpp"
#include <NvOnnxParser.h>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <stdexcept>

TrtLogger g_trt_logger;

bool TrtEngine::loadEngine(const std::string &engine_path)
{
    // 读取序列化engine文件
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good())
    {
        fprintf(stderr, "[TRT] Cannot open engine file: %s\n",
                engine_path.c_str());
        return false;
    }

    size_t size = file.tellg();   // 获取总字节数
    file.seekg(0, std::ios::beg); // 回到文件开头
    std::vector<char> data(size); // 分配缓冲区
    file.read(data.data(), size); // 读取全部数据
    file.close();                 // 关闭文件

    // 创建Runtime并反序列化
    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_)
    {
        fprintf(stderr, "[TRT] createInferRuntime failed\n");
        return false;
    }

    // 创建反序列化引擎，将二进制数据构建为ICudaEngine对象
    engine_ = runtime_->deserializeCudaEngine(data.data(), size);
    if (!engine_)
    {
        fprintf(stderr, "[TRT] deserializeCudaEngine failed\n");
        return false;
    }

    // 创建执行上下文
    context_ = engine_->createExecutionContext();
    if (!context_)
    {
        fprintf(stderr, "[TRT] createExecutionContext failed\n");
        return false;
    }

    // 解析输入/输出张量信息
    int n_tensors = engine_->getNbIOTensors();
    // printf("[TRT] Engine loaded: %d IO tensors\n", n_tensors);

    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = engine_->getIOTensorName(i);      // 获取张量名称
        nvinfer1::Dims dims = engine_->getTensorShape(name); // 获取形状
        auto mode = engine_->getTensorIOMode(name);          // 获取模式(输入/输出)

        // 打印维度信息
        // printf("[TRT]  %s %s [",
        //        mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT " : "OUTPUT",
        //        name);
        // for (int d = 0; d < dims.nbDims; ++d)
        //     printf("%lld%s", (long long)dims.d[d], d < dims.nbDims - 1 ? ", " : "");
        // printf("]\n");

        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            // 记录输入张量名称，并计算输入元素总数（方便后续校验）
            input_name_ = name;
            // 计算输入元素数 (N * C * H * W)
            input_size_ = 1;
            for (int d = 0; d < dims.nbDims; ++d)
                input_size_ *= static_cast<int>(dims.d[d]);
        }
        else
        {
            output_name_ = name;
            assert(dims.nbDims == 3); // 对于 YOLO26 输出，期望形状为 [1, 300, 6]
            output_channels_ = static_cast<int>(dims.d[1]);
            output_boxes_ = static_cast<int>(dims.d[2]);
        }
    }

    // 分配输出缓冲
    size_t out_bytes = 1ull * output_channels_ * output_boxes_ * sizeof(float);
    cudaError_t err = cudaMalloc(&output_dev_, out_bytes);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[TRT] cudaMalloc output failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }

    // printf("[TRT] output shape: [1, %d, %d] | %.2f MB\n",
    //        output_channels_, output_boxes_,
    //        static_cast<double>(out_bytes) / (1 << 20));
    return true;
}

bool TrtEngine::infer(float *input_dev, cudaStream_t stream)
{
    // TRT 10 API: 绑定输入输出地址后异步入队
    // 关联输入张量与设备指针
    if (!context_->setTensorAddress(input_name_.c_str(), input_dev))
    {
        fprintf(stderr, "[TRT] setTensorAddress (input) failed\n");
        return false;
    }
    // 关联输出张量与设备指针
    if (!context_->setTensorAddress(output_name_.c_str(), output_dev_))
    {
        fprintf(stderr, "[TRT] setTensorAddress (output) failed\n");
        return false;
    }
    // 推理任务异步入对到指定的cuda流
    if (!context_->enqueueV3(stream))
    {
        fprintf(stderr, "[TRT] enqueueV3 failed\n");
        return false;
    }
    return true;
}

void TrtEngine::synchronize(cudaStream_t stream)
{
    cudaStreamSynchronize(stream);
}

void TrtEngine::destroy()
{
    if (output_dev_)
    {
        cudaFree(output_dev_);
        output_dev_ = nullptr;
    }
    // TensorRT 10.x: destroy() 已废弃，直接 delete
    if (context_)
    {
        delete context_;
        context_ = nullptr;
    }
    if (engine_)
    {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_)
    {
        delete runtime_;
        runtime_ = nullptr;
    }
}
