#pragma once
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

/**
 * TrtEngine
 *
 * TensorRT 10.x 推理引擎封装。
 * 使用 setTensorAddress() + enqueueV3() 异步推理接口。
 *
 * 假定模型为 YOLO26 ONNX 静态 shape 导出：
 *   输入:  "images"   [1, 3, 640, 640]  float32
 *   输出:  "output0"  [1, 300, 6]   float32
 */
class TrtEngine
{
public:
    TrtEngine() = default;
    ~TrtEngine() { destroy(); }

    // 禁用拷贝构造
    TrtEngine(const TrtEngine &) = delete;
    TrtEngine &operator=(const TrtEngine &) = delete;

    /**
     * 从 .engine 文件加载推理引擎
     */
    bool loadEngine(const std::string &engine_path);

    /**
     * 异步执行推理
     * @param input_dev   输入张量设备指针 (float*, NCHW)
     * @param stream      CUDA stream
     */
    bool infer(float *input_dev, cudaStream_t stream);

    /**
     * 等待推理完成（同步流）
     */
    void synchronize(cudaStream_t stream);

    // 推理输出设备指针（float*，[1, 300, 6]）
    float *outputDevice() const { return output_dev_; }

    // 输出张量维度信息
    int outputChannels() const { return output_channels_; } // 4 + nc
    int outputBoxes() const { return output_boxes_; }       // 8400

    void destroy();

private:
    // TRT 10 对象（使用原始指针 + 手动 destroy，避免对 IRuntime/IEngine 包装问题）
    nvinfer1::IRuntime *runtime_ = nullptr;          // 运行时环境，用于反序列化引擎
    nvinfer1::ICudaEngine *engine_ = nullptr;        // 推理引擎，存储优化后的网络
    nvinfer1::IExecutionContext *context_ = nullptr; // 执行上下文，管理中间状态

    float *output_dev_ = nullptr; // 输出缓冲

    int input_size_ = 0;      // 输入张量的总元素数，通常为1*3*640*640
    int output_channels_ = 0; // 输出张量的通道数，300
    int output_boxes_ = 0;    // [x1, y1, x2, y2, conf, nc_id]

    std::string input_name_;  // 输入张量名称，从engine中获取
    std::string output_name_; // 输出张量名称
};

// ─────────────────────────────────────────────────────────────────────────────
// TensorRT Logger
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            printf("[TRT] %s\n", msg);
    }
};

// 全局 logger（供 TrtEngine 内部使用）
extern TrtLogger g_trt_logger;
