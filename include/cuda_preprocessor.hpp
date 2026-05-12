#pragma once
#include <cuda_runtime.h>
#include <cstdint>

/**
 * CudaPreprocessor
 *
 * 单 CUDA 核函数，完成：
 *   读取由opencv解码得到的HWC/BGR格式图像
 *   → 等比缩放
 *   → 填充
 *   → 归一化 /255.0f
 *   → 输出 NCHW float32，通道顺序 RGB
 *
 * 输入：GPU 可访问的原始帧（Pinned / UMA）
 * 输出：GPU 设备内存，shape [1, 3, dst_h, dst_w]
 */

struct LetterboxInfo
{
    float scale; // 缩放
    int pad_x;   // 左右填充
    int pad_y;   // 上下填充
    int src_w;
    int src_h;
    int dst_w;
    int dst_h;
};

struct AffineMatrix
{
    float i2d[6]; // image to dst(network), 2x3 matrix
    float d2i[6]; // dst to image, 2x3 matrix

    // 求解imat的逆矩阵，第三行是确定的0, 0, 1
    void invertAffineTransform(float imat[6], float omat[6]);

    // 计算i2d和d2i矩阵
    void compute(const int &src_width, const int &src_height, const int &dst_width, const int &dst_height);
};

class CudaPreprocessor
{
public:
    CudaPreprocessor() = default;
    ~CudaPreprocessor();

    bool init(int src_w, int src_h, int net_w = 640, int net_h = 640);
    void destroy();

    /**
     * 执行预处理（异步，在 stream 上提交）
     * @param input_dev  opencv HWC/BGR帧的设备/UMA 指针
     * @param stream    CUDA stream
     * @return 输出张量设备指针（float*，NCHW RGB）
     */
    float *process(const void *input_dev, cudaStream_t stream);

    float *output_dev() const { return output_dev_; }

    const LetterboxInfo& letterbox() const { return lb_; }

private:
    float *output_dev_ = nullptr; // [1, 3, net_h, net_w]
    AffineMatrix matrix_;         // 映射矩阵
    LetterboxInfo lb_{};          // letterbox信息
};