#include "cuda_preprocessor.hpp"
#include <cstdio>
#include <cmath>
#include <iostream>

void AffineMatrix::invertAffineTransform(float imat[6], float omat[6])
{
    float i00 = imat[0];
    float i01 = imat[1];
    float i02 = imat[2];
    float i10 = imat[3];
    float i11 = imat[4];
    float i12 = imat[5];

    // 计算行列式
    float D = i00 * i11 - i01 * i10;
    D = D != 0 ? 1.0 / D : 0;

    // 计算剩余的伴随矩阵除以行列式
    float A11 = i11 * D;
    float A22 = i00 * D;
    float A12 = -i01 * D;
    float A21 = -i10 * D;
    float b1 = -A11 * i02 - A12 * i12;
    float b2 = -A21 * i02 - A22 * i12;
    omat[0] = A11;
    omat[1] = A12;
    omat[2] = b1;
    omat[3] = A21;
    omat[4] = A22;
    omat[5] = b2;
}

void AffineMatrix::compute(const int &src_width, const int &src_height, const int &dst_width, const int &dst_height)
{
    float scale_x = dst_width / (float)src_width;
    float scale_y = dst_height / (float)src_height;

    float scale = std::min(scale_x, scale_y);

    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = -scale * src_width * 0.5 + dst_width * 0.5;

    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    // 求逆矩阵
    AffineMatrix::invertAffineTransform(i2d, d2i);
}

// 将目标图中每个像素映射回原图中
__device__ void affine_project(float *matrix, int x, int y, float *proj_x, float *proj_y)
{
    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

// 双线性插值核函数
__global__ void warp_affine_bilinear_kernel(
    const uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height,
    uint8_t fill_value, AffineMatrix matrix)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height)
        return;

    const float *m = matrix.d2i;
    float src_x = m[0] * dx + m[1] * dy + m[2];
    float src_y = m[3] * dx + m[4] * dy + m[5];

    // 默认填充值（归一化后的灰度）
    const float inv_255 = 1.0f / 255.0f;
    const float fill_norm = fill_value * inv_255;
    float r = fill_norm, g = fill_norm, b = fill_norm;

    // 只有当映射点完全落在原图有效区域（且其四个邻居都存在）时才进行插值
    // 避免边缘混合，保证填充区域纯色
    if (src_x >= 0 && src_x < src_width - 1 && src_y >= 0 && src_y < src_height - 1)
    {
        int y_low = __float2int_rd(src_y);
        int x_low = __float2int_rd(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        // 获取四个邻居的指针（输入为 BGR 顺序）
        const uint8_t *v1 = src + y_low * src_width * 3 + x_low * 3;
        const uint8_t *v2 = src + y_low * src_width * 3 + x_high * 3;
        const uint8_t *v3 = src + y_high * src_width * 3 + x_low * 3;
        const uint8_t *v4 = src + y_high * src_width * 3 + x_high * 3;

        // 对每个通道插值（B, G, R）
        float b_val = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        float g_val = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        float r_val = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];

        // 归一化并转换 RGB 顺序
        r = r_val * inv_255;
        g = g_val * inv_255;
        b = b_val * inv_255;
    }
    // 否则直接使用填充值（已初始化为 fill_norm）

    // 输出 CHW 布局：通道连续，每个通道是一整张 H*W 图像
    int idx = dy * dst_width + dx;
    int stride = dst_height * dst_width;
    dst[idx] = r;              // R 通道
    dst[1 * stride + idx] = g; // G 通道
    dst[2 * stride + idx] = b; // B 通道
}

bool CudaPreprocessor::init(int src_w, int src_h, int net_w, int net_h)
{
    // 计算映射与逆映射矩阵
    matrix_.compute(src_w, src_h, net_w, net_h);

    // letterbox信息
    lb_ = {matrix_.i2d[0], static_cast<int>(matrix_.i2d[2]),
           static_cast<int>(matrix_.i2d[5]), src_w, src_h, net_w, net_h};

    // 输出张量分配显存
    size_t sz = 1ull * 3 * net_h * net_w * sizeof(float);
    cudaError_t err = cudaMalloc(&output_dev_, sz);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[PreProc] cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

float *CudaPreprocessor::process(const void *input_dev, cudaStream_t stream)
{
    const auto &lb = lb_;
    dim3 block_size(32, 32);
    dim3 grid_size((lb.dst_w + 31) / 32, (lb.dst_h + 31) / 32);

    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(input_dev), lb.src_w, lb.src_h,
        output_dev_, lb.dst_w, lb.dst_h, 114, matrix_);

    return output_dev_;
}

CudaPreprocessor::~CudaPreprocessor() { destroy(); }

void CudaPreprocessor::destroy()
{
    if (output_dev_)
    {
        cudaFree(output_dev_);
        output_dev_ = nullptr;
    }
}