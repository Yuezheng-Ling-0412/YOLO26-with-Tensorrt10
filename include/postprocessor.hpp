#pragma once
#include <vector>
#include <string>

struct Detection
{
    int x1, y1, x2, y2; // 原始图像坐标（已还原 letterbox）
    float confidence;
    int class_id;
};

class PostProcessor
{
public:
    PostProcessor() = default;

    void init(float conf_thresh = 0.4f, int num_classes = 80, int max_dets = 64);

    int postprocess(float *output_host, std::vector<Detection> &dets,
                    float scale, int pad_x, int pad_y, int img_w, int img_h);

private:
    float conf_thresh_ = 0.4f;             // 置信度阈值
    int num_classes_ = 80;                 // 类别总数
    int max_dets_ = 64;                    // 最大检测目标数
};
