#include "postprocessor.hpp"
#include <algorithm>
#include <cmath>

void PostProcessor::init(float conf_thresh, int num_classes, int max_dets)
{
    conf_thresh_ = conf_thresh;
    num_classes_ = num_classes;
    max_dets_ = max_dets;
}

int PostProcessor::postprocess(float *output_host, std::vector<Detection> &dets,
                               float scale, int pad_x, int pad_y, int img_w, int img_h)
{
    dets.clear();
    dets.reserve(max_dets_);

    // 注意：模型输出 300 个检测框，每个 6 个 float（xyxy, conf, class_id）
    const int num_boxes = 300;
    for (int i = 0; i < num_boxes && dets.size() < max_dets_; ++i)
    {
        float *det = output_host + i * 6;
        float conf = det[4];
        if (conf < conf_thresh_)
            continue;

        float x1_f = (det[0] - pad_x) / scale;
        float y1_f = (det[1] - pad_y) / scale;
        float x2_f = (det[2] - pad_x) / scale;
        float y2_f = (det[3] - pad_y) / scale;

        // 裁剪到图像边界并四舍五入取整
        int x1 = static_cast<int>(std::round(std::max(0.0f, std::min(x1_f, static_cast<float>(img_w)))));
        int y1 = static_cast<int>(std::round(std::max(0.0f, std::min(y1_f, static_cast<float>(img_h)))));
        int x2 = static_cast<int>(std::round(std::max(0.0f, std::min(x2_f, static_cast<float>(img_w)))));
        int y2 = static_cast<int>(std::round(std::max(0.0f, std::min(y2_f, static_cast<float>(img_h)))));
        if (x1 > x2)
            std::swap(x1, x2);
        if (y1 > y2)
            std::swap(y1, y2);
        if (x1 == x2 || y1 == y2)
            continue; // 跳过退化框

        dets.push_back({x1, y1, x2, y2, conf, static_cast<int>(det[5])});
    }
    return static_cast<int>(dets.size());
}
