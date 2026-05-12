#include <sys/time.h>
#include <opencv2/opencv.hpp>

#include "trtpool.hpp"
#include "threadpool.hpp"

int main(int argc, char **argv)
{
    char *engine_path = NULL;
    if (argc != 3)
    {
        printf("Usage: %s <trt_engine_path> <video_path> \n", argv[0]);
        return -1;
    }
    engine_path = (char *)argv[1]; // 参数二，trt引擎所在路径
    char *image_name = argv[2];    // 参数三, 视频/摄像头
    printf("engine_path:\t%s\n", engine_path);

    cv::VideoCapture capture;
    cv::namedWindow("Camera FPS");
    if (strlen(image_name) == 1)
        capture.open((int)(image_name[0] - '0'));
    else
        capture.open(image_name);

    // 设置线程数
    int n = 3, frames = 0;
    printf("线程数:\t%d\n", n);
    // 类似于多个rk模型的集合?
    std::vector<trt_infer *> trtpool;
    // 线程池
    dpool::ThreadPool pool(n);
    // 线程队列
    std::queue<std::future<int>> futs;

    // trt_infer初始化需要原始图像宽高数据，读取第一帧获取图像尺寸
    cv::Mat first_frame;
    capture >> first_frame;
    if (first_frame.empty())
    {
        fprintf(stderr, "Failed to read first frame\n");
        return -1;
    }

    int src_w = first_frame.cols;
    int src_h = first_frame.rows;

    for (int i = 0; i < n; i++)
    {
        trt_infer *ptr = new trt_infer(engine_path, src_w, src_h);
        trtpool.push_back(ptr);
        capture >> ptr->img;
        futs.push(pool.submit(&trt_infer::infer, &(*ptr)));
    }

    struct timeval time;
    gettimeofday(&time, nullptr);
    auto initTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    gettimeofday(&time, nullptr);
    long tmpTime, lopTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    while (capture.isOpened())
    {
        if (futs.front().get() != 0)
            break;
        futs.pop();
        cv::imshow("Camera FPS", trtpool[frames % n]->img);
        if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出
            break;
        if (!capture.read(trtpool[frames % n]->img)) // 没有更多帧了
            break;
        futs.push(pool.submit(&trt_infer::infer, &(*trtpool[frames++ % n])));

        if (frames % 60 == 0)
        {
            gettimeofday(&time, nullptr);
            tmpTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("60帧平均帧率:\t%f帧\n", 60000.0 / (float)(tmpTime - lopTime));
            lopTime = tmpTime;
        }
    }

    gettimeofday(&time, nullptr);
    printf("\n平均帧率:\t%f帧\n", float(frames) / (float)(time.tv_sec * 1000 + time.tv_usec / 1000 - initTime + 0.0001) * 1000.0);

    // 释放剩下的资源
    while (!futs.empty())
    {
        if (futs.front().get())
            break;
        futs.pop();
    }
    for (int i = 0; i < n; i++)
        delete trtpool[i];
    capture.release();
    cv::destroyAllWindows();
    return 0;
}
