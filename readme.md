使用tensorrt-10在nvidia jetson nano部署YOLO26实现多线程推理

1.导出engine
python3 scripts/export_engine.py \
        --weights models/best.pt \
        --output  models/best.engine \
        --imgsz   640

2.编译
  mkdir build && cd build
  cmake ..
  make

实现视频加载推理
./main ../models/yolo26n.engine VIDEO_PATH

处理速度最高60fps
