使用tensorrt-10在nvidia jetson orin nano部署YOLO26实现多线程推理

1.导出engine
python3 scripts/export_engine.py \
        --weights models/best.pt \
        --output  models/best.engine \
        --imgsz   640

2.编译（注意修改CMakeLists.txt中cuda和tensorrt路径）
  mkdir build && cd build
  cmake ..
  make

3.视频加载推理
./main ../models/yolo26n.engine VIDEO_PATH

推理前使用performance.sh锁定cpu和gpu频率（该脚本仅适用于jetson orin nano）
处理速度最高60fps
