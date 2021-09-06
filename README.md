# YOLOv3tiny-TensorRT-KCF

## Description
**YOLOv3tiny-TensorRT-KCF** is a TensorRT Int8-Quantization implementation of YOLOv3-tiny on NVIDIA Jetson Xavier NX Board. The dataset we provide is about a red ball. So we also use this to drive a car to catch the red ball, along with KCF, a traditional Object Tracking method.

## Tutorials
### 1. Train
On GPU server, not NX.
```bash
git clone https://github.com/lingffff/YOLOv3tiny-TensorRT-KCF.git
cd YOLOv3tiny-TensorRT-KCF
cd yolov3
```
Download redball dataset from [here](), unzip and replace the folder **redball**. Then start training.
```bash
python train.py --device 0
```
### 2. Transfer
Now we get YOLOv3-tiny weights in **weights/best.pt**. Transfer it to binary file **redball.wts**, which convert weights to TensorRT for building inference engine.
```bash
python gen_wts.py
```
Then copy **output/redball.wts** to NX Board.
### 3. Build Engine
On NX Platform below.
```bash
git clone https://github.com/lingffff/YOLOv3tiny-TensorRT-KCF.git
cd YOLOv3tiny-TensorRT-KCF
```
Build the project.
```bash
mkdir build
cd build
cmake ..
make
```
Now we get executable file **build_engine** and **detect**.  
Run **build_engine**. Use **-s** argument to specify quantization options: int8, fp16, fp32(default).
```bash
./build_engine -s int8
```
### 4. Inference
Run **detect** to detect pictures or camera video. You can also check KCF tracking method here by other options below.  
```bash
./detect -p ../samples
```
Options:  
| Argument | Description |
| :-: | :-: |
| -p \<folder\> | Detect pictures in the folder. |
| -v  | Detect camera video stream.  |
| -t  | Detect video along with KCF tracking method.  |

## Benchmark
| Models | Device | BatchSize |	Mode | Input Size | Speed |
| :-: | :-: | :-: | :-: | :-: | :-:  |
| YOLOv3-tiny | NX | 1 | FP32 | 416x416 | 26ms |
| YOLOv3-tiny | NX | 1 | FP16 | 416x416 | 19ms |
| YOLOv3-tiny | NX | 1 | INT8 | 416x416 | 20ms |

## TODO
- [ ] Convert weights to TensorRT by more common way, like ONNX.
- [ ] Run detection and tracking in a multi-thread way.
- [ ] Learn the source codes of TensorRT and implement an infrerence framework myself.

## Acknowledge

YOLOv3 Pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3).  
YOLOv3 TensorRT implementation from [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx).  
TensorRT Int8 implementation from [NVIDIA/TensorRT/samples/sampleINT8](https://github.com/NVIDIA/TensorRT/tree/master/samples/sampleINT8).  

With my sincerely appreciation!

## About me  
Just call me **Al** (not ai but al. LOL.) / Albert / Ling Feng (in Chinese, pronounces like ***lin-phone***).  
E-mail: ling@stu.pku.edu.cn  
Gitee: https://gitee.com/lingff  
CSDN: https://blog.csdn.net/weixin_43214408  