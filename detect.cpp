#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "yololayer.h"

#include <opencv2/tracking.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.5
#define BBOX_CONF_THRESH 0.4

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = 1000 * 7 + 1;  // we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

cv::Mat preprocess_img(cv::Mat& img) {
    int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h* img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh = NMS_THRESH) {
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH) continue;
        Yolo::Detection det;
        memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}


void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    std::vector<std::string> file_names;
    int mode = 0;
    if (argc == 3 && std::string(argv[1]) == "-d") {
        mode = 1;
        if (read_files_in_dir(argv[2], file_names) < 0) {
            std::cout << "Read image files failed." << std::endl;
            return -1;
        }
    } else if (argc >= 2 && std::string(argv[1]) == "-v") {
        mode = 2;
    } else if (argc >= 2 && std::string(argv[1]) == "-t") {
        mode = 3;
    } else {
        std::cerr << "Wrong arguments!" << std::endl;
        return -1;
    }
#ifdef TINY
    std::ifstream file("yolov3-tiny.engine", std::ios::binary);
#else
    std::ifstream file("yolov3.engine", std::ios::binary);
#endif
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    // prepare input data ---------------------------
    static float data[3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    static float prob[OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    cv::Mat img, pr_img;
    std::vector<Yolo::Detection> res;

    if (mode == 1) {
        int fcount = 0;
        for (auto f: file_names) {
            fcount++;
            std::cout << fcount << "  " << f << std::endl;
            img = cv::imread(std::string(argv[2]) + "/" + f);
            if (img.empty()) continue;
            pr_img = preprocess_img(img);
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
            }

            // Run inference
            auto start = std::chrono::system_clock::now();
            doInference(*context, data, prob, 1);
            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
            nms(res, prob);
            for (int i=0; i<20; i++) {
                std::cout << prob[i] << ",";
            }
            std::cout << res.size() << std::endl;
            for (size_t j = 0; j < res.size(); j++) {
                float *p = (float*)&res[j];
                for (size_t k = 0; k < 7; k++) {
                    std::cout << p[k] << ", ";
                }
                std::cout << std::endl;
                cv::Rect2d r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            res.clear();
            cv::imwrite("_" + f, img);
        }
    }
    else if (mode == 2) {
        // set input video
	    cv::VideoCapture cap(0); 
        while (true) {
            cap >> img;
            // clock_gettime(CLOCK_MONOTONIC, &t1);
            pr_img = preprocess_img(img);
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
            }
            // Run inference
            doInference(*context, data, prob, 1);
            nms(res, prob);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect2d r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(255, 0, 0), 2);
            }
            // clock_gettime(CLOCK_MONOTONIC, &t2);
            res.clear();
            cv::imshow("camera", img);
            if (cv::waitKey(1) == 27 || cv::waitKey(1) == 113) break;
        }
    }
    else if (mode == 3) {
        // set input video
	    cv::VideoCapture cap(0); 
        // init KCF
        cv::TrackerKCF::Params param;
        param.detect_thresh = 0.5f;
        cv::Ptr<cv::TrackerKCF> tracker = cv::TrackerKCF::create(param);
        do { // test first detect to init tracker
            cap >> img;
            pr_img = preprocess_img(img);
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
            }
            // Run inference
            doInference(*context, data, prob, 1);
            nms(res, prob);
            cv::imshow("camera", img);
            if (cv::waitKey(1) == 27 || cv::waitKey(1) == 113) break;
        } while(res.empty());
        cv::Rect2d r = get_rect(img, res[0].bbox);
        // initialize the tracker
        tracker->init(img, r);

        bool tracker_valid = false, detect_valid = false;
        while (true) {
            cap >> img;
            // clock_gettime(CLOCK_MONOTONIC, &t1);
            tracker_valid = tracker->update(img, r);
            if (!tracker_valid) {  // cannot track the object, then detect        
                pr_img = preprocess_img(img);
                for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                    data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
                    data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
                    data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
                }
                // Run inference
                doInference(*context, data, prob, 1);
                res.clear();
                nms(res, prob);
                detect_valid = (bool) !res.empty();
                if(detect_valid) r = get_rect(img, res[0].bbox);
            }
            printf("tracker status: %d, detect status: %d\n", tracker_valid, detect_valid);
            // clock_gettime(CLOCK_MONOTONIC, &t2);
            if (tracker_valid || detect_valid) {
                cv::rectangle(img, r, cv::Scalar(255, 0, 0), 2);
            }
            cv::imshow("camera", img);
            if (cv::waitKey(1) == 27 || cv::waitKey(1) == 113) break;
        }
    }
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}