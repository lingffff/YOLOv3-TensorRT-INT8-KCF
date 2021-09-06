#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "yololayer.h"
#include <cmath>

#include "BatchStream.h"
#include "EntropyCalibrator.h"

#define DEVICE 0  // GPU id

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = 1000 * 7 + 1;  // we assume the yololayer outputs no more than 1000 boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;


// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dataType) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../redball.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto lr0 = convBnLeaky(network, weightMap, *data, 16, 3, 1, 1, 0);
    auto pool1 = network->addPoolingNd(*lr0->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool1->setStrideNd(DimsHW{2, 2});
    auto lr2 = convBnLeaky(network, weightMap, *pool1->getOutput(0), 32, 3, 1, 1, 2);
    auto pool3 = network->addPoolingNd(*lr2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool3->setStrideNd(DimsHW{2, 2});
    auto lr4 = convBnLeaky(network, weightMap, *pool3->getOutput(0), 64, 3, 1, 1, 4);
    auto pool5 = network->addPoolingNd(*lr4->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool5->setStrideNd(DimsHW{2, 2});
    auto lr6 = convBnLeaky(network, weightMap, *pool5->getOutput(0), 128, 3, 1, 1, 6);
    auto pool7 = network->addPoolingNd(*lr6->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool7->setStrideNd(DimsHW{2, 2});
    auto lr8 = convBnLeaky(network, weightMap, *pool7->getOutput(0), 256, 3, 1, 1, 8);
    auto pool9 = network->addPoolingNd(*lr8->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool9->setStrideNd(DimsHW{2, 2});
    auto lr10 = convBnLeaky(network, weightMap, *pool9->getOutput(0), 512, 3, 1, 1, 10);
    auto pad11 = network->addPaddingNd(*lr10->getOutput(0), DimsHW{0, 0}, DimsHW{1, 1});
    auto pool11 = network->addPoolingNd(*pad11->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
    pool11->setStrideNd(DimsHW{1, 1});
    auto lr12 = convBnLeaky(network, weightMap, *pool11->getOutput(0), 1024, 3, 1, 1, 12);
    auto lr13 = convBnLeaky(network, weightMap, *lr12->getOutput(0), 256, 1, 1, 0, 13);
    auto lr14 = convBnLeaky(network, weightMap, *lr13->getOutput(0), 512, 3, 1, 1, 14);
    IConvolutionLayer* conv15 = network->addConvolutionNd(*lr14->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.15.Conv2d.weight"], weightMap["module_list.15.Conv2d.bias"]);
    // 16 is yolo
    auto l17 = lr13;
    auto lr18 = convBnLeaky(network, weightMap, *l17->getOutput(0), 128, 1, 1, 0, 18);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 128 * 2 * 2));
    for (int i = 0; i < 128 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts19{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv19 = network->addDeconvolutionNd(*lr18->getOutput(0), 128, DimsHW{2, 2}, deconvwts19, emptywts);
    assert(deconv19);
    deconv19->setStrideNd(DimsHW{2, 2});
    deconv19->setNbGroups(128);
    weightMap["deconv19"] = deconvwts19;

    ITensor* inputTensors[] = {deconv19->getOutput(0), lr8->getOutput(0)};
    auto cat20 = network->addConcatenation(inputTensors, 2);
    auto lr21 = convBnLeaky(network, weightMap, *cat20->getOutput(0), 256, 3, 1, 1, 21);
    IConvolutionLayer* conv22 = network->addConvolutionNd(*lr21->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.22.Conv2d.weight"], weightMap["module_list.22.Conv2d.bias"]);
    // 22 is yolo

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {conv15->getOutput(0), conv22->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 2, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
    // FP16 or INT8
    if (dataType == DataType::kHALF)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (dataType == DataType::kINT8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }
    // Dims inputDims{4, maxBatchSize, 3, INPUT_H, INPUT_W};
    // IOptimizationProfile* profile = builder->createOptimizationProfile();
    // profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMAX, inputDims);
    // profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kMIN, inputDims);
    // profile->setDimensions(INPUT_BLOB_NAME, OptProfileSelector::kOPT, inputDims);
    // config->addOptimizationProfile(profile);

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;
    if (dataType == DataType::kINT8)
    { 
        RedballBatchStream calibrationStream(10, 50, "redball/labels.txt", "../yolov3/");
        calibrator.reset(new Int8EntropyCalibrator2<RedballBatchStream>(
            calibrationStream, 0, "yolov3-tiny", INPUT_BLOB_NAME));
        config->setInt8Calibrator(calibrator.get());
        // config->setCalibrationProfile(profile);
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, DataType dataType) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine;
    engine = createEngine(maxBatchSize, builder, config, dataType);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}



int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    DataType buildDataType;
    if (argc == 3 && std::string(argv[1]) == "-s" && std::string(argv[2]) == "int8") {
        buildDataType = DataType::kINT8;
    }
    else if (argc == 3 && std::string(argv[1]) == "-s" && std::string(argv[2]) == "fp16") {
        buildDataType = DataType::kHALF;
    }
    else {
        buildDataType = DataType::kFLOAT;
    }
    IHostMemory* modelStream{nullptr};
    APIToModel(1, &modelStream, buildDataType);
    assert(modelStream != nullptr);
    std::ofstream p("yolov3-tiny.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file (engine)." << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0; 
}