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

#ifdef TINY
    std::map<std::string, Weights> weightMap = loadWeights("../redball-tiny.wts");
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
#else
    std::map<std::string, Weights> weightMap = loadWeights("../redball.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    
    auto lr0 = convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
    auto lr1 = convBnLeaky(network, weightMap, *lr0->getOutput(0), 64, 3, 2, 1, 1);
    auto lr2 = convBnLeaky(network, weightMap, *lr1->getOutput(0), 32, 1, 1, 0, 2);
    auto lr3 = convBnLeaky(network, weightMap, *lr2->getOutput(0), 64, 3, 1, 1, 3);
    auto ew4 = network->addElementWise(*lr3->getOutput(0), *lr1->getOutput(0), ElementWiseOperation::kSUM);
    auto lr5 = convBnLeaky(network, weightMap, *ew4->getOutput(0), 128, 3, 2, 1, 5);
    auto lr6 = convBnLeaky(network, weightMap, *lr5->getOutput(0), 64, 1, 1, 0, 6);
    auto lr7 = convBnLeaky(network, weightMap, *lr6->getOutput(0), 128, 3, 1, 1, 7);
    auto ew8 = network->addElementWise(*lr7->getOutput(0), *lr5->getOutput(0), ElementWiseOperation::kSUM);
    auto lr9 = convBnLeaky(network, weightMap, *ew8->getOutput(0), 64, 1, 1, 0, 9);
    auto lr10 = convBnLeaky(network, weightMap, *lr9->getOutput(0), 128, 3, 1, 1, 10);
    auto ew11 = network->addElementWise(*lr10->getOutput(0), *ew8->getOutput(0), ElementWiseOperation::kSUM);
    auto lr12 = convBnLeaky(network, weightMap, *ew11->getOutput(0), 256, 3, 2, 1, 12);
    auto lr13 = convBnLeaky(network, weightMap, *lr12->getOutput(0), 128, 1, 1, 0, 13);
    auto lr14 = convBnLeaky(network, weightMap, *lr13->getOutput(0), 256, 3, 1, 1, 14);
    auto ew15 = network->addElementWise(*lr14->getOutput(0), *lr12->getOutput(0), ElementWiseOperation::kSUM);
    auto lr16 = convBnLeaky(network, weightMap, *ew15->getOutput(0), 128, 1, 1, 0, 16);
    auto lr17 = convBnLeaky(network, weightMap, *lr16->getOutput(0), 256, 3, 1, 1, 17);
    auto ew18 = network->addElementWise(*lr17->getOutput(0), *ew15->getOutput(0), ElementWiseOperation::kSUM);
    auto lr19 = convBnLeaky(network, weightMap, *ew18->getOutput(0), 128, 1, 1, 0, 19);
    auto lr20 = convBnLeaky(network, weightMap, *lr19->getOutput(0), 256, 3, 1, 1, 20);
    auto ew21 = network->addElementWise(*lr20->getOutput(0), *ew18->getOutput(0), ElementWiseOperation::kSUM);
    auto lr22 = convBnLeaky(network, weightMap, *ew21->getOutput(0), 128, 1, 1, 0, 22);
    auto lr23 = convBnLeaky(network, weightMap, *lr22->getOutput(0), 256, 3, 1, 1, 23);
    auto ew24 = network->addElementWise(*lr23->getOutput(0), *ew21->getOutput(0), ElementWiseOperation::kSUM);
    auto lr25 = convBnLeaky(network, weightMap, *ew24->getOutput(0), 128, 1, 1, 0, 25);
    auto lr26 = convBnLeaky(network, weightMap, *lr25->getOutput(0), 256, 3, 1, 1, 26);
    auto ew27 = network->addElementWise(*lr26->getOutput(0), *ew24->getOutput(0), ElementWiseOperation::kSUM);
    auto lr28 = convBnLeaky(network, weightMap, *ew27->getOutput(0), 128, 1, 1, 0, 28);
    auto lr29 = convBnLeaky(network, weightMap, *lr28->getOutput(0), 256, 3, 1, 1, 29);
    auto ew30 = network->addElementWise(*lr29->getOutput(0), *ew27->getOutput(0), ElementWiseOperation::kSUM);
    auto lr31 = convBnLeaky(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
    auto lr32 = convBnLeaky(network, weightMap, *lr31->getOutput(0), 256, 3, 1, 1, 32);
    auto ew33 = network->addElementWise(*lr32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
    auto lr34 = convBnLeaky(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
    auto lr35 = convBnLeaky(network, weightMap, *lr34->getOutput(0), 256, 3, 1, 1, 35);
    auto ew36 = network->addElementWise(*lr35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
    auto lr37 = convBnLeaky(network, weightMap, *ew36->getOutput(0), 512, 3, 2, 1, 37);
    auto lr38 = convBnLeaky(network, weightMap, *lr37->getOutput(0), 256, 1, 1, 0, 38);
    auto lr39 = convBnLeaky(network, weightMap, *lr38->getOutput(0), 512, 3, 1, 1, 39);
    auto ew40 = network->addElementWise(*lr39->getOutput(0), *lr37->getOutput(0), ElementWiseOperation::kSUM);
    auto lr41 = convBnLeaky(network, weightMap, *ew40->getOutput(0), 256, 1, 1, 0, 41);
    auto lr42 = convBnLeaky(network, weightMap, *lr41->getOutput(0), 512, 3, 1, 1, 42);
    auto ew43 = network->addElementWise(*lr42->getOutput(0), *ew40->getOutput(0), ElementWiseOperation::kSUM);
    auto lr44 = convBnLeaky(network, weightMap, *ew43->getOutput(0), 256, 1, 1, 0, 44);
    auto lr45 = convBnLeaky(network, weightMap, *lr44->getOutput(0), 512, 3, 1, 1, 45);
    auto ew46 = network->addElementWise(*lr45->getOutput(0), *ew43->getOutput(0), ElementWiseOperation::kSUM);
    auto lr47 = convBnLeaky(network, weightMap, *ew46->getOutput(0), 256, 1, 1, 0, 47);
    auto lr48 = convBnLeaky(network, weightMap, *lr47->getOutput(0), 512, 3, 1, 1, 48);
    auto ew49 = network->addElementWise(*lr48->getOutput(0), *ew46->getOutput(0), ElementWiseOperation::kSUM);
    auto lr50 = convBnLeaky(network, weightMap, *ew49->getOutput(0), 256, 1, 1, 0, 50);
    auto lr51 = convBnLeaky(network, weightMap, *lr50->getOutput(0), 512, 3, 1, 1, 51);
    auto ew52 = network->addElementWise(*lr51->getOutput(0), *ew49->getOutput(0), ElementWiseOperation::kSUM);
    auto lr53 = convBnLeaky(network, weightMap, *ew52->getOutput(0), 256, 1, 1, 0, 53);
    auto lr54 = convBnLeaky(network, weightMap, *lr53->getOutput(0), 512, 3, 1, 1, 54);
    auto ew55 = network->addElementWise(*lr54->getOutput(0), *ew52->getOutput(0), ElementWiseOperation::kSUM);
    auto lr56 = convBnLeaky(network, weightMap, *ew55->getOutput(0), 256, 1, 1, 0, 56);
    auto lr57 = convBnLeaky(network, weightMap, *lr56->getOutput(0), 512, 3, 1, 1, 57);
    auto ew58 = network->addElementWise(*lr57->getOutput(0), *ew55->getOutput(0), ElementWiseOperation::kSUM);
    auto lr59 = convBnLeaky(network, weightMap, *ew58->getOutput(0), 256, 1, 1, 0, 59);
    auto lr60 = convBnLeaky(network, weightMap, *lr59->getOutput(0), 512, 3, 1, 1, 60);
    auto ew61 = network->addElementWise(*lr60->getOutput(0), *ew58->getOutput(0), ElementWiseOperation::kSUM);
    auto lr62 = convBnLeaky(network, weightMap, *ew61->getOutput(0), 1024, 3, 2, 1, 62);
    auto lr63 = convBnLeaky(network, weightMap, *lr62->getOutput(0), 512, 1, 1, 0, 63);
    auto lr64 = convBnLeaky(network, weightMap, *lr63->getOutput(0), 1024, 3, 1, 1, 64);
    auto ew65 = network->addElementWise(*lr64->getOutput(0), *lr62->getOutput(0), ElementWiseOperation::kSUM);
    auto lr66 = convBnLeaky(network, weightMap, *ew65->getOutput(0), 512, 1, 1, 0, 66);
    auto lr67 = convBnLeaky(network, weightMap, *lr66->getOutput(0), 1024, 3, 1, 1, 67);
    auto ew68 = network->addElementWise(*lr67->getOutput(0), *ew65->getOutput(0), ElementWiseOperation::kSUM);
    auto lr69 = convBnLeaky(network, weightMap, *ew68->getOutput(0), 512, 1, 1, 0, 69);
    auto lr70 = convBnLeaky(network, weightMap, *lr69->getOutput(0), 1024, 3, 1, 1, 70);
    auto ew71 = network->addElementWise(*lr70->getOutput(0), *ew68->getOutput(0), ElementWiseOperation::kSUM);
    auto lr72 = convBnLeaky(network, weightMap, *ew71->getOutput(0), 512, 1, 1, 0, 72);
    auto lr73 = convBnLeaky(network, weightMap, *lr72->getOutput(0), 1024, 3, 1, 1, 73);
    auto ew74 = network->addElementWise(*lr73->getOutput(0), *ew71->getOutput(0), ElementWiseOperation::kSUM);
    auto lr75 = convBnLeaky(network, weightMap, *ew74->getOutput(0), 512, 1, 1, 0, 75);
    auto lr76 = convBnLeaky(network, weightMap, *lr75->getOutput(0), 1024, 3, 1, 1, 76);
    auto lr77 = convBnLeaky(network, weightMap, *lr76->getOutput(0), 512, 1, 1, 0, 77);
    auto lr78 = convBnLeaky(network, weightMap, *lr77->getOutput(0), 1024, 3, 1, 1, 78);
    auto lr79 = convBnLeaky(network, weightMap, *lr78->getOutput(0), 512, 1, 1, 0, 79);
    auto lr80 = convBnLeaky(network, weightMap, *lr79->getOutput(0), 1024, 3, 1, 1, 80);
    IConvolutionLayer* conv81 = network->addConvolutionNd(*lr80->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.81.Conv2d.weight"], weightMap["module_list.81.Conv2d.bias"]);
    assert(conv81);
    // 82 is yolo
    auto l83 = lr79;
    auto lr84 = convBnLeaky(network, weightMap, *l83->getOutput(0), 256, 1, 1, 0, 84);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts85{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv85 = network->addDeconvolutionNd(*lr84->getOutput(0), 256, DimsHW{2, 2}, deconvwts85, emptywts);
    assert(deconv85);
    deconv85->setStrideNd(DimsHW{2, 2});
    deconv85->setNbGroups(256);
    weightMap["deconv85"] = deconvwts85;

    ITensor* inputTensors[] = {deconv85->getOutput(0), ew61->getOutput(0)};
    auto cat86 = network->addConcatenation(inputTensors, 2);
    auto lr87 = convBnLeaky(network, weightMap, *cat86->getOutput(0), 256, 1, 1, 0, 87);
    auto lr88 = convBnLeaky(network, weightMap, *lr87->getOutput(0), 512, 3, 1, 1, 88);
    auto lr89 = convBnLeaky(network, weightMap, *lr88->getOutput(0), 256, 1, 1, 0, 89);
    auto lr90 = convBnLeaky(network, weightMap, *lr89->getOutput(0), 512, 3, 1, 1, 90);
    auto lr91 = convBnLeaky(network, weightMap, *lr90->getOutput(0), 256, 1, 1, 0, 91);
    auto lr92 = convBnLeaky(network, weightMap, *lr91->getOutput(0), 512, 3, 1, 1, 92);
    IConvolutionLayer* conv93 = network->addConvolutionNd(*lr92->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.93.Conv2d.weight"], weightMap["module_list.93.Conv2d.bias"]);
    assert(conv93);
    // 94 is yolo
    auto l95 = lr91;
    auto lr96 = convBnLeaky(network, weightMap, *l95->getOutput(0), 128, 1, 1, 0, 96);
    Weights deconvwts97{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv97 = network->addDeconvolutionNd(*lr96->getOutput(0), 128, DimsHW{2, 2}, deconvwts97, emptywts);
    assert(deconv97);
    deconv97->setStrideNd(DimsHW{2, 2});
    deconv97->setNbGroups(128);
    ITensor* inputTensors1[] = {deconv97->getOutput(0), ew36->getOutput(0)};
    auto cat98 = network->addConcatenation(inputTensors1, 2);
    auto lr99 = convBnLeaky(network, weightMap, *cat98->getOutput(0), 128, 1, 1, 0, 99);
    auto lr100 = convBnLeaky(network, weightMap, *lr99->getOutput(0), 256, 3, 1, 1, 100);
    auto lr101 = convBnLeaky(network, weightMap, *lr100->getOutput(0), 128, 1, 1, 0, 101);
    auto lr102 = convBnLeaky(network, weightMap, *lr101->getOutput(0), 256, 3, 1, 1, 102);
    auto lr103 = convBnLeaky(network, weightMap, *lr102->getOutput(0), 128, 1, 1, 0, 103);
    auto lr104 = convBnLeaky(network, weightMap, *lr103->getOutput(0), 256, 3, 1, 1, 104);
    IConvolutionLayer* conv105 = network->addConvolutionNd(*lr104->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.105.Conv2d.weight"], weightMap["module_list.105.Conv2d.bias"]);
    assert(conv105);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {conv81->getOutput(0), conv93->getOutput(0), conv105->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
#endif
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
            calibrationStream, 0, "Yolov3", INPUT_BLOB_NAME));
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
#ifdef TINY
    std::ofstream p("yolov3-tiny.engine", std::ios::binary);
#else
    std::ofstream p("yolov3.engine", std::ios::binary);
#endif
    if (!p) {
        std::cerr << "could not open plan output file (engine)." << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();
    return 0; 
}