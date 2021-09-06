/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

#include <opencv2/opencv.hpp>
#include "yololayer.h"
static const int CAL_INPUT_H = Yolo::INPUT_H;
static const int CAL_INPUT_W = Yolo::INPUT_W;

class IBatchStream
{
public:
    virtual void reset(int firstBatch) = 0;
    virtual bool next() = 0;
    virtual void skip(int skipCount) = 0;
    virtual float* getBatch() = 0;
    virtual float* getLabels() = 0;
    virtual int getBatchesRead() const = 0;
    virtual int getBatchSize() const = 0;
    virtual nvinfer1::Dims getDims() const = 0;
};

class RedballBatchStream : public IBatchStream
{
public:
    RedballBatchStream(int batchSize, int maxBatches, const std::string& dataFile, const std::string& directories)
        : baseDirectory{directories}
        , mBatchSize{batchSize}
        , mMaxBatches{maxBatches}
        , mDims{3, 3, CAL_INPUT_H, CAL_INPUT_W} // dimensions: n_dims, dims...
    {
        int numElements = mBatchSize * mDims.d[0] * mDims.d[1] * mDims.d[2];
        batchData.resize(numElements);
        readTrainFile(baseDirectory + dataFile);
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount >= mMaxBatches)
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {
        std::string imgPath;
        std::vector<float> oneImg;
        cv::Mat img, imgPreprocessed;

        for (int i = 0; i < mBatchSize; i++) {
            imgPath = baseDirectory + imgNames[(mBatchCount - 1) * mBatchSize + i];
            img = cv::imread(imgPath.c_str());
            std::cout << "Calibrating with " << imgPath << std::endl;
            imgPreprocessed = preprocessImg(img);
            oneImg = (std::vector<float>)imgPreprocessed.reshape(1, 1);
            for (unsigned int j = 0; j < oneImg.size(); j++) {
                oneImg[j] /= 255.f;
            }
            batchData.insert(batchData.begin(), oneImg.begin(), oneImg.end());
        }
        return batchData.data();
    }

    float* getLabels() override
    {
        return nullptr;
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return nvinfer1::Dims{4, {mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]}, {}};
    }

private:
    cv::Mat preprocessImg(cv::Mat& img) 
    {
        int w, h, x, y;
        float r_w = CAL_INPUT_W / (img.cols*1.0);
        float r_h = CAL_INPUT_H / (img.rows*1.0);
        if (r_h > r_w) {
            w = CAL_INPUT_W;
            h = r_w * img.rows;
            x = 0;
            y = (CAL_INPUT_H - h) / 2;
        } else {
            w = r_h* img.cols;
            h = CAL_INPUT_H;
            x = (CAL_INPUT_W - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
        cv::Mat out(CAL_INPUT_H, CAL_INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        return out;
    }

    void readTrainFile(const std::string& trainTxtPath)
    {
        std::ifstream file(trainTxtPath.c_str());
        std::string line;
        for (int i = 0; i < mBatchSize * mMaxBatches; i++) {
            getline(file, line); 
            imgNames.push_back(line);
        }
    }

    int mCurrentIndex{0};
    std::vector<std::string> imgNames{};
    std::string baseDirectory{};
    std::vector<float> batchData{};

    int mBatchSize{0};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    nvinfer1::Dims mDims{};
};

#endif
