#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>
#include <assert.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>

#include "yolov8_utils.h"


// 以下示例捕获所有警告消息，但忽略信息性消息
class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // 抑制信息级别的消息
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// 加载模型文件
std::vector<unsigned char> load_engine_file(const std::string &file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
    return engine_data;
}


int main(int argc, char **argv)
{
    // 实例化ILogger
    Logger logger;

    std::unique_ptr<nvinfer1::IRuntime> runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (runtime == nullptr){return false;}
    
    std::string file_path = "yolov8s.engine";
    auto plan = load_engine_file(file_path);

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    if (engine == nullptr){return false;}

    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (context == nullptr){return false;}

    auto idims = engine->getTensorShape("images");
    auto odims = engine->getTensorShape("output0");
    nvinfer1::Dims4 inputDims = { idims.d[0], idims.d[1], idims.d[2], idims.d[3] };
    nvinfer1::Dims3 outputDims = { idims.d[0], idims.d[1], idims.d[2] };
    context->setInputShape("images", inputDims);

    void* buffers[2];
    const int inputIndex = 0;
	const int outputIndex = 1;

    cudaMalloc(&buffers[inputIndex], idims.d[0] * idims.d[1] * idims.d[2] * idims.d[3] * sizeof(float));
	cudaMalloc(&buffers[outputIndex], odims.d[0] * odims.d[1] * odims.d[2] * sizeof(float));

    // 设定数据地址
    context->setTensorAddress("images", buffers[inputIndex]);
    context->setTensorAddress("output0", buffers[outputIndex]);

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 读取文件执行推理
    for(int i = 0; i < 2; i += 1)
    {
        // 读取图片
        cv::Mat img;
        std::string file_name = "img/img" + std::to_string(i) + ".jpg";
        img = cv::imread(file_name);
        if (img.empty())  //检测image有无数据，无数据 image.empty()返回 真
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }

        cv::Mat LetterBoxImg;
	    cv::Vec4d params;
	    LetterBox(img, LetterBoxImg, params, cv::Size(640, 640));

        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false, CV_32F);

        // 将图像拷贝到GPU
        cudaMemcpyAsync(buffers[inputIndex], blob.data,3 * 640 * 640 * sizeof(float), cudaMemcpyHostToDevice, stream);

        //执行推理
        if(context->enqueueV3(stream)){
            std::cout << "enqueued successfully!" << std::endl;
        }
        cudaStreamSynchronize(stream);

        float rst[1][84][8400];
        cudaMemcpyAsync(&rst, buffers[outputIndex], 1 * 84 * 8400 * sizeof(float), cudaMemcpyDeviceToHost, stream);

        postprocess(rst, img, params);
    }

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}