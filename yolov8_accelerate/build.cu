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

#include "yolov8_utils.h"
#include "calibrator.h"

using namespace std;
using namespace nvinfer1;

// 以下示例捕获所有警告消息，但忽略信息性消息
class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // 抑制信息级别的消息
        if (severity <= Severity::kWARNING)
            cout << msg << endl;
    }
};

int main(int argc, char **argv)
{
    // 实例化ILogger
    Logger logger;

    // 创建builder
    auto builder = unique_ptr<IBuilder>(createInferBuilder(logger));

    // 创建网络(显性batch)
    uint32_t flag = 1U <<static_cast<uint32_t>
    (NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(flag));

    // 创建ONNX解析器：parser
    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    // 读取文件
    char *file_path = "yolov8x.onnx";
    parser->parseFromFile(file_path, static_cast<int32_t>(ILogger::Severity::kWARNING));

    // 创建构建配置，用来指定trt如何优化模型
    auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    // 设定配置
    // 工作空间大小
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
    // 设置精度
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto* calibrator = new Int8Calibrator(1, 640, 640, "./coco_calib/", "int8calib.table", "");
    config->setInt8Calibrator(calibrator);

    // 创建引擎
    auto engine = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    //序列化保存engine
    ofstream engine_file("./yolov8x.engine", ios::binary);
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char *)engine->data(), engine->size());
    engine_file.close();

    cout << "Engine build success!" << endl;
    return 0;
}