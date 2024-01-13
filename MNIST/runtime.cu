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

int softmax(const float(&rst)[10]){
    float cache = 0;
    int idx = 0;
    for(int i = 0; i < 10; i += 1)
    {
        if(rst[i]>cache)
        {
            cache = rst[i];
            idx = i;
        };
    };
    return idx;
}

int main(int argc, char **argv)
{
    // 实例化ILogger
    Logger logger;

    // 创建runtime
    auto runtime = unique_ptr<IRuntime>(createInferRuntime(logger));

    // 读取engine,反序列化
    string file_path = "MNIST.engine";
    auto plan = load_engine_file(file_path);
    auto engine = shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(plan.data(), plan.size()));

    // 创建执行上下文
    auto context = unique_ptr<IExecutionContext>(engine->createExecutionContext());

    auto idims = engine->getTensorShape("input.1");// 这里的名字可以在导出时修改
    auto odims = engine->getTensorShape("23");
    Dims4 inputDims = { 1, idims.d[1], idims.d[2], idims.d[3] };
    Dims2 outputDims = { 1, 10 };
    context->setInputShape("input.1", inputDims);

    void* buffers[2];
    const int inputIndex = 0;
	const int outputIndex = 1;

    cudaMalloc(&buffers[inputIndex], 1 * 28 * 28 * sizeof(float));
	cudaMalloc(&buffers[outputIndex], 10 * sizeof(float));

    // 设定数据地址
    context->setTensorAddress("input.1", buffers[inputIndex]);
    context->setTensorAddress("23", buffers[outputIndex]);

    // 创建cuda流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 读取文件执行推理
    for(int i = 0; i < 10; i += 1)
    {
        // 读取图片
        cv::Mat img0;
        std::string file_name = "img/img" + std::to_string(i) + ".png";
        img0 = cv::imread(file_name, 0);// 0为灰度图片
        if (img0.empty())  //检测image有无数据，无数据 image.empty()返回 真
        {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        cv::Mat img;
        img0.convertTo(img, CV_32F);
        // cv::imshow(file_name,img);
        // cv::waitKey(0);

        // 将图像拷贝到GPU
        cudaMemcpyAsync(buffers[inputIndex], img.data,1 * 28 * 28 * sizeof(float), cudaMemcpyHostToDevice, stream);

        //执行推理
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        float rst[10];
        cudaMemcpyAsync(&rst, buffers[outputIndex], 1 * 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cout << file_name << " 推理结果: " << softmax(rst) <<endl;
    }

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
}

//https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-861/developer-guide/index.html#perform-inference
//https://blog.csdn.net/weixin_38241876/article/details/133177813
//https://www.dotndash.net/2023/03/09/using-tensorrt-with-opencv-cuda.html