#include <iostream>
#include <iterator>
#include <dirent.h>
#include <fstream>
#include <opencv2/dnn/dnn.hpp>
#include "calibrator.h"
#include "yolov8_utils.h"
#include "cuda_utils.h"

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
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

Int8Calibrator::Int8Calibrator(int batchsize, int input_w, int input_h, const char* img_dir, const char* calib_table_name,
                                               const char* input_blob_name, bool read_cache)
    : batchsize_(batchsize)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , input_blob_name_(input_blob_name)
    , read_cache_(read_cache)
{
    input_count_ = 3 * input_w * input_h * batchsize;
    CUDA_CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    read_files_in_dir(img_dir, img_files_);
}

Int8Calibrator::~Int8Calibrator()
{
    CUDA_CHECK(cudaFree(device_input_));
}

int Int8Calibrator::getBatchSize() const noexcept
{
    return batchsize_;
}

bool Int8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (img_idx_ + batchsize_ > (int)img_files_.size()) {
        return false;
    }

    std::vector<cv::Mat> input_imgs_;
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::Mat LetterBoxImg;
        cv::Vec4d params;
	    LetterBox(temp, LetterBoxImg, params, cv::Size(640, 640));
        input_imgs_.push_back(LetterBoxImg);
    }
    img_idx_ += batchsize_;
    cv::Mat blob = cv::dnn::blobFromImage(input_imgs_[0], 1 / 255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false, CV_32F);
    CUDA_CHECK(cudaMemcpy(device_input_, blob.ptr<float>(0), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    // assert(!strcmp(names[0], input_blob_name_));
    bindings[0] = device_input_;
    return true;
}

const void* Int8Calibrator::readCalibrationCache(size_t& length) noexcept
{
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}
