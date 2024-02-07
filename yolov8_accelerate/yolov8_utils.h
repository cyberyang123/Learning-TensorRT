#pragma once
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(0, 0, 0)
);

void postprocess(float (&rst)[1][84][8400], cv::Mat &img, cv::Vec4d params);