#pragma once
#include "yolov8_utils.h"

using namespace cv;
using namespace std;

void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void postprocess(float (&rst)[1][84][8400], cv::Mat &img, cv::Vec4d params)
{	
	std::vector<cv::Rect> boxes;
	std::vector<float> scores;
	std::vector<int> det_rst;
	static const float score_threshold = 0.6;
    static const float nms_threshold = 0.45;
    std::vector<int> indices;

	for(int Anchors=0 ;Anchors < 8400; Anchors++)
	{
		float max_score = 0.0;
		int max_score_det = 99;
		float pdata[4];
		for(int prob = 4; prob < 84; prob++)
		{
			if(rst[0][prob][Anchors] > max_score){
				max_score = rst[0][prob][Anchors];
				max_score_det = prob - 4;
				pdata[0] = rst[0][0][Anchors];
				pdata[1] = rst[0][1][Anchors];
				pdata[2] = rst[0][2][Anchors];
				pdata[3] = rst[0][3][Anchors];
			}
		}
		if(max_score >= score_threshold)
		{
			float x = (pdata[0] - params[2]) / params[0];  
			float y = (pdata[1] - params[3]) / params[1];  
			float w = pdata[2] / params[0];  
			float h = pdata[3] / params[1];  
			int left = MAX(int(x - 0.5 * w + 0.5), 0);
			int top = MAX(int(y - 0.5 * h + 0.5), 0);
			boxes.push_back(Rect(left, top, int(w + 0.5), int(h + 0.5)));
			scores.emplace_back(max_score);
			det_rst.emplace_back(max_score_det);
		}
	}

	cv::dnn::NMSBoxes(boxes, scores, score_threshold, nms_threshold, indices);

	for (int i = 0; i < indices.size(); i++) {
        std::cout << boxes[indices[i]] << std::endl;
		cv::rectangle(img, boxes[indices[i]], Scalar(255, 0, 0), 2, LINE_8,0);
    }

	cv::imshow("rst",img);
	cv::waitKey(0);
}