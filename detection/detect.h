#pragma once

#include <iostream>
#include "opencv2/core/core.hpp"

bool manuDetect(cv::Mat &, std::vector<cv::Rect>& bboxes);

bool colorDetect(cv::Mat &, cv::Scalar, cv::Scalar, double, double, cv::Rect &);

bool circleDetect(cv::Mat &img, double min_r, double max_r, cv::Rect &bbox);


