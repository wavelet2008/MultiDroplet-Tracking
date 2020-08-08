#pragma once
#include "CF Tracker/KCF_Tracker.hpp"

extern int num;

namespace multitracker {
    void init(cv::Mat& frame, std::vector<cv::Rect>& bbox);
    void update(cv::Mat& frame, std::vector<cv::Rect>& bbox);
    void close();
}