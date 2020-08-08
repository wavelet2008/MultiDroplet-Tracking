#include "tracking.h"
#include "CF Tracker/utils.hpp"

using namespace std;
using namespace cv;

vector<KCF_Tracker*> trackers;
extern int num;

void multitracker::init(cv::Mat& frame, vector<Rect>& bboxes) {
    num = bboxes.size();
    for (int i = 0; i < num; i++) {
        KCF_Tracker* t = new KCF_Tracker;
        t->use_colorname = true;
        t->use_gray = true;
        t->init(frame, bboxes[i]);
        trackers.push_back(t);
    }

}

void multitracker::update(cv::Mat& frame, vector<Rect>& bboxes) {
    for (int i = 0; i < num; i++) {
        trackers[i]->track(frame);
        bboxes[i] = trackers[i]->getBBox();
    }
}

void multitracker::close() {
    for (int i = 0; i < num; i++) {
        delete trackers[i];
    }
}

