#include "opencv2/imgproc/imgproc.hpp"

#include "multidroplet_tracking.h"
#include "camera/camera.h"
#include "preprocessing/preprocessing.h"
#include "detection/detect.h"
#include "tracking/tracking.h"

using namespace std;
using namespace cv;

// globals
cv::Mat frame;
cv::Rect bbox;
vector<Rect> bboxes;
int num;
int labels[5];  // center upper lower left right
std::string winname{ "Droplet Tracking" };

// functions
void openCamera(int cameraIndex, double width, double height, double fps) {
    camera::open(cameraIndex, width, height, fps);
    recorder::open();
}

void openVideo(char* path) {
    camera::openVideo(path);
    recorder::open();
}

void setCameraParams(double brightness, double contrast, double saturation, double hue, double gain, double exposure) {
    camera::setParams(brightness, contrast, saturation, hue, gain, exposure);
}

void imshow() {
    for (int i = 0; i < num; i++) {
        cv::rectangle(frame, bboxes[i], cv::Scalar(50, 180, 20), 3, 8, 0);
    }
    cv::imshow(winname, frame);
    cv::waitKey(1);
}

void initTrackers() {
    camera::read(frame);
    preprocessing(frame);
    num = manuDetect(frame, bboxes);
    //bool ret = circleDetect(frame, 25, 50, bbox);
    if(num<=0) exit(-1);
    multitracker::init(frame, bboxes);
    imshow();
}

int* updateTrackers() {
    if (! camera::read(frame)) return nullptr;
    perspective(frame);
    multitracker::update(frame, bboxes);
    getLabels(bbox, labels);
    imshow();
    return labels;
}

void closeTrackers(){
    cv::destroyAllWindows();
    multitracker::close();
    recorder::close();
    camera::close();
}