#include "SelectROIs.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

static int pts[4];
static std::string winname = "Select ROIs";

static void drawCross(Mat &img, int x, int y) {
    int m = img.size().height;
    int n = img.size().width;
    line(img, Point(0, y), Point(n - 1, y), Scalar(255, 255, 255), 1);
    line(img, Point(x, 0), Point(x, m - 1), Scalar(255, 255, 255), 1);
}

static void onMouse(int event, int x, int y, int flag, void *userdata) {
    cv::Mat &src = *(cv::Mat *) userdata;
    cv::Mat drawing = src.clone();
    switch (event) {
        case cv::EVENT_MOUSEMOVE:       // move
            break;

        case cv::EVENT_LBUTTONDOWN:     // left -> select
            if (!pts[0]) {
                pts[0] = x, pts[1] = y;
            } else {
                pts[2] = x, pts[3] = y;
            }
            break;

        case cv::EVENT_RBUTTONDOWN:     // right -> delete
            if (pts[3]) {
                pts[2] = 0, pts[3] = 0;
            } else {
                pts[0] = 0, pts[1] = 0;
            }
            drawing = src.clone();
            break;

        default:
            break;
    }

    if (pts[0] && pts[3]) {
        rectangle(drawing, Point(pts[0], pts[1]), Point(pts[2], pts[3]), Scalar(20, 200, 255));
    } else if (pts[0]) {
        circle(drawing, Point(pts[0], pts[1]), 3, Scalar(20, 200, 255), -1);
    } else;
    drawCross(drawing, x, y);
    cv::imshow(winname, drawing);
    cv::waitKey(1);
}


void selectROIs(cv::Mat &img, std::vector<cv::Rect> &bboxes) {
    bboxes.clear();
    Mat drawing = img.clone();
    while (true) {
        pts[0] = 0, pts[1] = 0, pts[2] = 0, pts[3] = 0;
        imshow(winname, drawing);
        setMouseCallback(winname, onMouse, &drawing);
        int key = cv::waitKey(0);
        if (key == 27) {
            break;
        } else if (!pts[3]) continue;
        else {
            bboxes.push_back(Rect(Point(pts[0], pts[1]), Point(pts[2], pts[3])));
            rectangle(drawing, bboxes.back(), Scalar(20, 180, 20), 2);
        }
    }
    destroyAllWindows();
}
