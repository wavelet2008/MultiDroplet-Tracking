#include <iostream>
#include <opencv2/opencv.hpp>
#include "multidroplet_tracking.h"

using namespace std;

int main(int argc, char* argv[]) {
    char path[] = "../videos/multi-droplets-1000ms.mp4";
    openVideo(path);

    initTrackers();

    int* labels;
    while (true) {
        labels = updateTrackers();
        if (!labels) break;
        for (int i = 0; i < 5; ++i) {
            std::cout << labels[i] << ", ";
        }
        std::cout << std::endl;
        char key = cv::waitKey(1);
        if (key == 27) break;
    }
    closeTrackers();
    return 0;
}