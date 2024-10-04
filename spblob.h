
#include <opencv2/opencv.hpp>

typedef struct anchors {
    int detections;
    int* vertices;
    double zoom;
} anchors_t;

void anchor(cv::Mat &image, anchors_t &anchors, double prepzoom);
void filter_mean_color(cv::Mat &colored, anchors_t &anchors);