
#include <opencv2/opencv.hpp>

typedef struct anchors {
    int detections;
    int* vertices;
    double zoom;
} anchors_t;

void anchor(cv::Mat &image, anchors_t &anchors, double prepzoom);
void filter_mean_color(cv::Mat &colored, anchors_t &anchors);

void elongate_forward_oblique(cv::Mat &binary);
void elongate_backward_oblique(cv::Mat &binary);
void elongate_vertical(cv::Mat &binary);
void elongate_horizontal(cv::Mat &binary);
void meet_vertical(cv::Mat &binary);
void meet_horizontal(cv::Mat &binary);
void reverse(cv::Mat &binary);
void color_significance(cv::Mat &hsv, cv::Mat &grayscale, double orient);

void show(cv::Mat &matrix, const char* window, int width = 800, int height = 600);