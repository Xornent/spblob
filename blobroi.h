
#include "blob.h"

struct arguments {
    int save_count;
    char data_output_path[1024];
    double scale_factor;
    double pair_distance;
    double scale_width;
    double proximal;
    double distal;
    char input[1024];
    bool directory;
    bool fname_as_sample;
};

typedef struct anchors {
    int detections;
    int* vertices;
    double zoom;
} anchors_t;

double process(char *file, char* purefname, bool show_msg, struct arguments* args);
void anchor(cv::Mat &image, anchors_t &anchors, double prepzoom);
void filter_mean_color(cv::Mat &colored, anchors_t &anchors);
