
#include <opencv2/opencv.hpp>

struct arguments {
    int save_count;
    char log_file_path[1024];
    char stat_file_path[1024];
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
void reverse(cv::Mat &binary);
void color_significance(cv::Mat &hsv, cv::Mat &grayscale, double orient);

std::vector< std::pair< uchar, int >> extract_line(
    cv::Mat &grayscale, cv::Point start, cv::Point end
);

double distance(cv::Point2d p1, cv::Point2d p2);

int imax(int n, double* arr);
int amax(int n, double* arr);
int imin(int n, double* arr);
int amin(int n, double* arr);

int boundary(cv::Mat &grayscale, cv::Point2d origin, cv::Point2d step, int maximal, double tolerance);
uchar bilinear(uchar p1, uchar p2, uchar p3, uchar p4, double x, double y);
uchar get_bilinear(cv::Mat &grayscale, double x, double y);
void infect(cv::Mat& grayscale, cv::Mat& out, cv::Point init, double cutoff);

void extract_flank(
    cv::Mat &grayscale, cv::Mat &out, cv::Point2d origin,
    cv::Point2d orient, cv::Point2d up,
    int flank, int extend
);

void plot(int n, double vec[], const char* title);
void show(cv::Mat &matrix, const char* window, int width = 800, int height = 600);
void hist(cv::Mat &grayscale, cv::Mat mask);
int quartile(cv::Mat &grayscale, cv::Mat mask, double lower);
int any(cv::Mat &binary);
int any_right(cv::Mat &binary, int col);
