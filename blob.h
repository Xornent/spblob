
#include <opencv2/opencv.hpp>

struct arguments {
    int save_count;
    char log_file_path[1024];
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

void elongate_forward_oblique(cv::Mat &binary);
void elongate_backward_oblique(cv::Mat &binary);
void elongate_vertical(cv::Mat &binary);
void elongate_horizontal(cv::Mat &binary);
void meet_vertical(cv::Mat &binary);
void meet_horizontal(cv::Mat &binary);
void reverse(cv::Mat &binary);
void color_significance(cv::Mat &hsv, cv::Mat &grayscale, double orient);

std::vector< std::pair< uchar, int >> extract_line(
    cv::Mat &grayscale, cv::Point start, cv::Point end
);

double distance(cv::Point2d p1, cv::Point2d p2);

double levene(int n1, int n2, double* arr1, double* arr2);
double normal(int n, double* arr, double x);
double t(int n1, int n2, double* arr1, double* arr2);

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

inline double au(
    double foremean, int foresize,
    double backstct, double backlse,
    int dark, int light, double refsize
);

// do not directly use distrib.h.  it is the internal header for the statistics
// library, the useful and clean export for libdistrib is written here.

extern "C" {

    typedef enum { FALSE = 0, TRUE } bool_t;
    double df(double x, double m, double n, int give_log);
    double dgamma(double x, double shape, double scale, int give_log);
    double dpois(double x, double lambda, int give_log);
    double dbinom(double x, double n, double p, int give_log);
    double dchisq(double x, double df, int give_log);
    double dnorm4(double x, double mu, double sigma, int give_log);
    double dt(double x, double n, int give_log);

    double pf(double x, double df1, double df2, int lower_tail, int log_p);
    double pchisq(double x, double df, int lower_tail, int log_p);
    double pgamma(double x, double alph, double scale, int lower_tail, int log_p);
    double pbinom(double x, double n, double p, int lower_tail, int log_p);
    double pbeta(double x, double a, double b, int lower_tail, int log_p);
    double pnorm5(double x, double mu, double sigma, int lower_tail, int log_p);
    double pt(double x, double n, int lower_tail, int log_p);
    
    double fmax2(double x, double y);
    double sinpi(double x);
    double tanpi(double x);
    double cospi(double x);
    double lgammafn(double x);
    double gammafn(double x);

    #define dnorm dnorm4
    #define pnorm pnorm5

}