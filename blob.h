
#include <opencv2/opencv.hpp>

typedef struct anchors {
    int detections;
    int* vertices;
    double zoom;
} anchors_t;

double process(char* file);

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

typedef struct regions {
    int start;
    int end;
    std::vector<double> values;
    std::vector<double> trimmed;
    double mean;
    double variance;
    double trimmed_mean;
    double trimmed_variance;
} regions_t;

std::vector<regions_t> turning(int n, double* arr, int boot, double cutoff, int trim);

void plot(int n, double vec[], const char* title);
void show(cv::Mat &matrix, const char* window, int width = 800, int height = 600);

extern "C" {

    double df(double x, double m, double n, int give_log);
    double dgamma(double x, double shape, double scale, int give_log);
    double dpois(double x, double lambda, int give_log);
    double dbinom(double x, double n, double p, int give_log);
    double dchisq(double x, double df, int give_log);
    double dnorm4(double x, double mu, double sigma, int give_log);

    double pf(double x, double df1, double df2, int lower_tail, int log_p);
    double pchisq(double x, double df, int lower_tail, int log_p);
    double pgamma(double x, double alph, double scale, int lower_tail, int log_p);
    double pbinom(double x, double n, double p, int lower_tail, int log_p);
    double pbeta(double x, double a, double b, int lower_tail, int log_p);
    double pnorm5(double x, double mu, double sigma, int lower_tail, int log_p);
    
    double fmax2(double x, double y);
    double sinpi(double x);
    double tanpi(double x);
    double cospi(double x);
    double lgammafn(double x);
    double gammafn(double x);

    #define dnorm dnorm4
    #define pnorm pnorm5

}