
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>

#ifndef unix
#define ssize_t int
#endif

void reverse(cv::Mat& binary);
void color_significance(cv::Mat& hsv, cv::Mat& grayscale, double orient);

double distance(cv::Point2d p1, cv::Point2d p2);

int imax(int n, double* arr);
double amax(int n, double* arr);
int imin(int n, double* arr);
double amin(int n, double* arr);

int boundary(cv::Mat& grayscale, cv::Point2d origin, cv::Point2d step, int maximal, double tolerance);
uchar bilinear(uchar p1, uchar p2, uchar p3, uchar p4, double x, double y);
uchar get_bilinear(cv::Mat& grayscale, double x, double y);
void infect(cv::Mat& grayscale, cv::Mat& out, cv::Point init, double cutoff);

std::vector< std::pair< uchar, int >> extract_line(
    cv::Mat& grayscale, cv::Point start, cv::Point end
);

void extract_flank(
    cv::Mat& grayscale, cv::Mat& out, cv::Point2d origin,
    cv::Point2d orient, cv::Point2d up,
    int flank, int extend
);

void show(cv::Mat& matrix, const char* window, int width = 800, int height = 600);
void hist(cv::Mat& grayscale, cv::Mat mask);
int quartile(cv::Mat& grayscale, cv::Mat mask, double lower);
int any(cv::Mat& binary);
int any_right(cv::Mat& binary, int col);
