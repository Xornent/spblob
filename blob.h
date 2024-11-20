
//    Copyright (C) 2024 Zheng Yang <xornent@outlook.com>
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#define _ARGPARSE_NO_PRINT_ARGUMENT_PROPS

#include <iostream>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>

#ifdef unix
#define soft_br ""
#else
#define ssize_t int
#define soft_br "\n"
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
