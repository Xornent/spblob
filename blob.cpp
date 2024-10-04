
#define filter_color

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "spblob.h"

int main(int argc, char* argv[]) {

    if (argc == 1) {
        printf("usage: spblob [PATH]\n");
        return 1;
    }

    // read the specified image in different color spaces.
    cv::Mat grayscale = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat colored = cv::imread(argv[1], cv::IMREAD_COLOR);
    
    double zoom_first_round = 1;
    anchors_t anch;
    anchor(grayscale, anch, zoom_first_round);
    filter_mean_color(colored, anch);

    return 0;
}

void anchor(cv::Mat &image, anchors_t &anchors, double prepzoom) {
    
    cv::Mat smaller;
    cv::resize(image, smaller, cv::Size(0, 0), prepzoom, prepzoom);

    cv::Mat blurred;
    cv::GaussianBlur(smaller, blurred, cv::Size(5, 5), 0);
    cv::imshow("blurred", blurred);
    cv::waitKey(0);

    cv::Mat blur_usm, usm;
    cv::GaussianBlur(blurred, blur_usm, cv::Size(0, 0), 25);
    cv::addWeighted(blurred, 1.5, blur_usm, -0.5, 0, usm);
    cv::imshow("sharpened", usm);
    cv::waitKey(0);

    // find contours of reference triangles on a threshold.
    // the thresholding step may also be adaptive.

    // cv::Mat thresh;
    // cv::adaptiveThreshold(
    //     usm, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C,
    //     cv::THRESH_BINARY, 55, 25
    // );
    // cv::imshow("adaptive thresholded (current)", thresh);
    // cv::waitKey(0);

    // erosion and dilation
    // cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);
    // cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    // cv::imshow("morphology", thresh);
    // cv::waitKey(0);

    cv::Mat edges;
    cv::Canny(usm, edges, 55, 35);
    cv::imshow("edges", edges);
    cv::waitKey(0);

    cv::Mat morph1, morph;

    int full[3][3] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    cv::Mat kernel_full(3, 3, CV_8U, full);
    cv::morphologyEx(edges, morph, cv::MORPH_DILATE, kernel_full, cv::Point(-1, -1), 1);
    cv::imshow("morphology", morph);
    cv::waitKey(0);

    std::vector< std::vector< cv::Point> > contours;
 	cv::findContours(morph, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector< std::vector< cv::Point> > vertices(contours.size());
    std::vector< int > filter_indices;
    
    for (int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(
            contours[i], vertices[i],
            0.1 * cv::arcLength(contours[i], true), true
        );

        std::size_t lvert = vertices[i].size();
        if (lvert == 3) {

            // here, we should apply simple color, shape and size threshold
            // for selecting valid red triangle as reference.

            double area = cv::contourArea(vertices[i], false);
            double length = cv::arcLength(vertices[i], true);
            double ratio = length * length / area;
            
            if (ratio < 22 || ratio > 25) continue;
            if (area < 10) continue;

            filter_indices.push_back(i);
            cv::drawContours(
                smaller, contours, i,
                cv::Scalar(255, 0, 0, 0), 2, 8
            );
        }
    }

    int* array = (int*)malloc(sizeof(int) * 6 * filter_indices.size());
    for (int j = 0; j < filter_indices.size(); j++) {
        array[j * 6 + 0] = (int)(vertices[filter_indices[j]][0].x / prepzoom);
        array[j * 6 + 1] = (int)(vertices[filter_indices[j]][0].y / prepzoom);
        array[j * 6 + 2] = (int)(vertices[filter_indices[j]][1].x / prepzoom);
        array[j * 6 + 3] = (int)(vertices[filter_indices[j]][1].y / prepzoom);
        array[j * 6 + 4] = (int)(vertices[filter_indices[j]][2].x / prepzoom);
        array[j * 6 + 5] = (int)(vertices[filter_indices[j]][2].y / prepzoom);
    }

    cv::imshow("annotated", smaller);
    cv::waitKey(0);

    anchors.vertices = array;
    anchors.detections = filter_indices.size();
    anchors.zoom = prepzoom;
}

void filter_mean_color(cv::Mat &colored, anchors_t &anchors) {
    
    cv::Mat smaller, hsv;
    double zoom = anchors.zoom;
    cv::resize(colored, smaller, cv::Size(0, 0), zoom, zoom);
    cv::cvtColor(smaller, hsv, cv::COLOR_BGR2HSV);
    std::vector<std::vector<cv::Point>> contours;

    for (int i = 0; i < anchors.detections; i++) {
        cv::Point p1(anchors.vertices[6 * i + 0] * zoom, anchors.vertices[6 * i + 1] * zoom);
        cv::Point p2(anchors.vertices[6 * i + 2] * zoom, anchors.vertices[6 * i + 3] * zoom);
        cv::Point p3(anchors.vertices[6 * i + 4] * zoom, anchors.vertices[6 * i + 5] * zoom);
        std::vector <cv::Point> cont;
        cont.push_back(p1);
        cont.push_back(p2);
        cont.push_back(p3);
        contours.push_back(cont);
    }

    std::vector< int > filter_indices;
    for (int i = 0; i < contours.size(); i++) {
        cv::Mat contour_mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
        cv::drawContours(contour_mask, contours, i, cv::Scalar(255), -1);
        cv::Scalar contour_mean = cv::mean(hsv, contour_mask);

        double h = contour_mean[0];
        if (h < 90) h += 180;
        double s = contour_mean[1];
        double v = contour_mean[2];

#ifdef filter_color
        if ((h > 140 || h < 220) && s > 80 && v > 30) {
#else
        if (true) {
#endif
            filter_indices.push_back(i);
        }
    }

    // by now, generate the valid reference red triangles.
    int* array = (int*)malloc(sizeof(int) * 6 * filter_indices.size());
    double total_length = 0;
    for (int j = 0; j < filter_indices.size(); j++) {
        array[j * 6 + 0] = (int)(contours[filter_indices[j]][0].x / zoom);
        array[j * 6 + 1] = (int)(contours[filter_indices[j]][0].y / zoom);
        array[j * 6 + 2] = (int)(contours[filter_indices[j]][1].x / zoom);
        array[j * 6 + 3] = (int)(contours[filter_indices[j]][1].y / zoom);
        array[j * 6 + 4] = (int)(contours[filter_indices[j]][2].x / zoom);
        array[j * 6 + 5] = (int)(contours[filter_indices[j]][2].y / zoom);

        cv::drawContours(
            smaller, contours, filter_indices[j],
            cv::Scalar(255, 0, 0, 0), 2, 8
        );
        total_length += cv::arcLength(contours[filter_indices[j]], true);
    }

    total_length /= filter_indices.size();
    free(anchors.vertices);
    anchors.vertices = array;
    anchors.zoom = (34.14 / total_length) * zoom;
    cv::imshow("annotated colored", smaller);
    cv::waitKey(0);

    printf("designated zoom: %4f\n", anchors.zoom);
}