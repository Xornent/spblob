
#define filter_color

#define anchordet_extendmorph
#undef anchordet_lineseg

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
    cv::Mat annot;
    colored.copyTo(annot);

    // cv::namedWindow("select roi", cv::WINDOW_NORMAL);
    // cv::resizeWindow("select roi", 800, 600);
    // cv::Rect roi = cv::selectROI(colored);
    // cv::destroyAllWindows();
    // 
    // cv::Mat grayscale_roi(grayscale, roi);
    // cv::Mat colored_roi(colored, roi);

    double zoom_first_round = 1;
    int roisize = 100;
    int totalh = 2 * (grayscale.size().height - roisize) / roisize;
    int totalw = 2 * (grayscale.size().width - roisize) / roisize;

    for (int h = 0; h < grayscale.size().height - roisize; h += roisize / 2) {
        for (int w = 0; w < grayscale.size().width - roisize; w += roisize / 2) {

            printf(". processing (%d, %d) out of (%d, %d)", h, w, totalh, totalw);
            cv::Rect roi(w, h, roisize, roisize);
            cv::Mat grayscale_roi(grayscale, roi);
            cv::Mat colored_roi(colored, roi);
            anchors_t anch;
            anchor(grayscale_roi, anch, zoom_first_round);
            filter_mean_color(colored_roi, anch);

            if (anch.detections > 0) {
                std::vector< std::vector <cv::Point> > contours;
                for (int i = 0; i < anch.detections; i++) {
                    cv::Point p1(anch.vertices[6 * i + 0] + w, anch.vertices[6 * i + 1] + h);
                    cv::Point p2(anch.vertices[6 * i + 2] + w, anch.vertices[6 * i + 3] + h);
                    cv::Point p3(anch.vertices[6 * i + 4] + w, anch.vertices[6 * i + 5] + h);
                    std::vector <cv::Point> cont;
                    cont.push_back(p1);
                    cont.push_back(p2);
                    cont.push_back(p3);
                    contours.push_back(cont);
                }

                cv::drawContours(
                    annot, contours, -1,
                    cv::Scalar(255, 0, 0, 0), 2, 8
                );
            }
        }
    }

    show(annot, "annotated");

    return 0;
}

void anchor(cv::Mat &image, anchors_t &anchors, double prepzoom) {
    
    cv::Mat smaller;
    cv::resize(image, smaller, cv::Size(0, 0), prepzoom, prepzoom);
    
    cv::Mat blurred;
    cv::GaussianBlur(smaller, blurred, cv::Size(5, 5), 0);
    
    cv::Mat blur_usm, usm;
    cv::GaussianBlur(blurred, blur_usm, cv::Size(0, 0), 25);
    cv::addWeighted(blurred, 1.5, blur_usm, -0.5, 0, usm);
    // show(usm, "sharpened");

    cv::Mat edges;
    cv::Canny(usm, edges, 35, 55);
    cv::Mat morph = cv::Mat::zeros(edges.size(), CV_8UC1);
    for (int j = 0; j < 3; j ++) elongate_forward_oblique(edges);
    for (int j = 0; j < 3; j ++) elongate_backward_oblique(edges);
    // show(edges, "edges");

#if defined(anchordet_extendmorph)

    // for (int j = 0; j < 3; j ++) elongate_forward_oblique(edges);
    // for (int j = 0; j < 3; j ++) elongate_backward_oblique(edges);
    meet_horizontal(edges);
    meet_vertical(edges);

    cv::Mat kernel_full = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(edges, morph, cv::MORPH_DILATE, kernel_full, cv::Point(-1, -1), 1);
    meet_horizontal(morph);
    meet_vertical(morph);
    // show(morph, "lines");

#elif defined(anchordet_lineseg)

    cv::Ptr<cv::LineSegmentDetector> detector = cv::createLineSegmentDetector();
    std::vector<cv::Vec4f> segments;
    detector->detect(edges, segments);

    for (const auto &line : segments) {
        cv::Point2f p1(line[0], line[1]);
        cv::Point2f p2(line[2], line[3]);
        double length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));

        // elongate the line by 20% to fill the gaps.

        float deltax = p2.x - p1.x;
        float deltay = p2.y - p1.y;
        float elongation_ratio = 0.2f;
        p1.x = p1.x - elongation_ratio * deltax;
        p1.y = p1.y - elongation_ratio * deltay;
        p2.x = p2.x + elongation_ratio * deltax;
        p2.y = p2.y + elongation_ratio * deltay;

        if (length > 10) {
            cv::line(morph, p1, p2, cv::Scalar(255), 2);
        }
    }

    // show(morph, "lines");

#endif

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

    // show(smaller, "annotated");

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
    
    anchors.detections = filter_indices.size();
    anchors.vertices = array;
    anchors.zoom = (34.14 / total_length) * zoom;
    // show(smaller, "annotated colored");

    printf("designated zoom: %4f\n", anchors.zoom);
}

// for one dimension image (grayscale or binary) only.
void elongate_forward_oblique(cv::Mat &binary) {
    
    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 1; line < height - 2; line ++) {
        uchar* upper = binary.ptr<uchar>(line - 1);
        uchar* condu = origin.ptr<uchar>(line - 1);
        uchar* cond1 = origin.ptr<uchar>(line);
        uchar* cond2 = origin.ptr<uchar>(line + 1);
        uchar* condl = binary.ptr<uchar>(line + 2);
        uchar* lower = binary.ptr<uchar>(line + 2);

        for (int col = 1; col < width - 2; col ++) {
            if (cond1[col + 1] > 0 && cond2[col] > 0 &&
                (cond1[col - 1] + condl[col + 1] + condu[col] + cond1[col + 2] == 0)) {
                upper[col + 2] = 255;
                lower[col - 1] = 255;
            }
        }
    }
}

void elongate_backward_oblique(cv::Mat &binary) {
    
    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 1; line < height - 2; line ++) {
        uchar* upper = binary.ptr<uchar>(line - 1);
        uchar* condu = origin.ptr<uchar>(line - 1);
        uchar* cond1 = origin.ptr<uchar>(line);
        uchar* cond2 = origin.ptr<uchar>(line + 1);
        uchar* condl = binary.ptr<uchar>(line + 2);
        uchar* lower = binary.ptr<uchar>(line + 2);

        for (int col = 1; col < width - 2; col ++) {
            if (cond1[col] > 0 && cond2[col + 1] > 0 &&
                (cond2[col - 1] + condu[col + 1] + condl[col] + cond1[col + 2] == 0)) {
                upper[col - 1] = 255;
                lower[col + 2] = 255;
            }
        }
    }
}

void elongate_vertical(cv::Mat &binary) {
    
    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 1; line < height - 2; line ++) {
        uchar* upper = binary.ptr<uchar>(line - 1);
        uchar* cond1 = origin.ptr<uchar>(line);
        uchar* cond2 = origin.ptr<uchar>(line + 1);
        uchar* lower = binary.ptr<uchar>(line + 2);

        for (int col = 1; col < width; col ++) {
            if (cond1[col] > 0 && cond2[col] > 0) {
                upper[col] = 255;
                lower[col] = 255;
            }
        }
    }
}

void elongate_horizontal(cv::Mat &binary) {
    
    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 0; line < height; line ++) {
        uchar* row = binary.ptr<uchar>(line);
        uchar* orig = origin.ptr<uchar>(line);

        for (int col = 1; col < width - 2; col ++) {
            if (orig[col] > 0 && orig[col + 1] > 0) {
                row[col - 1] = 255;
                row[col + 2] = 255;
            }
        }
    }
}

void meet_vertical(cv::Mat &binary) {
    
    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 1; line < height - 1; line ++) {
        uchar* meet = binary.ptr<uchar>(line);
        uchar* cond1 = origin.ptr<uchar>(line - 1);
        uchar* cond2 = origin.ptr<uchar>(line + 1);

        for (int col = 1; col < width; col ++) {
            if (cond1[col] > 0 && cond2[col] > 0) {
                meet[col] = 255;
            }
        }
    }
}

void meet_horizontal(cv::Mat &binary) {
    
    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 0; line < height; line ++) {
        uchar* row = binary.ptr<uchar>(line);
        uchar* orig = origin.ptr<uchar>(line);

        for (int col = 1; col < width - 1; col ++) {
            if (orig[col - 1] > 0 && orig[col + 1] > 0) {
                row[col] = 255;
            }
        }
    }
}

void reverse(cv::Mat &binary) {
    
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 0; line < height; line ++) {
        uchar* row = binary.ptr<uchar>(line);
        for (int col = 1; col < width; col ++) {
            row[col] = 255 - row[col];
        }
    }
}

void show(cv::Mat &matrix, const char* window, int width, int height) {
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, width, height);
    cv::imshow(window, matrix);
    cv::waitKey();
    cv::destroyAllWindows();
}