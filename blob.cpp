
// conditional compilation switches =========================================== 

// enable color filtering: the anchor triangles will be filtered to retain
//     those with perceptible red color.

#define filter_color

// anchor detection modes:
//
// extendmorph: find the edges of the grayscale (or color channel) images,
//     trying to connect slightly incontinuous segments, and widen the contour
//     outline by a single dilation morphology operation.
//
// lineseg: simplify the canny edges with a predictive line segments, and draw
//     these segments with a greater length to a new bitmap as the edges.
//
// threshold: direct thresholding (static). while a normalization step is
//     required to tolerate more variable images, current tests suggests that
//     this static threshold is enough for now. if you use raw grayscale, the
//     thresholding step is rather dissatisfying. however, if you use the
//     extracted red-ness degrees (i figured it out through a simple hsv transformation)
//     the red parts is extensively highlighted and the threshold make sense,
//     and this color transform overperform all attempts before.

#undef anchordet_extendmorph
#undef anchordet_lineseg
#define anchordet_threshold

// operation mode:
//
// splitimage: split the image into several roisize * roisize blocks. and
//     process each block seperately before mapping the results to the original
//     image. (a batch work, and may use parallel.). it is not recommended now,
//     this takes more time than processing as a whole even it's parallel, and
//     serves only to gain greater performance (with less distraction information)
//     when using extendmorph detection algorithm.
//
// wholeimage: process as a whole.
//
// debug switch: this will allow you to specify the region of interest when in
//     split-image mode, and will toggle on the verbose field, so you will see
//     images of each step showing up. you can specify the region of interest
//     with debug_w and debug_h.

#undef debug

#undef mode_splitimage
#define debug_w 1050
#define debug_h 825
#define roisize 150

#define mode_wholeimage

// ============================================================================

#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;
#include <chrono>
namespace chrono = std::chrono;

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/plot.hpp>

#include "blob.h"

int main(int argc, char* argv[]) {

    if (argc == 1) {
        printf("spblob: blob semen patches from images.\n");
        printf("usage: spblob [-d] [PATH]\n\n");
        printf("positional arguments:\n");
        printf("PATH   the path to the image file, or the directory containing\n");
        printf("       image files when supplying -d option\n\n");
        printf("options:\n");
        printf("-d     toogle directory process mode\n");
        return 1;
    }

    if (strcmp(argv[1], "-d") == 0) {

        char* dir = argv[2];
        std::string path(dir);
        for (const auto & entry : fs::directory_iterator(path)) {
            if (!entry.is_directory()) {
                printf("processing %s ... ", (char*)(entry.path().filename().c_str()));
                double dur = process((char*)(entry.path().c_str()));
                printf(" %.4f s\n", dur);
            }
        }

    } else {
        printf("processing %s ... ", argv[1]);
        double dur = process(argv[1]);
        printf(" %.4f s\n", dur);
    }

    return 0;
}

double process(char* file) {
    
    // read the specified image in different color spaces.

    cv::Mat grayscale = cv::imread(file, cv::IMREAD_GRAYSCALE);
    cv::Mat colored = cv::imread(file, cv::IMREAD_COLOR);
    
    cv::Mat colored_hsv;
    cv::cvtColor(colored, colored_hsv, cv::COLOR_BGR2HSV);
    
    cv::Mat component_red;
    grayscale.copyTo(component_red);
    color_significance(colored_hsv, component_red, 0.0);
    show(component_red, "red", 800, 600);

    cv::Mat annot;
    colored.copyTo(annot);

    double zoom_first_round = 1;

    auto start = chrono::system_clock::now();

#if defined(mode_splitimage)

    int totalh = 2 * (grayscale.size().height - roisize) / roisize;
    int totalw = 2 * (grayscale.size().width - roisize) / roisize;

#ifdef debug
#define verbose

    cv::Rect roi(debug_w, debug_h, roisize, roisize);
    cv::Mat grayscale_roi(component_red, roi);
    cv::Mat colored_roi(colored, roi);
    anchors_t anch;
    anchor(grayscale_roi, anch, zoom_first_round);
    filter_mean_color(colored_roi, anch);

#else

    for (int h = 0; h < grayscale.size().height - roisize; h += roisize / 2) {
        for (int w = 0; w < grayscale.size().width - roisize; w += roisize / 2) {

            printf(". processing (%d, %d) out of (%d, %d)\n", h, w, totalh, totalw);
            cv::Rect roi(w, h, roisize, roisize);
            cv::Mat grayscale_roi(component_red, roi);
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

#endif

#elif defined(mode_wholeimage)

#ifdef debug
#define verbose
#endif

    anchors_t anch;
    anchor(component_red, anch, zoom_first_round);
    filter_mean_color(colored, anch);

    if (anch.detections > 0) {
        std::vector< std::vector <cv::Point> > contours;
        for (int i = 0; i < anch.detections; i++) {
            cv::Point p1(anch.vertices[6 * i + 0], anch.vertices[6 * i + 1]);
            cv::Point p2(anch.vertices[6 * i + 2], anch.vertices[6 * i + 3]);
            cv::Point p3(anch.vertices[6 * i + 4], anch.vertices[6 * i + 5]);
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

#endif

    // here, we will scale the image to a relatively uniform size. and infer
    // the relative center for each detection.

    cv::Mat scaled_gray, scaled_color;
    cv::resize(grayscale, scaled_gray, cv::Size(0, 0), anch.zoom, anch.zoom);
    cv::resize(colored, scaled_color, cv::Size(0, 0), anch.zoom, anch.zoom);
    double zoom = anch.zoom;

    std::vector< std::pair< int, cv::Point2d >> meeting_points;
    std::vector< std::pair< int, int >> paired;
    std::vector< cv::Point2d > base_vertice;
    std::vector< cv::Point2d > base_meeting;

    for (int i = 0; i < anch.detections; i++) {
        cv::Point2d p1(anch.vertices[6 * i + 0] * zoom, anch.vertices[6 * i + 1] * zoom);
        cv::Point2d p2(anch.vertices[6 * i + 2] * zoom, anch.vertices[6 * i + 3] * zoom);
        cv::Point2d p3(anch.vertices[6 * i + 4] * zoom, anch.vertices[6 * i + 5] * zoom);
        
        cv::Point2d vert, hei;

        if (distance(p1, p2) < distance(p2, p3) && distance(p1, p3) < distance(p2, p3)) {
            vert = p1;
            hei = cv::Point2d((p2.x + p3.x) / 2, (p2.y + p3.y) / 2);
        }

        if (distance(p2, p1) < distance(p1, p3) && distance(p2, p3) < distance(p1, p3)) {
            vert = p2;
            hei = cv::Point2d((p1.x + p3.x) / 2, (p1.y + p3.y) / 2);
        }

        if (distance(p3, p2) < distance(p1, p2) && distance(p3, p1) < distance(p1, p2)) {
            vert = p3;
            hei = cv::Point2d((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
        }

        std::pair<int, cv::Point2d> temp;
        temp.first = i;
        temp.second = cv::Point2d(
            vert.x + (hei.x - vert.x) * 5.02,
            vert.y + (hei.y - vert.y) * 5.02
        );
        meeting_points.push_back(temp);
        base_vertice.push_back(vert);
    }

    // approximate pairs by adjacent meeting points
    // (tolerate a range within 20px range)
    
    for (int i = 0; i < meeting_points.size(); i ++) {
        for (int j = i + 1; j < meeting_points.size(); j ++) {
            if (distance(meeting_points[i].second, meeting_points[j].second) < 20.0) {
                paired.push_back(std::pair<int, int>(
                    meeting_points[i].first, meeting_points[j].first
                ));
                base_meeting.push_back( cv::Point2d(
                    (meeting_points[i].second.x + meeting_points[j].second.x) * 0.5,
                    (meeting_points[i].second.y + meeting_points[j].second.y) * 0.5
                ));
            }
        }
    }

    // calculate the grayscale on the uncorrected base line.

    for (int i = 0; i < paired.size(); i++) {
        cv::Point2d v1 = base_vertice[paired[i].first];
        cv::Point2d v2 = base_vertice[paired[i].second];
        cv::Point2d vtop, vbottom;

        cv::Point2d origin((v1.x + v2.x) * 0.5, (v1.y + v2.y) * 0.5);
        cv::Point2d meet = base_meeting[i];

        int signx = -1;
        int signy = 1;

        if (v1.y > v2.y) { vtop = v1; vbottom = v2; }
        else { vtop = v2; vbottom = v1; }

        double dx = vtop.x - vbottom.x;
        double dy = vtop.y - vbottom.y;

        double testx, testy;
        testx = origin.x + dy * signy;
        testy = origin.y + dx * signx;
        double prod = testx * (meet.x - origin.x) + testy * (meet.y - origin.y);
        if (prod < 0) { signx = 1; signy = -1; }

        cv::Point2d end(
            origin.x + dy * 4.5 * signy,
            origin.y + dx * 4.5 * signx
        );

        double unify = dx * signx / sqrt(pow(dx, 2) + pow(dy, 2));
        double unifx = dy * signy / sqrt(pow(dx, 2) + pow(dy, 2));

        auto hist = extract_line(scaled_gray, origin, end);
        int length = hist.size();
        double* array = (double*)malloc(sizeof(double) * length);
        printf("------------------------------");
        for (int j = 0; j < length; j++) {
            array[j] = hist[j].first * 1.0;
            printf("%d\n", hist[j].first);
        }

        plot(length, array, "histogram");

        cv::line(
            annot, cv::Point2d(origin.x / zoom, origin.y / zoom),
            cv::Point2d(end.x / zoom, end.y / zoom),
            cv::Scalar(0, 0, 255, 0), 2, 8
        );

        // we will then extract the sharp turning point using a method based on
        // statistical observation on t-test.

        auto regions_raw = rmdm(length, array, 0.01, 100);
        
        for (auto reg : regions_raw) {
            cv::circle(
                annot,
                cv::Point2d(
                    (origin.x + unifx * reg.start) / zoom,
                    (origin.y + unify * reg.start) / zoom
                ), 5, cv::Scalar(0, 0, 255, 0), cv::FILLED
            );

            cv::circle(
                annot,
                cv::Point2d(
                    (origin.x + unifx * reg.end) / zoom,
                    (origin.y + unify * reg.end) / zoom
                ), 5, cv::Scalar(0, 255, 0, 0), cv::FILLED
            );
        }
    }

    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double ms = double(duration.count()) * chrono::milliseconds::period::num /
        chrono::milliseconds::period::den;

    show(annot, "annotated", 800, 600);
    return ms;
}

void anchor(cv::Mat &image, anchors_t &anchors, double prepzoom) {
    
    cv::Mat smaller;
    cv::resize(image, smaller, cv::Size(0, 0), prepzoom, prepzoom);
    
    cv::Mat blurred;
    cv::GaussianBlur(smaller, blurred, cv::Size(5, 5), 0);
    
    cv::Mat blur_usm, usm;
    cv::GaussianBlur(blurred, blur_usm, cv::Size(0, 0), 25);
    cv::addWeighted(blurred, 1.5, blur_usm, -0.5, 0, usm);
    
#ifdef verbose
    show(usm, "sharpened", 800, 600);
#endif

    cv::Mat edges;
    cv::Canny(usm, edges, 35, 55);
    cv::Mat morph = cv::Mat::zeros(edges.size(), CV_8UC1);
    for (int j = 0; j < 3; j ++) elongate_forward_oblique(edges);
    for (int j = 0; j < 3; j ++) elongate_backward_oblique(edges);

#ifdef verbose
    show(edges, "edges", 800, 600);
#endif

#if defined(anchordet_extendmorph)

    meet_horizontal(edges);
    meet_vertical(edges);

    cv::Mat kernel_full = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(edges, morph, cv::MORPH_DILATE, kernel_full, cv::Point(-1, -1), 1);
    meet_horizontal(morph);
    meet_vertical(morph);
    reverse(morph);

#ifdef verbose
    show(morph, "dilated", 800, 600);
#endif

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

#ifdef verbose
    show(morph, "lines", 800, 600);
#endif

#elif defined(anchordet_threshold)

    cv::threshold(usm, morph, 60, 255, cv::THRESH_BINARY);
    
#ifdef verbose
    show(morph, "threshold", 800, 600);
#endif

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

#ifdef verbose

            cv::drawContours(
                smaller, contours, i,
                cv::Scalar(255, 0, 0, 0), 2, 8
            );

#endif
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

#ifdef verbose
    show(smaller, "annotated", 800, 600);
#endif

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

#ifdef verbose

        cv::drawContours(
            smaller, contours, filter_indices[j],
            cv::Scalar(255, 0, 0, 0), 2, 8
        );

#endif

        total_length += cv::arcLength(contours[filter_indices[j]], true);
    }

    total_length /= filter_indices.size();
    free(anchors.vertices);

    anchors.detections = filter_indices.size();
    anchors.vertices = array;
    anchors.zoom = (34.14 / total_length) * zoom;
    
#ifdef verbose

    show(smaller, "annotated colored", 800, 600);
    printf("designated zoom: %4f\n", anchors.zoom);

#endif

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

// calculate the perceptible color intensity. a simple transformation from
// the hsv colorspace: cos(delta.H) * S * V.
// orient: the hue [0, 360] to extract color intensity, 0 as red.

void color_significance(cv::Mat &hsv, cv::Mat &grayscale, double orient) {
    
    int width = hsv.size().width;
    int height = hsv.size().height;

    for (int line = 0; line < height; line ++) {
        cv::Vec3b* rhsv = hsv.ptr<cv::Vec3b>(line);
        uchar* rgray = grayscale.ptr<uchar>(line);

        for (int col = 1; col < width; col ++) {
            double hue = 2.0 * rhsv[col][0];
            double proj = cos((hue - orient) * CV_PI / 180.0) *
                (rhsv[col][1] / 255.0) *
                (rhsv[col][2] / 255.0);
            
            if (proj < 0) proj = 0;
            rgray[col] = (uchar) lround(proj * 255);
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

std::vector< std::pair< uchar, int >> extract_line(
    cv::Mat &grayscale, cv::Point start, cv::Point end
) {

    std::vector< std::pair<uchar, int>> result;
    int row = grayscale.rows;
    int col = grayscale.cols;

    int r1 = start.y;
    int c1 = start.x;
    int r2 = end.y;
    int c2 = end.x;

    // distance between the two anchors
    float dist = round(sqrt(pow(float(r2) - float(r1), 2.0) + pow(float(c2) - float(c1), 2.0)));
    if (dist <= 0.00001f) {
        // too short distance. return the origin point.
        std::pair<uchar, int> temp;
        temp.first = grayscale.at<uchar>(r1, c1);
        temp.second = 0;
        result.push_back(temp);
        return result;
    }

    float slope_r = (float(r2) - float(r1)) / dist;
    float slope_c = (float(c2) - float(c1)) / dist;

    int k = 0;
    for (float i = 0; i <= dist; ++i) {
        int posy = int(r1) + int(round(i * slope_r));
        int posx = int(c1) + int(round(i * slope_c));

        if (posx > grayscale.cols) continue;
        if (posy > grayscale.rows) continue;
        if (posx < 0) continue;
        if (posy < 0) continue;

        std::pair<uchar, int> temp;
        temp.first = grayscale.at<uchar>(posy, posx);
        temp.second = k;
        k++;
        result.push_back(temp);
    }

    return result;
}

void plot(int n, double vec[], const char* title) {

    cv::Mat data_x(1, n, CV_64F);
    cv::Mat data_y(1, n, CV_64F);

    // fill the matrix with custom data.
    for (int i = 0; i < data_x.cols; i++) {
        data_x.at<double>(0, i) = i;
        data_y.at<double>(0, i) = vec[i];
    }

    cv::Mat plot_result;
    cv::Ptr<cv::plot::Plot2d> plot = cv::plot::Plot2d::create(data_x, data_y);
    plot -> render(plot_result);

    cv::imshow(title, plot_result);
    cv::waitKey();
    cv::destroyAllWindows();
}

double distance(cv::Point2d p1, cv::Point2d p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

std::vector<regions_t> seqseg(int n, double* arr, int boot, double cutoff, double zcutoff, int skip) {
    static std::vector<regions_t> res;
    res.clear();
    
    regions_t* current = NULL;
    for (int i = 0; i < n; i ++) {

        if (current == NULL) {
            current = (regions_t*) malloc(sizeof(regions_t));
            current -> start = i;
            current -> end = i + boot - 1 > n ? n : i + boot - 1;

            for (int j = current -> start; j <= current -> end; j ++)
                current -> values[j -  current -> start] = (arr[j]);
            i = current -> end;
            continue;
        }

        else {

            int next_start = i;
            int next_end = i + boot - 1 > n ? n : i + boot - 1;
            double* an = (double*) malloc(sizeof(double) * (next_end - next_start + 1));
            for (int j = 0; j <= next_end - next_start; j ++) an[j] = arr[next_start + j];

            double lev = levene(
                current -> end - current -> start + 1,
                next_end - next_start + 1,
                current -> values, an);

            if (lev < cutoff) {
                for (int j = next_start; j <= next_end; j ++) {
                    double z = normal(*current, arr[j]);
                    
                    if (z >= zcutoff) {
                        current -> end = j;
                        current -> values[j - current -> start] = (arr[j]);
                        i = j;

                    } else {
                        i = j + skip;
                        res.push_back(*current);
                        current = NULL;
                        break;
                    }
                }

            } else {
                for (int j = next_start; j <= next_end; j ++)
                    current -> values[j - current -> start] = (arr[j]);
                current -> end = next_end;
                i = next_end;
                continue;
            }
        }
    }

    return res;
}

#define rmdm_boot 5

int rmdm_rec(int n, double* arr, double cutoff, std::vector<int> &seps, int start, int round) {

    if (n < 2 * rmdm_boot) return -1;
    if (round <= 0) return -1;

    double* t_vals = (double*) malloc(sizeof(double) * (n - 2 * rmdm_boot + 1));
    for (int i = rmdm_boot; i < n - rmdm_boot + 1; i ++)
        t_vals[i - rmdm_boot] = fabs(t(i, n - i, arr, (arr + i)));
    double tmax = amax(n - (2 * rmdm_boot - 1), t_vals);
    int idtmax = imax(n - (2 * rmdm_boot - 1), t_vals);
    double p = 1 - pt(tmax, n - 1, TRUE, FALSE);
    
    if (p < cutoff) {
        seps.push_back(idtmax + rmdm_boot + start);
        rmdm_rec(idtmax + rmdm_boot, arr, cutoff, seps, start, round - 1);
        rmdm_rec(n - idtmax - rmdm_boot, arr + idtmax, cutoff, seps, start + idtmax + rmdm_boot, round - 1);
    }

    return idtmax + rmdm_boot + start;
}

std::vector<regions_t> rmdm(int n, double* arr, double cutoff, int round) {
    std::vector<int> bounds;
    rmdm_rec(n, arr, cutoff, bounds, 0, round);
    sort(bounds.begin(), bounds.end());
    
    std::vector<regions_t> regions;
    int prev = 0;
    for (int n : bounds) {
        regions_t* reg = (regions_t*) malloc(sizeof(regions_t));
        reg -> start = prev;
        reg -> end = n - 1;
        prev = n;
        regions.push_back(*reg);
    }

    return regions;
}

double levene(int n1, int n2, double* arr1, double* arr2) {

    double* deviance1 = (double*) malloc(sizeof(double) * n1);
    double* deviance2 = (double*) malloc(sizeof(double) * n2);
    double sum1 = 0, sum2 = 0;
    double dev1 = 0, dev2 = 0;
    double sqdev1 = 0, sqdev2 = 0; 

    for (int i = 0; i < n1; i ++) sum1 += arr1[i];
    for (int i = 0; i < n2; i ++) sum2 += arr2[i];

    for (int i = 0; i < n1; i ++) {
        deviance1[i] = fabs(arr1[i] - sum1 / n1);
        dev1 += deviance1[i];
    }

    for (int i = 0; i < n2; i ++) {
        deviance2[i] = fabs(arr2[i] - sum2 / n1);
        dev2 += deviance2[i];
    }

    dev1 /= n1;
    dev2 /= n2;

    for (int i = 0; i < n1; i ++) sqdev1 += pow(deviance1[i] - dev1, 2);
    for (int i = 0; i < n2; i ++) sqdev2 += pow(deviance2[i] - dev2, 2);

    double devm = (dev1 + dev2) * 0.5;
    double m = (n1 + n2 - 2) *
               (n1 * pow(dev1 - devm, 2) + n2 * pow(dev2 - devm, 2)) /
               (sqdev1 + sqdev2);
    
    return 1 - pf(m, 1, n1 + n2 - 2, TRUE, FALSE);
}

double t(int n1, int n2, double* arr1, double* arr2) {

    double mean1 = 0, mean2 = 0;
    double s1 = 0, s2 = 0;

    for (int i = 0; i < n1; i ++) mean1 += arr1[i];
    for (int i = 0; i < n2; i ++) mean2 += arr2[i];
    mean1 /= n1;
    mean2 /= n2;

    for (int i = 0; i < n1; i ++) s1 += pow(arr1[i] - mean1, 2);
    for (int i = 0; i < n2; i ++) s2 += pow(arr1[i] - mean2, 2);
    s1 /= (n1 - 1);
    s2 /= (n2 - 1);
    
    return (mean1 - mean2) /
           sqrt((((n1 - 1.) * s1 + (n2 - 1.) * s2) / (n1 + n2 - 2.)) *
                (1. / n1 + 1. / n2));
}

int imax(int n, double* arr) {
   int i = 0;
   double m = std::numeric_limits<double>::min();
   for (int j = 0; j < n; j ++) if (arr[j] > m) { m = arr[j]; i = j; }
   return i;
}

int amax(int n, double* arr) {
   int i = 0;
   double m = std::numeric_limits<double>::min();
   for (int j = 0; j < n; j ++) if (arr[j] > m) { m = arr[j]; i = j; }
   return m;
}

int imin(int n, double* arr) {
   int i = 0;
   double m = std::numeric_limits<double>::max();
   for (int j = 0; j < n; j ++) if (arr[j] < m) { m = arr[j]; i = j; }
   return i;
}

int amin(int n, double* arr) {
   int i = 0;
   double m = std::numeric_limits<double>::max();
   for (int j = 0; j < n; j ++) if (arr[j] < m) { m = arr[j]; i = j; }
   return m;
}

double normal(regions_t &r, double x) {
    
    double sum = 0;
    double dev = 0;
    size_t n = r.end - r.start + 1;

    for (int i = 0; i < n; i++) sum += r.values[i];
    for (int i = 0; i < n; i++) dev += pow(r.values[i] - (sum / n), 2);
    r.mean = sum / n;
    r.stdvar = sqrt(dev / (n - 1));

    return 1 - pnorm(x, r.mean, r.stdvar, TRUE, FALSE);
}