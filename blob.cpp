
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

int main(int argc, char *argv[])
{

    if (argc == 1)
    {
        printf("spblob: blob semen patches from images.\n");
        printf("usage: spblob [-d] [PATH]\n\n");
        printf("positional arguments:\n");
        printf("PATH   the path to the image file, or the directory containing\n");
        printf("       image files when supplying -d option\n\n");
        printf("options:\n");
        printf("-d     toogle directory process mode\n");
        return 1;
    }

    if (strcmp(argv[1], "-d") == 0)
    {
        char *dir = argv[2];
        std::string path(dir);
        for (const auto &entry : fs::directory_iterator(path))
        {
            if (!entry.is_directory())
            {
                printf("processing %s ... ", (char *)(entry.path().filename().c_str()));
                double dur = process((char *)(entry.path().c_str()), true);
                printf(" %.4f s\n", dur);
            }
        }
    }
    else
    {
        printf("processing %s ... \n", argv[1]);
        double dur = process(argv[1], true);
        printf("< %.4f s\n", dur);
    }

    return 0;
}

double process(char *file, bool show_msg)
{

    // read the specified image in different color spaces.

    cv::Mat grayscale = cv::imread(file, cv::IMREAD_GRAYSCALE);
    cv::Mat colored = cv::imread(file, cv::IMREAD_COLOR);

    cv::Mat colored_hsv;
    cv::cvtColor(colored, colored_hsv, cv::COLOR_BGR2HSV);

    cv::Mat component_red;
    grayscale.copyTo(component_red);
    color_significance(colored_hsv, component_red, 0.0);

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

    for (int h = 0; h < grayscale.size().height - roisize; h += roisize / 2)
    {
        for (int w = 0; w < grayscale.size().width - roisize; w += roisize / 2)
        {

            printf("  [.] processing (%d, %d) out of (%d, %d)\n", h, w, totalh, totalw);
            cv::Rect roi(w, h, roisize, roisize);
            cv::Mat grayscale_roi(component_red, roi);
            cv::Mat colored_roi(colored, roi);
            anchors_t anch;
            anchor(grayscale_roi, anch, zoom_first_round);
            filter_mean_color(colored_roi, anch);

            if (anch.detections > 0)
            {
                std::vector<std::vector<cv::Point>> contours;
                for (int i = 0; i < anch.detections; i++)
                {
                    cv::Point p1(anch.vertices[6 * i + 0] + w, anch.vertices[6 * i + 1] + h);
                    cv::Point p2(anch.vertices[6 * i + 2] + w, anch.vertices[6 * i + 3] + h);
                    cv::Point p3(anch.vertices[6 * i + 4] + w, anch.vertices[6 * i + 5] + h);
                    std::vector<cv::Point> cont;
                    cont.push_back(p1);
                    cont.push_back(p2);
                    cont.push_back(p3);
                    contours.push_back(cont);
                }

                cv::drawContours(
                    annot, contours, -1,
                    cv::Scalar(255, 0, 0, 0), 2, 8);
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

    if (anch.detections > 0)
    {
        std::vector<std::vector<cv::Point>> contours;
        for (int i = 0; i < anch.detections; i++)
        {
            cv::Point p1(anch.vertices[6 * i + 0], anch.vertices[6 * i + 1]);
            cv::Point p2(anch.vertices[6 * i + 2], anch.vertices[6 * i + 3]);
            cv::Point p3(anch.vertices[6 * i + 4], anch.vertices[6 * i + 5]);
            std::vector<cv::Point> cont;
            cont.push_back(p1);
            cont.push_back(p2);
            cont.push_back(p3);
            contours.push_back(cont);
        }

        cv::drawContours(
            annot, contours, -1,
            cv::Scalar(255, 0, 0, 0), 2, 8);
    }

#endif

    // here, we will scale the image to a relatively uniform size. and infer
    // the relative center for each detection.

    cv::Mat scaled_gray, scaled_color;
    cv::resize(grayscale, scaled_gray, cv::Size(0, 0), anch.zoom, anch.zoom);
    cv::resize(colored, scaled_color, cv::Size(0, 0), anch.zoom, anch.zoom);
    double zoom = anch.zoom;

    std::vector<std::pair<int, cv::Point2d>> meeting_points;
    std::vector<std::pair<int, int>> paired;
    std::vector<cv::Point2d> base_vertice;
    std::vector<cv::Point2d> base_meeting;

    for (int i = 0; i < anch.detections; i++)
    {
        cv::Point2d p1(anch.vertices[6 * i + 0] * zoom, anch.vertices[6 * i + 1] * zoom);
        cv::Point2d p2(anch.vertices[6 * i + 2] * zoom, anch.vertices[6 * i + 3] * zoom);
        cv::Point2d p3(anch.vertices[6 * i + 4] * zoom, anch.vertices[6 * i + 5] * zoom);

        cv::Point2d vert, hei;

        if (distance(p1, p2) < distance(p2, p3) && distance(p1, p3) < distance(p2, p3))
        {
            vert = p1;
            hei = cv::Point2d((p2.x + p3.x) / 2, (p2.y + p3.y) / 2);
        }

        if (distance(p2, p1) < distance(p1, p3) && distance(p2, p3) < distance(p1, p3))
        {
            vert = p2;
            hei = cv::Point2d((p1.x + p3.x) / 2, (p1.y + p3.y) / 2);
        }

        if (distance(p3, p2) < distance(p1, p2) && distance(p3, p1) < distance(p1, p2))
        {
            vert = p3;
            hei = cv::Point2d((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);
        }

        std::pair<int, cv::Point2d> temp;
        temp.first = i;
        temp.second = cv::Point2d(
            vert.x + (hei.x - vert.x) * 5.02,
            vert.y + (hei.y - vert.y) * 5.02);
        meeting_points.push_back(temp);
        base_vertice.push_back(vert);
    }

    // approximate pairs by adjacent meeting points
    // (tolerate a range within 20px range)

    for (int i = 0; i < meeting_points.size(); i++)
    {
        for (int j = i + 1; j < meeting_points.size(); j++)
        {
            if (distance(meeting_points[i].second, meeting_points[j].second) < 40.0)
            { // FIXME. CHANGE
                paired.push_back(std::pair<int, int>(
                    meeting_points[i].first, meeting_points[j].first));
                base_meeting.push_back(cv::Point2d(
                    (meeting_points[i].second.x + meeting_points[j].second.x) * 0.5,
                    (meeting_points[i].second.y + meeting_points[j].second.y) * 0.5));
            }
        }
    }

    // calculate the grayscale on the uncorrected base line.

    std::vector<cv::Mat> rois;
    std::vector<cv::Mat> scales;
    std::vector<cv::Mat> usms;
    std::vector<bool> pass1;

    for (int i = 0; i < paired.size(); i++)
    {
        cv::Point2d v1 = base_vertice[paired[i].first];
        cv::Point2d v2 = base_vertice[paired[i].second];
        cv::Point2d vtop, vbottom;

        cv::Point2d origin((v1.x + v2.x) * 0.5, (v1.y + v2.y) * 0.5);
        cv::Point2d meet = base_meeting[i];

        int signx = -1;
        int signy = 1;

        if (v1.y > v2.y)
        {
            vtop = v1;
            vbottom = v2;
        }
        else
        {
            vtop = v2;
            vbottom = v1;
        }

        double dx = vtop.x - vbottom.x;
        double dy = vtop.y - vbottom.y;

        double testx, testy;
        testx = dy * signy;
        testy = dx * signx;
        double prod = testx * (meet.x - origin.x) + testy * (meet.y - origin.y);
        if (prod < 0)
        {
            signx = 1;
            signy = -1;
        }

        cv::Point2d end(
            origin.x + dy * 4.5 * signy,
            origin.y + dx * 4.5 * signx
        );

        double unify = dx * signx / sqrt(pow(dx, 2) + pow(dy, 2));
        double unifx = dy * signy / sqrt(pow(dx, 2) + pow(dy, 2));

        cv::line(
            annot, cv::Point2d(origin.x / zoom, origin.y / zoom),
            cv::Point2d(end.x / zoom, end.y / zoom),
            cv::Scalar(0, 0, 255, 0), 2, 8);

        double downx = -unify;
        double downy = +unifx;
        double upx = +unify;
        double upy = -unifx;

        cv::Mat scale_bar;
        extract_flank(
            scaled_gray, scale_bar, origin, cv::Point2d(unifx, unify),
            cv::Point2d(upx, upy), (distance(vtop, vbottom) / 2 - 3), 125 // FIXME: CHANGE
        );

        scales.push_back(scale_bar);

        // search for meeting boundary

        int maximal_search_length = int(100. / zoom);

        // here, the 160 and 180 is associated with the default zoom constant
        // 68.28 (in filter_color) which indicated the zoomed image is set to
        // a uniform length of 20px of the scale bar.

        cv::Point2d orig_b1((origin.x + unifx * 160.) / zoom, (origin.y + unify * 160.) / zoom); // FIXME: CHANGE
        cv::Point2d orig_b2((origin.x + unifx * 180.) / zoom, (origin.y + unify * 180.) / zoom); // FIXME: CHANGE
        int ub1 = boundary(grayscale, orig_b1, cv::Point2d(upx, upy), maximal_search_length, 0.05);
        int ub2 = boundary(grayscale, orig_b2, cv::Point2d(upx, upy), maximal_search_length, 0.05);
        int db1 = boundary(grayscale, orig_b1, cv::Point2d(downx, downy), maximal_search_length, 0.05);
        int db2 = boundary(grayscale, orig_b2, cv::Point2d(downx, downy), maximal_search_length, 0.05);

        auto ub1p = cv::Point2d((orig_b1.x + upx * ub1), (orig_b1.y + upy * ub1));
        auto db1p = cv::Point2d((orig_b1.x + downx * db1), (orig_b1.y + downy * db1));
        auto cp1 = cv::Point2d((ub1p.x + db1p.x) * 0.5 * zoom, (ub1p.y + db1p.y) * 0.5 * zoom);

        auto ub2p = cv::Point2d((orig_b2.x + upx * ub2), (orig_b2.y + upy * ub2));
        auto db2p = cv::Point2d((orig_b2.x + downx * db2), (orig_b2.y + downy * db2));
        auto cp2 = cv::Point2d((ub2p.x + db2p.x) * 0.5 * zoom, (ub2p.y + db2p.y) * 0.5 * zoom);

        cv::line(
            annot, cv::Point2d((orig_b1.x + upx * ub1), (orig_b1.y + upy * ub1)),
            cv::Point2d((orig_b1.x + downx * db1), (orig_b1.y + downy * db1)),
            cv::Scalar(0, 0, 255, 0), 3);

        cv::line(
            annot, cv::Point2d((orig_b2.x + upx * ub2), (orig_b2.y + upy * ub2)),
            cv::Point2d((orig_b2.x + downx * db2), (orig_b2.y + downy * db2)),
            cv::Scalar(0, 255, 0, 0), 3);

        // remap and construct regions of interest

        // corrected orientation.

        double corrorientx = cp2.x - cp1.x;
        double corrorienty = cp2.y - cp1.y;
        corrorientx /= distance(cp1, cp2);
        corrorienty /= distance(cp1, cp2);

        double corrupx = +corrorienty;
        double corrupy = -corrorientx;
        double corrdownx = -corrorienty;
        double corrdowny = +corrorientx;

        double width = (upx * corrupx + upy * corrupy) * ((ub1 + db1 + ub2 + db2) * zoom * 0.25);
        width -= 5; // remove the 5px boundary.
        double corratio = fabs((ub1 + db1 - ub2 - db2) / fmax(ub1 + db1, ub2 + db2));

        if (width > 1)
        {
            if (corratio < 0.1) { pass1.push_back(true); }
            else { 
                pass1.push_back(false);

                // placeholder to ensure the length of vector
                usms.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
                rois.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
                continue;
            }

            int roih = int(width);
            int roiw = 250;

            cv::Mat roi;
            extract_flank(
                scaled_gray, roi, cv::Point2d(cp1.x, cp1.y),
                cv::Point2d(corrorientx, corrorienty), cv::Point2d(corrupx, corrupy),
                roih, roiw
            );

            // second round detection, in the first round detection, we may find
            // that the small interval may introduce great error of orientation,
            // so we attempt to locate the right boundary line and perform a
            // second correction. if the line that match our criteria is not
            // found, we will just skip this process.

            cv::Mat blurred;
            cv::GaussianBlur(roi, blurred, cv::Size(5, 5), 0);

            cv::Mat blur_usm, usm;
            cv::GaussianBlur(blurred, blur_usm, cv::Size(0, 0), 25);
            cv::addWeighted(blurred, 1.5, blur_usm, -0.5, 0, usm);
            
            blur_usm.release();
            usms.push_back(usm);
            rois.push_back(roi);
        }
    }

    // process the roi

    std::vector< cv::Mat > back_strict;
    std::vector< cv::Mat > back_loose;
    std::vector< cv::Mat > foreground;
    std::vector< cv::Mat > overlap;
    std::vector< bool > has_foreground;
    int croi = 0;
    for (auto roi : rois)
    {
        if (!pass1.at(croi)) {
            back_strict.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
            back_loose.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
            foreground.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
            overlap.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
            has_foreground.push_back(false);
            croi += 1;
            continue;
        }

        croi += 1;
        cv::Mat bgstrict, bgloose, fg, ol;
        cv::Mat red(roi.size(), CV_8UC3, cv::Scalar(0, 0, 255));
        cv::Mat green(roi.size(), CV_8UC3, cv::Scalar(0, 255, 0));
        cv::Mat blue(roi.size(), CV_8UC3, cv::Scalar(255, 0, 0));

        bool detected = false;
        int maxiter = 4;
        double finethresh = 0.0781;
        double coarsethresh = 0.1875; 

        while (!detected && maxiter > 0) {

            maxiter -= 1;
            finethresh *= 0.8;
            coarsethresh *= 0.8;
            bgstrict = cv::Mat::zeros(roi.size(), CV_8U);
            bgloose = cv::Mat::zeros(roi.size(), CV_8U);
            fg = cv::Mat::zeros(roi.size(), CV_8U);

            cv::cvtColor(roi, ol, cv::COLOR_GRAY2BGR);

            if (show_msg) printf("  [.] performing infection for %d ... \n", croi);
            infect(usms.at(croi - 1), bgstrict, cv::Point(1, (roi.rows - 1) / 2 + 1), finethresh);
            infect(usms.at(croi - 1), bgloose, cv::Point(1, (roi.rows - 1) / 2 + 1), coarsethresh);

            // extract the foreground from the looser background, as an inner circle

            cv::Mat kernel_full = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::Mat morph;

            cv::morphologyEx(bgloose, morph, cv::MORPH_CLOSE, kernel_full, cv::Point(-1, -1), 1);
            reverse(morph);
            cv::morphologyEx(morph, morph, cv::MORPH_CLOSE, kernel_full, cv::Point(-1, -1), 2);

            // extract the central circle.

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(morph, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

            // match a roughly circular shape, with an estimated rational size.

            int idc = 0;
            for (auto cont : contours) {
                double lenconts = cv::arcLength(cont, true);
                double area = cv::contourArea(cont, false);
                double ratio = lenconts * lenconts / area;

                if (ratio > 3 * CV_PI && ratio < 6 * CV_PI &&
                    area > 3000 && area < 6000) {
                    cv::drawContours(ol, contours, idc, cv::Scalar(0, 0, 255), 2);
                    cv::drawContours(fg, contours, idc, cv::Scalar(255), cv::FILLED);
                    detected = true;
                } else {
                    cv::drawContours(ol, contours, idc, cv::Scalar(0, 0, 0), 1);
                }

                idc ++;
            }
        }

        // draw the visualization map.

        cv::Mat temp1, temp2, temp3;
        cv::bitwise_and(blue, blue, temp1, bgloose);
        cv::bitwise_and(green, green, temp2, bgstrict);
        cv::bitwise_and(red, red, temp3, fg);
        cv::addWeighted(temp1, 0.3, ol, 0.7, 0, ol);
        cv::addWeighted(temp2, 0.3, ol, 0.7, 0, ol);
        cv::addWeighted(temp3, 0.3, ol, 0.7, 0, ol);

        back_strict.push_back(bgstrict);
        back_loose.push_back(bgloose);
        foreground.push_back(fg);
        overlap.push_back(ol);
        has_foreground.push_back(detected);
    }

    // here, we will extract the scale mark and reads some of the critical
    // information from the scale mark image for finer adjustments.

    std::vector< uchar > scale_dark;
    std::vector< uchar > scale_light;
    std::vector< double > scale_size;
    std::vector< bool > scale_success;
    for (auto sc : scales)
    {
        cv::Mat blurred;
        cv::GaussianBlur(sc, blurred, cv::Size(5, 5), 0);

        cv::Mat blur_usm, usm;
        cv::GaussianBlur(blurred, blur_usm, cv::Size(0, 0), 25);
        cv::addWeighted(blurred, 1.5, blur_usm, -0.5, 0, usm);
        blur_usm.release();

        std::vector< cv::Vec3f > circles;
        cv::HoughCircles(usm, circles, cv::HOUGH_GRADIENT, 1, 10, 40, 50, 10, 60);
        cv::Mat darker_mask(sc.size(), CV_8U, cv::Scalar(0));
        cv::Mat lighter_mask(sc.size(), CV_8U, cv::Scalar(255));

        for (auto circ : circles) {
            cv::circle(sc, cv::Point2d(circ[0], circ[1]), circ[2], cv::Scalar(0), 2);
            
            cv::circle(
                darker_mask,

                // relatively shrink the circle to make the darker area more pure.

                cv::Point2d(circ[0], circ[1]), circ[2] - 2,
                cv::Scalar(255), cv::FILLED
            );

            cv::circle(
                lighter_mask,

                // relatively extends the circle, note that the two small red
                // triangle marks lies within lighter mask but with distinct
                // grayscale compared to the background. we may just use the 
                // median filter to ignore them.

                cv::Point2d(circ[0], circ[1]), circ[2] + 4,
                cv::Scalar(0), cv::FILLED
            );
        }

        scale_dark.push_back(quartile(sc, darker_mask, 0.40));
        scale_light.push_back(quartile(sc, lighter_mask, 0.60));
        
        if (circles.size() == 1) {
            scale_success.push_back(true);
            scale_size.push_back(CV_PI * pow(circles[0][2], 2));
        } else {
            scale_success.push_back(false);
            scale_size.push_back(1);
        }
    }

    // by now, all the detection works are done. and we will pretty-print the
    // data for further analysis by other software

    // here we have, in lengths of total scale detections:

    // std::vector<cv::Mat> rois;            raw image
    // std::vector<cv::Mat> scales;          raw scale image
    // std::vector<cv::Mat> usms;            sharpened raw image
    // std::vector<bool> pass1;              pass for clipping flank orientation

    // std::vector< uchar > scale_dark;      darker principle grayscale
    // std::vector< uchar > scale_light;     lighter principle grayscale
    // std::vector< double > scale_size;     scale circle size
    // std::vector< bool > scale_success;    no exception in the scale recognition

    // std::vector< cv::Mat > back_strict;   stricter background mask
    // std::vector< cv::Mat > back_loose;    looser background mask
    // std::vector< cv::Mat > foreground;    foreground mask
    // std::vector< bool > has_foreground;   has a foreground detection

    // note that `overlap` is not of the same length

    if (show_msg) printf("  [.] data begin \n\n");

    // table header

    printf("  [..]  orient  scale  fore  fore.mean  fore.size  back.s.mean  back.l.mean  sc.dark  sc.light  sc.size       [au] \n");
    printf("  ---- ------- ------ ----- ---------- ---------- ------------ ------------ -------- --------- -------- ---------- \n");

    for (int i = 0; i < rois.size(); i++) {
        printf("  [%2d] ", i);

        if (pass1.at(i)) printf("      x ");
        else printf("      . ");

        if (scale_success.at(i)) printf("     x ");
        else printf("     . ");

        if (has_foreground.at(i)) printf("    x ");
        else printf("    . ");

        double fm; int fsz;
        if (has_foreground.at(i)) {
            cv::Mat kernel_full = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
            cv::Mat morph;

            cv::morphologyEx(
                foreground.at(i), morph,
                cv::MORPH_DILATE, kernel_full,
                cv::Point(-1, -1), 2
            );

            auto foremean = cv::mean(rois.at(i), morph);
            auto masksum = any(morph);
            fm = foremean[0]; fsz = masksum;
            printf("    %6.2f %10d ", foremean[0], masksum);
        } else printf("         .          . ");

        auto backsmean = cv::mean(rois.at(i), back_strict.at(i));
        auto backlmean = cv::mean(rois.at(i), back_loose.at(i));
        printf("      %6.2f       %6.2f ", backsmean[0], backlmean[0]);

        printf("%8d %9d ", scale_dark.at(i), scale_light.at(i));
        if (scale_success.at(i)) {
            printf("%8.1f ", scale_size.at(i));
        } else printf("       . ");

        if (pass1.at(i) && scale_success.at(i) && has_foreground.at(i)) {
            printf("%10.2f ", au(
                fm, fsz,
                backsmean[0], backlmean[0],
                scale_dark.at(i), scale_light.at(i), scale_size.at(i))
            );
        } else printf("         . ");

        printf("\n");
    }

    if (show_msg) printf("\n  [.] data end \n");

    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double ms = double(duration.count()) * chrono::milliseconds::period::num /
                chrono::milliseconds::period::den;

    return ms;
}

void anchor(cv::Mat &image, anchors_t &anchors, double prepzoom)
{

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
    for (int j = 0; j < 3; j++)
        elongate_forward_oblique(edges);
    for (int j = 0; j < 3; j++)
        elongate_backward_oblique(edges);

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

    for (const auto &line : segments)
    {
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

        if (length > 10)
        {
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

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morph, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> vertices(contours.size());
    std::vector<int> filter_indices;

    for (int i = 0; i < contours.size(); i++)
    {
        cv::approxPolyDP(
            contours[i], vertices[i],
            0.1 * cv::arcLength(contours[i], true), true);

        std::size_t lvert = vertices[i].size();
        if (lvert == 3)
        {

            // here, we should apply simple color, shape and size threshold
            // for selecting valid red triangle as reference.

            double area = cv::contourArea(vertices[i], false);
            double length = cv::arcLength(vertices[i], true);
            double ratio = length * length / area;

            if (ratio < 22 || ratio > 25)
                continue;
            if (area < 10)
                continue;

            filter_indices.push_back(i);

#ifdef verbose

            cv::drawContours(
                smaller, contours, i,
                cv::Scalar(255, 0, 0, 0), 2, 8);

#endif
        }
    }

    int *array = (int *)malloc(sizeof(int) * 6 * filter_indices.size());
    for (int j = 0; j < filter_indices.size(); j++)
    {
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

void filter_mean_color(cv::Mat &colored, anchors_t &anchors)
{

    cv::Mat smaller, hsv;
    double zoom = anchors.zoom;
    cv::resize(colored, smaller, cv::Size(0, 0), zoom, zoom);
    cv::cvtColor(smaller, hsv, cv::COLOR_BGR2HSV);
    std::vector<std::vector<cv::Point>> contours;

    for (int i = 0; i < anchors.detections; i++)
    {
        cv::Point p1(anchors.vertices[6 * i + 0] * zoom, anchors.vertices[6 * i + 1] * zoom);
        cv::Point p2(anchors.vertices[6 * i + 2] * zoom, anchors.vertices[6 * i + 3] * zoom);
        cv::Point p3(anchors.vertices[6 * i + 4] * zoom, anchors.vertices[6 * i + 5] * zoom);
        std::vector<cv::Point> cont;
        cont.push_back(p1);
        cont.push_back(p2);
        cont.push_back(p3);
        contours.push_back(cont);
    }

    std::vector<int> filter_indices;
    for (int i = 0; i < contours.size(); i++)
    {
        cv::Mat contour_mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
        cv::drawContours(contour_mask, contours, i, cv::Scalar(255), -1);
        cv::Scalar contour_mean = cv::mean(hsv, contour_mask);

        double h = contour_mean[0];
        if (h < 90)
            h += 180;
        double s = contour_mean[1];
        double v = contour_mean[2];

#ifdef filter_color
        if ((h > 140 || h < 220) && s > 80 && v > 30)
        {
#else
        if (true)
        {
#endif
            filter_indices.push_back(i);
        }
    }

    // by now, generate the valid reference red triangles.

    int *array = (int *)malloc(sizeof(int) * 6 * filter_indices.size());
    double total_length = 0;
    for (int j = 0; j < filter_indices.size(); j++)
    {
        array[j * 6 + 0] = (int)(contours[filter_indices[j]][0].x / zoom);
        array[j * 6 + 1] = (int)(contours[filter_indices[j]][0].y / zoom);
        array[j * 6 + 2] = (int)(contours[filter_indices[j]][1].x / zoom);
        array[j * 6 + 3] = (int)(contours[filter_indices[j]][1].y / zoom);
        array[j * 6 + 4] = (int)(contours[filter_indices[j]][2].x / zoom);
        array[j * 6 + 5] = (int)(contours[filter_indices[j]][2].y / zoom);

#ifdef verbose

        cv::drawContours(
            smaller, contours, filter_indices[j],
            cv::Scalar(255, 0, 0, 0), 2, 8);

#endif

        total_length += cv::arcLength(contours[filter_indices[j]], true);
    }

    total_length /= filter_indices.size();
    free(anchors.vertices);

    anchors.detections = filter_indices.size();
    anchors.vertices = array;
    anchors.zoom = (68.28 / total_length) * zoom;

#ifdef verbose

    show(smaller, "annotated colored", 800, 600);
    printf("designated zoom: %4f\n", anchors.zoom);

#endif
}

// for one dimension image (grayscale or binary) only.
void elongate_forward_oblique(cv::Mat &binary)
{

    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 1; line < height - 2; line++)
    {
        uchar *upper = binary.ptr<uchar>(line - 1);
        uchar *condu = origin.ptr<uchar>(line - 1);
        uchar *cond1 = origin.ptr<uchar>(line);
        uchar *cond2 = origin.ptr<uchar>(line + 1);
        uchar *condl = binary.ptr<uchar>(line + 2);
        uchar *lower = binary.ptr<uchar>(line + 2);

        for (int col = 1; col < width - 2; col++)
        {
            if (cond1[col + 1] > 0 && cond2[col] > 0 &&
                (cond1[col - 1] + condl[col + 1] + condu[col] + cond1[col + 2] == 0))
            {
                upper[col + 2] = 255;
                lower[col - 1] = 255;
            }
        }
    }
}

void elongate_backward_oblique(cv::Mat &binary)
{

    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 1; line < height - 2; line++)
    {
        uchar *upper = binary.ptr<uchar>(line - 1);
        uchar *condu = origin.ptr<uchar>(line - 1);
        uchar *cond1 = origin.ptr<uchar>(line);
        uchar *cond2 = origin.ptr<uchar>(line + 1);
        uchar *condl = binary.ptr<uchar>(line + 2);
        uchar *lower = binary.ptr<uchar>(line + 2);

        for (int col = 1; col < width - 2; col++)
        {
            if (cond1[col] > 0 && cond2[col + 1] > 0 &&
                (cond2[col - 1] + condu[col + 1] + condl[col] + cond1[col + 2] == 0))
            {
                upper[col - 1] = 255;
                lower[col + 2] = 255;
            }
        }
    }
}

void elongate_vertical(cv::Mat &binary)
{

    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 1; line < height - 2; line++)
    {
        uchar *upper = binary.ptr<uchar>(line - 1);
        uchar *cond1 = origin.ptr<uchar>(line);
        uchar *cond2 = origin.ptr<uchar>(line + 1);
        uchar *lower = binary.ptr<uchar>(line + 2);

        for (int col = 1; col < width; col++)
        {
            if (cond1[col] > 0 && cond2[col] > 0)
            {
                upper[col] = 255;
                lower[col] = 255;
            }
        }
    }
}

void elongate_horizontal(cv::Mat &binary)
{

    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 0; line < height; line++)
    {
        uchar *row = binary.ptr<uchar>(line);
        uchar *orig = origin.ptr<uchar>(line);

        for (int col = 1; col < width - 2; col++)
        {
            if (orig[col] > 0 && orig[col + 1] > 0)
            {
                row[col - 1] = 255;
                row[col + 2] = 255;
            }
        }
    }
}

void meet_vertical(cv::Mat &binary)
{

    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 1; line < height - 1; line++)
    {
        uchar *meet = binary.ptr<uchar>(line);
        uchar *cond1 = origin.ptr<uchar>(line - 1);
        uchar *cond2 = origin.ptr<uchar>(line + 1);

        for (int col = 1; col < width; col++)
        {
            if (cond1[col] > 0 && cond2[col] > 0)
            {
                meet[col] = 255;
            }
        }
    }
}

void meet_horizontal(cv::Mat &binary)
{

    cv::Mat origin;
    binary.copyTo(origin);
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 0; line < height; line++)
    {
        uchar *row = binary.ptr<uchar>(line);
        uchar *orig = origin.ptr<uchar>(line);

        for (int col = 1; col < width - 1; col++)
        {
            if (orig[col - 1] > 0 && orig[col + 1] > 0)
            {
                row[col] = 255;
            }
        }
    }
}

void reverse(cv::Mat &binary)
{

    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 0; line < height; line++)
    {
        uchar *row = binary.ptr<uchar>(line);
        for (int col = 1; col < width; col++)
        {
            row[col] = 255 - row[col];
        }
    }
}

// calculate the perceptible color intensity. a simple transformation from
// the hsv colorspace: cos(delta.H) * S * V.
// orient: the hue [0, 360] to extract color intensity, 0 as red.

void color_significance(cv::Mat &hsv, cv::Mat &grayscale, double orient)
{

    int width = hsv.size().width;
    int height = hsv.size().height;

    for (int line = 0; line < height; line++)
    {
        cv::Vec3b *rhsv = hsv.ptr<cv::Vec3b>(line);
        uchar *rgray = grayscale.ptr<uchar>(line);

        for (int col = 1; col < width; col++)
        {
            double hue = 2.0 * rhsv[col][0];
            double proj = cos((hue - orient) * CV_PI / 180.0) *
                          (rhsv[col][1] / 255.0) *
                          (rhsv[col][2] / 255.0);

            if (proj < 0)
                proj = 0;
            rgray[col] = (uchar)lround(proj * 255);
        }
    }
}

void show(cv::Mat &matrix, const char *window, int width, int height)
{
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, width, height);
    cv::imshow(window, matrix);
    cv::waitKey();
    cv::destroyAllWindows();
}

std::vector<std::pair<uchar, int>> extract_line(
    cv::Mat &grayscale, cv::Point start, cv::Point end)
{

    std::vector<std::pair<uchar, int>> result;
    int row = grayscale.rows;
    int col = grayscale.cols;

    int r1 = start.y;
    int c1 = start.x;
    int r2 = end.y;
    int c2 = end.x;

    // distance between the two anchors
    float dist = round(sqrt(pow(float(r2) - float(r1), 2.0) + pow(float(c2) - float(c1), 2.0)));
    if (dist <= 0.00001f)
    {
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
    for (float i = 0; i <= dist; ++i)
    {
        int posy = int(r1) + int(round(i * slope_r));
        int posx = int(c1) + int(round(i * slope_c));

        if (posx > grayscale.cols)
            continue;
        if (posy > grayscale.rows)
            continue;
        if (posx < 0)
            continue;
        if (posy < 0)
            continue;

        std::pair<uchar, int> temp;
        temp.first = grayscale.at<uchar>(posy, posx);
        temp.second = k;
        k++;
        result.push_back(temp);
    }

    return result;
}

void plot(int n, double vec[], const char *title)
{

    cv::Mat data_x(1, n, CV_64F);
    cv::Mat data_y(1, n, CV_64F);

    // fill the matrix with custom data.
    for (int i = 0; i < data_x.cols; i++)
    {
        data_x.at<double>(0, i) = i;
        data_y.at<double>(0, i) = vec[i];
    }

    cv::Mat plot_result;
    cv::Ptr<cv::plot::Plot2d> plot = cv::plot::Plot2d::create(data_x, data_y);
    plot->render(plot_result);

    cv::imshow(title, plot_result);
    cv::waitKey();
    cv::destroyAllWindows();
}

double distance(cv::Point2d p1, cv::Point2d p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

double levene(int n1, int n2, double *arr1, double *arr2)
{

    double *deviance1 = (double *)malloc(sizeof(double) * n1);
    double *deviance2 = (double *)malloc(sizeof(double) * n2);
    double sum1 = 0, sum2 = 0;
    double dev1 = 0, dev2 = 0;
    double sqdev1 = 0, sqdev2 = 0;

    for (int i = 0; i < n1; i++)
        sum1 += arr1[i];
    for (int i = 0; i < n2; i++)
        sum2 += arr2[i];

    for (int i = 0; i < n1; i++)
    {
        deviance1[i] = fabs(arr1[i] - sum1 / n1);
        dev1 += deviance1[i];
    }

    for (int i = 0; i < n2; i++)
    {
        deviance2[i] = fabs(arr2[i] - sum2 / n1);
        dev2 += deviance2[i];
    }

    dev1 /= n1;
    dev2 /= n2;

    for (int i = 0; i < n1; i++)
        sqdev1 += pow(deviance1[i] - dev1, 2);
    for (int i = 0; i < n2; i++)
        sqdev2 += pow(deviance2[i] - dev2, 2);

    double devm = (dev1 + dev2) * 0.5;
    double m = (n1 + n2 - 2) *
               (n1 * pow(dev1 - devm, 2) + n2 * pow(dev2 - devm, 2)) /
               (sqdev1 + sqdev2);

    return 1 - pf(m, 1, n1 + n2 - 2, TRUE, FALSE);
}

double t(int n1, int n2, double *arr1, double *arr2)
{

    double mean1 = 0, mean2 = 0;
    double s1 = 0, s2 = 0;

    for (int i = 0; i < n1; i++)
        mean1 += arr1[i];
    for (int i = 0; i < n2; i++)
        mean2 += arr2[i];
    mean1 /= n1;
    mean2 /= n2;

    for (int i = 0; i < n1; i++)
        s1 += pow(arr1[i] - mean1, 2);
    for (int i = 0; i < n2; i++)
        s2 += pow(arr1[i] - mean2, 2);
    s1 /= (n1 - 1);
    s2 /= (n2 - 1);

    return (mean1 - mean2) /
           sqrt((((n1 - 1.) * s1 + (n2 - 1.) * s2) / (n1 + n2 - 2.)) *
                (1. / n1 + 1. / n2));
}

int imax(int n, double *arr)
{
    int i = 0;
    double m = std::numeric_limits<double>::min();
    for (int j = 0; j < n; j++)
        if (arr[j] > m)
        {
            m = arr[j];
            i = j;
        }
    return i;
}

int amax(int n, double *arr)
{
    int i = 0;
    double m = std::numeric_limits<double>::min();
    for (int j = 0; j < n; j++)
        if (arr[j] > m)
        {
            m = arr[j];
            i = j;
        }
    return m;
}

int imin(int n, double *arr)
{
    int i = 0;
    double m = std::numeric_limits<double>::max();
    for (int j = 0; j < n; j++)
        if (arr[j] < m)
        {
            m = arr[j];
            i = j;
        }
    return i;
}

int amin(int n, double *arr)
{
    int i = 0;
    double m = std::numeric_limits<double>::max();
    for (int j = 0; j < n; j++)
        if (arr[j] < m)
        {
            m = arr[j];
            i = j;
        }
    return m;
}

double normal(int n, double *arr, double x)
{

    double sum = 0;
    double dev = 0;

    for (int i = 0; i < n; i++)
        sum += arr[i];
    for (int i = 0; i < n; i++)
        dev += pow(arr[i] - (sum / n), 2);
    int mean = sum / n;
    int stdvar = sqrt(dev / (n - 1));
    double p = pnorm(x, mean, stdvar, TRUE, FALSE);
    return p > 0.5 ? 1 - p : p;
}

double mean(int n, double *arr)
{

    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += arr[i];
    int mean = sum / n;
    return mean;
}

int boundary(cv::Mat &grayscale, cv::Point2d origin, cv::Point2d step, int maximal, double tolerance)
{
    double o = grayscale.at<uchar>(int(origin.y), int(origin.x)) * 1.0;

    for (float i = 0; i <= maximal; ++i)
    {
        int posy = int(origin.y) + int(round(i * step.y));
        int posx = int(origin.x) + int(round(i * step.x));

        if (posx > grayscale.cols)
            continue;
        if (posy > grayscale.rows)
            continue;
        if (posx < 0)
            continue;
        if (posy < 0)
            continue;

        uchar c = grayscale.at<uchar>(posy, posx);
        if (c < 150)
            return i;
    }

    return maximal;
}

uchar bilinear(uchar p1, uchar p2, uchar p3, uchar p4, double x, double y)
{
    double x1 = (p1 + (p2 - p1) * x);
    double x2 = (p3 + (p4 - p3) * x);
    return uchar(int(x1 + (x2 - x1) * y));
}

uchar get_bilinear(cv::Mat &grayscale, double x, double y)
{
    int borderx = -1, bordery = -1;
    if (x < 0)
        borderx = 0;
    if (y < 0)
        bordery = 0;
    if (x >= grayscale.cols)
        borderx = grayscale.cols - 1;
    if (y >= grayscale.rows)
        bordery = grayscale.rows - 1;

    if (borderx != -1 && bordery != -1)
        return grayscale.at<uchar>(bordery, borderx);

    if (borderx != -1)
    {
        uchar y1 = grayscale.at<uchar>(int(floor(y)), borderx);
        uchar y2 = grayscale.at<uchar>(int(ceil(y)), borderx);
        return uchar(int(y1 + (y2 - y1) * (y - floor(y))));
    }

    if (bordery != -1)
    {
        uchar x1 = grayscale.at<uchar>(bordery, int(floor(x)));
        uchar x2 = grayscale.at<uchar>(bordery, int(ceil(x)));
        return uchar(int(x1 + (x2 - x1) * (x - floor(x))));
    }

    return bilinear(
        grayscale.at<uchar>(int(floor(y)), int(floor(x))),
        grayscale.at<uchar>(int(floor(y)), int(ceil(x))),
        grayscale.at<uchar>(int(ceil(y)), int(floor(x))),
        grayscale.at<uchar>(int(ceil(y)), int(ceil(x))),
        x - floor(x), y - floor(y));
}

void extract_flank(
    cv::Mat &grayscale, cv::Mat &out, cv::Point2d origin,
    cv::Point2d orient, cv::Point2d up,
    int flank, int extend)
{

    int width = extend;
    int height = 2 * flank + 1;

    cv::Mat roi = cv::Mat::zeros(cv::Size(width, height), CV_8U);
    for (int h = -flank; h <= +flank; h++)
    {
        for (int w = 0; w < width; w++)
        {

            cv::Point2d mapped(
                origin.x + w * orient.x + h * up.x,
                origin.y + w * orient.y + h * up.y);

            cv::Point target(w, flank - h);

            if (mapped.x > grayscale.cols)
                continue;
            if (mapped.y > grayscale.rows)
                continue;
            if (mapped.x < 0)
                continue;
            if (mapped.y < 0)
                continue;

            roi.at<uchar>(target.y, target.x) =
                get_bilinear(grayscale, mapped.x, mapped.y);
        }
    }

    roi.copyTo(out);
}

uchar** matrix(cv::Mat& mat) {
    uchar** ptrs = (uchar**) malloc(sizeof(uchar*) * mat.rows);
    for (int r = 0; r < mat.rows; r ++) ptrs[r] = mat.ptr(r);
    return ptrs;
}

// this is not to be called recursively. loops occur directly in infect.
// only extract the logic out to make infect simpler.

int infect_cell(
    uchar** inp, uchar** out, uchar** flag, int width, int height,
    cv::Point center, std::queue<cv::Point> &next,
    std::vector<double> &bg, double cutoff)
{
    int x = center.x;
    int y = center.y;

    bool do_top = true, do_bottom = true, do_left = true, do_right = true;
    if (x == 0)
        do_left = false;
    if (x == width - 1)
        do_right = false;
    if (y == 0)
        do_top = false;
    if (y == height - 1)
        do_bottom = false;

    // penalty

    if (do_left) if(flag[y][x - 1] == 2) return 0;
    if (do_right) if(flag[y][x + 1] == 2) return 0;
    if (do_top) if(flag[y - 1][x] == 2) return 0;
    if (do_bottom) if(flag[y + 1][x] == 2) return 0;

    int is_dirty = 0;

#define infect_point(_x, _y)                                    \
    {                                                           \
        if (flag[_y][_x] != 0)                                  \
        {                                                       \
        }                                                       \
        else                                                    \
        {                                                       \
            double mbg = mean(bg.size(), bg.data());            \
            double pval = fabs((inp[_y][_x] * 1.) - mbg) / mbg; \
            if (pval > cutoff)                                  \
            {                                                   \
                flag[_y][_x] = 2;                               \
            }                                                   \
            else                                                \
            {                                                   \
                flag[_y][_x] = 1;                               \
                bg.push_back(inp[_y][_x] * 1.);                 \
                out[_y][_x] = 255;                              \
                next.push(cv::Point(_x, _y));                   \
                is_dirty += 1;                                  \
            }                                                   \
        }                                                       \
    }

    if (do_left)
        infect_point(x - 1, y);
    if (do_right)
        infect_point(x + 1, y);
    if (do_top)
        infect_point(x, y - 1);
    if (do_bottom)
        infect_point(x, y + 1);

#undef infect_point

    return is_dirty;
}

void infect(cv::Mat &grayscale, cv::Mat &out, cv::Point init, double cutoff)
{

    std::queue<cv::Point> nexts;
    std::vector<double> bg;
    cv::Mat flag = cv::Mat::zeros(grayscale.size(), CV_8U);

    // the init point is ensured previously to not be on the border of images.
    // the four directions has valid pixel surroundings.

    // the flag matrix has the following conventions:
    // 0 - undetermined.
    // 1 - background points.
    // 2 - foreground points.

    bg.push_back(grayscale.at<uchar>(init.y, init.x) * 1.0);

    bg.push_back(grayscale.at<uchar>(init.y - 1, init.x) * 1.0);
    bg.push_back(grayscale.at<uchar>(init.y + 1, init.x) * 1.0);
    bg.push_back(grayscale.at<uchar>(init.y, init.x - 1) * 1.0);
    bg.push_back(grayscale.at<uchar>(init.y, init.x + 1) * 1.0);

    nexts.push(cv::Point(init.x - 1, init.y));
    nexts.push(cv::Point(init.x + 1, init.y));
    nexts.push(cv::Point(init.x, init.y - 1));
    nexts.push(cv::Point(init.x, init.y + 1));

    flag.at<uchar>(init.y, init.x) = 1;
    flag.at<uchar>(init.y - 1, init.x) = 1;
    flag.at<uchar>(init.y + 1, init.x) = 1;
    flag.at<uchar>(init.y, init.x - 1) = 1;
    flag.at<uchar>(init.y, init.x + 1) = 1;

    out.at<uchar>(init.y, init.x) = 255;
    out.at<uchar>(init.y - 1, init.x) = 255;
    out.at<uchar>(init.y + 1, init.x) = 255;
    out.at<uchar>(init.y, init.x - 1) = 255;
    out.at<uchar>(init.y, init.x + 1) = 255;

    // the main loop. if any changes made in the infect_cell call, it returns
    // a non-zero value, otherwise, 0 is returned to indicate a stop.

    uchar** ptr_in = matrix(grayscale);
    uchar** ptr_out = matrix(out);
    uchar** ptr_flag = matrix(flag);

    while (nexts.size() > 0)
    {
        auto point = nexts.front();
        infect_cell(
            ptr_in, ptr_out, ptr_flag, grayscale.cols, grayscale.rows,
            point, nexts, bg, cutoff);
        nexts.pop();
    }
}

void hist(cv::Mat &grayscale, cv::Mat mask) {
    
    const int channels[] = { 0 };
	cv::Mat hist;
	int dims = 1;
	const int histSize[] = { 256 };

	float pranges[] = { 0, 255 }; // for dimension 0.
	const float* ranges[] = { pranges };
	
	cv::calcHist(
        &grayscale, 1, channels, mask, hist, dims,
        histSize, ranges, true, false
    );

    int scale = 2;
	int hist_height = 256;
	cv::Mat hist_img = cv::Mat::zeros(hist_height, 256 * scale, CV_8UC3);
	
    double max_val;
	cv::minMaxLoc(hist, 0, &max_val, 0, 0);

	for (int i = 0; i < 256; i++)
	{
		float bin_val = hist.at<float>(i);
		int intensity = cvRound(bin_val*hist_height / max_val);
		cv::rectangle(
            hist_img, cv::Point(i * scale, hist_height - 1),
            cv::Point((i + 1) * scale - 1, hist_height - intensity),
            cv::Scalar(255, 255, 255)
        );
	}

    show(hist_img, "histogram");
}

int quartile(cv::Mat &grayscale, cv::Mat mask, double lower) {
    
    const int channels[] = { 0 };
	cv::Mat histo;
	int dims = 1;
	const int histSize[] = { 256 };

	float pranges[] = { 0, 255 }; // for dimension 0.
	const float* ranges[] = { pranges };
	
	cv::calcHist(
        &grayscale, 1, channels, mask, histo, dims,
        histSize, ranges, true, false
    );
    
    float sum, accum;
    for (int i = 0; i < 256; i++) sum += histo.at<float>(i);
    for (int i = 0; i < 256; i++) {
        accum += histo.at<float>(i);
        if (accum > sum * lower) return i;
    }

    hist(grayscale, mask);
    return 255;
}

int any(cv::Mat &binary) {
    int count;
    for (int r = 0; r < binary.rows; r ++) {
        auto ptr = binary.ptr(r);
        for(int c = 0; c < binary.cols; c ++) {
            if (ptr[c] > 0) count += 1;
        }
    }
    return count;
}

inline double au(
    double foremean, int foresize,
    double backstct, double backlse,
    int dark, int light, double refsize
) {
    return (backstct - foremean) * foresize;
}