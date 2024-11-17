
// conditional compilation switches ===========================================

// enable color filtering: the anchor triangles will be filtered to retain
//     those with perceptible red color.

int red_thresh = (40);
int size_thresh = (50);
#define filter_color

// anchor detection modes:
//
// threshold: direct thresholding (static). while a normalization step is
//     required to tolerate more variable images, current tests suggests that
//     this static threshold is enough for now. if you use raw grayscale, the
//     thresholding step is rather dissatisfying. however, if you use the
//     extracted red-ness degrees (i figured it out through a simple hsv transformation)
//     the red parts is extensively highlighted and the threshold make sense,
//     and this color transform overperform all attempts before.

#define anchordet_threshold

// operation mode:
//
// wholeimage: process as a whole.
//
// debug switch: this will allow you to specify the region of interest when in
//     split-image mode, and will toggle on the verbose field, so you will see
//     images of each step showing up. you can specify the region of interest
//     with debug_w and debug_h.

#undef debug
#undef verbose
#define mode_wholeimage

// ============================================================================

#include <iostream>
#include <filesystem>
#include <chrono>

#ifdef unix
#include <argp.h>
#endif

namespace fs = std::filesystem;
namespace chrono = std::chrono;

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/plot.hpp>

#include "blob.h"

// ============================================================================

// logging

static FILE* logfile = NULL;
static FILE* statfile = NULL;
static int save_count = 1;
static char logfpath[1024] = "raw.tsv";
static char statfpath[1024] = "stats.tsv";
static char datapath[1024] = ".";

// ============================================================================

// geometric constants

// india and china-short constant set.

// #define c_scale_factor (2)
// #define c_pair_distance_threshold (30.0 * c_scale_factor)
// #define c_scale_width (60.0 * c_scale_factor)
// #define c_proximal (210.0)
// #define c_distal (240.0)

// china-long constant set.

// #define c_scale_factor (2)
// #define c_pair_distance_threshold (30.0 * c_scale_factor)
// #define c_scale_width (60.0 * c_scale_factor)
// #define c_proximal (310.0)
// #define c_distal (340.0)

// india-wide constant set.

static double c_scale_factor = (2.6);
static double c_pair_distance_threshold = (30.0 * c_scale_factor);
static double c_scale_width = (60.0 * c_scale_factor);
static double c_proximal = (270.0);
static double c_distal = (300.0);

#ifdef debug
#define verbose
#endif

// ============================================================================

// argument parser

static char doc[] = 
    "spblob [conventional]: detect intensity parameters from semen patches on test papers. "
    "this software does not rely on neural network model. but entirely by traditional "
    "image segmentation with a watershed-like algorithm.";

static char args_doc[] = 
    "[--save-start N] [--raw RAW] [--stat STAT] \n"
    "[--scale SCALE] [--size SIZE] [--proximal PROX] [--distal DIST] \n"
    "[--posang-size PSIZE] [--posang-thresh PTHRESH] \n"
    "[-o OUTPUT] [-d] [-f] INPUT";

#ifdef unix
static struct argp_option options[] = {
    { "save-start", 'n', "N", 0, "starting index of the output dataset clips (0)"},
    { "raw", 'l', "RAW", 0, "location of the output raw data file (raw.tsv)"},
    { "stat", 'a', "STAT", 0, "location of the statistics summary file (stats.tsv)"},
    { "scale", 'x', "SCALE", 0,
      "the relative scale factor of the output dataset clips (the image dataset for later neural-network "
      "based detection routine. this takes the perpendicular edge length of the positioning triangle "
      "to be unified into fold changes from 10px. the default value requires that for each output image, "
      "the positioning triangle should have an edge length of 26px. (2.6)" },
    { "posang-size", 'y', "PSIZE", 0, "the minimal size of the positioning angle (50)" },
    { "posang-thresh", 'z', "PTHRESH", 0, "the red visual intensity threshold for the positioning angle (40)" },
    { "size", 's', "SIZE", 0, "resolution for the final image. stating that every 1 "
      "unit in --scale should represent 60px in the dataset image. (60.0)"},
    { "proximal", 'p', "PROX", 0, "proximal detetion position (270.0)" },
    { "distal", 't', "DIST", 0, "distal detetion position (300.0)" },
    { "output", 'o', "OUTPUT", 0, "dataset output directory. must exist prior to running"},
    { "dir", 'd', 0, 0, "input be a directory of images in *.jpg"}, 
    { "fas", 'f', 0, 0, "filename as sample, accept the file name of the image as the sample name "
      "without prompting the user to enter the sample names manually"}, 
    { 0 }
};

const char *argp_program_version = "spblob/c 1.0";
const char *argp_program_bug_address = "yang-z. <xornent@outlook.com>";
bool set_scale_width = false; // a flag if user defined scale widths manually.
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    
    struct arguments *arguments = (struct arguments*) state -> input;

    switch (key) {
        case 'n':
            arguments -> save_count = atoi(arg);
            save_count = atoi(arg);
            break;
        case 'l': 
            strcpy(arguments -> log_file_path, arg);
            strcpy(logfpath, arg);
            break;
        case 'a':
            strcpy(arguments -> stat_file_path, arg);
            strcpy(statfpath, arg);
            break;
        case 'x':
            arguments -> scale_factor = atof(arg);
            c_scale_factor = arguments -> scale_factor;
            break;
        case 's':
            arguments -> scale_width = atof(arg);
            set_scale_width = true;
            break;
        case 'p':
            arguments -> proximal = atof(arg);
            c_proximal = atof(arg);
            break;
        case 't':
            arguments -> distal = atof(arg);
            c_distal = atof(arg);
            break;
        case 'o':
            strcpy(arguments -> data_output_path, arg);
            strcpy(datapath, arg);
            break;
        case 'd':
            arguments -> directory = true;
            break;
        case 'f':
            arguments -> fname_as_sample = true;
            break;
        case 'y':
            size_thresh = atoi(arg);
            break;
        case 'z':
            red_thresh = atoi(arg);
            break;
        case ARGP_KEY_ARG:
            strcpy(arguments -> input, arg);
            break;
        case ARGP_KEY_END:
            if (state -> arg_num != 1) argp_usage(state);
            if (arguments -> input[0] == 0) argp_usage(state);
            if (set_scale_width)
                c_scale_width = arguments -> scale_width * c_scale_factor;
            break;
        default: return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };
#endif

int main(int argc, char *argv[])
{
    struct arguments arguments;
    arguments.save_count = save_count;
    strcpy(arguments.log_file_path, logfpath);
    strcpy(arguments.stat_file_path, statfpath);
    strcpy(arguments.data_output_path, datapath);
    arguments.scale_factor = c_scale_factor;
    arguments.scale_width = c_scale_width;
    arguments.proximal = c_proximal;
    arguments.distal = c_distal;
    arguments.directory = false;
    arguments.fname_as_sample = false;
    strcpy(arguments.input, "\0");

#ifdef unix
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
#else

    if (argc != 14) {
        printf(args_doc);
        return 1;
    }

    arguments.save_count = atoi(argv[1]);
    save_count = arguments.save_count;

    strcpy(arguments.log_file_path, argv[2]);
    strcpy(logfpath, argv[2]);

    strcpy(arguments.stat_file_path, argv[3]);
    strcpy(statfpath, argv[3]);

    arguments.scale_factor = atof(argv[4]);
    c_scale_factor = arguments.scale_factor;

    arguments.scale_width = atof(argv[5]);
    c_scale_width = arguments.scale_width * c_scale_factor;

    arguments.proximal = atof(argv[6]);
    c_proximal = arguments.proximal;

    arguments.distal = atof(argv[7]);
    c_distal = arguments.distal;

    size_thresh = atoi(argv[8]);
    red_thresh = atoi(argv[9]);
    
    strcpy(arguments.data_output_path, argv[10]);
    strcpy(datapath, argv[10]);
     
    arguments.directory = strcmp(argv[11], "-d") == 0;
    arguments.fname_as_sample = strcmp(argv[12], "-f") == 0;
    strcpy(arguments.input, argv[13]);

#endif

    // open the log file and append.

    logfile = fopen(logfpath, "a+");
    statfile = fopen(statfpath, "a+");

    // make sure the data path exist, and create subdirectories if they are not.

    std::string opath(datapath);
    if (fs::is_directory(opath)) {
        if (!fs::is_directory(opath + "/sources")) fs::create_directories(opath + "/sources");
        if (!fs::is_directory(opath + "/scales")) fs::create_directories(opath + "/scales");
        if (!fs::is_directory(opath + "/annots")) fs::create_directories(opath + "/annots");
        if (!fs::is_directory(opath + "/masks")) fs::create_directories(opath + "/masks");
    } else {
        printf("[e] data output path do not exist! \n");
        return 1;
    }

    if (arguments.directory)
    {
        char *dir = arguments.input;
        std::string path(dir);
        for (const auto &entry : fs::directory_iterator(path))
        {
            if (!entry.is_directory())
            {
                // needed to add those .string() before .c_str(). without this works fine
                // on linux, but not on msys-windows platforms.

#ifdef unix
                printf("processing %s ... \n", (char *)(entry.path().filename().c_str()));
                double dur = process(
                    (char*) (entry.path().c_str()),
                    (char*) (entry.path().filename().replace_extension().c_str()),
                    true, &arguments
                );
#else
                printf("processing %s ... \n", (char *)(entry.path().filename().string().c_str()));
                double dur = process(
                    (char*) (entry.path().string().c_str()),
                    (char*) (entry.path().filename().replace_extension().string().c_str()),
                    true, &arguments
                );
#endif
                
                printf("< %.3f s\n", dur);
            }
        }
    }
    else
    {
        std::string path(arguments.input);
        fs::path entry = path;

#ifdef unix
        printf("processing %s ... \n", (char*) entry.filename().replace_extension().c_str());
        double dur = process(
            arguments.input,
            (char*) entry.filename().replace_extension().c_str(),
            true, &arguments
        );
#else
        printf("processing %s ... \n", (char*) entry.filename().replace_extension().string().c_str());
        double dur = process(
            arguments.input,
            (char*) entry.filename().replace_extension().string().c_str(),
            true, &arguments
        );
#endif

        printf("< %.3f s\n", dur);
    }

    fclose(logfile);
    fclose(statfile);

    return 0;
}

double process(char *file, char* purefname, bool show_msg, struct arguments* args)
{
    // read the specified image in different color spaces.

    cv::Mat grayscale = cv::imread(file, cv::IMREAD_GRAYSCALE);
    cv::Mat colored = cv::imread(file, cv::IMREAD_COLOR);

    cv::Mat colored_hsv;
    cv::cvtColor(colored, colored_hsv, cv::COLOR_BGR2HSV);

    cv::Mat component_red;
    grayscale.copyTo(component_red);
    color_significance(colored_hsv, component_red, 0.0);

#ifdef verbose
    show(component_red, "red");
#endif

    cv::Mat annot;
    colored.copyTo(annot);

    double zoom_first_round = 1;

    auto start = chrono::system_clock::now();

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

    // here, we will scale the image to a relatively uniform size. and infer
    // the relative center for each detection.

    double zoom = anch.zoom;
    printf("zoom: %.4f \n", zoom);

    if (isnan(zoom) || zoom < 0) {
        printf("  [e] aborting. \n");
        return 0;
    }

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
            if (distance(meeting_points[i].second, meeting_points[j].second) < c_pair_distance_threshold)
            { // FIXME. CHANGE
                paired.push_back(std::pair<int, int>(
                    meeting_points[i].first, meeting_points[j].first));
                base_meeting.push_back(cv::Point2d(
                    (meeting_points[i].second.x + meeting_points[j].second.x) * 0.5,
                    (meeting_points[i].second.y + meeting_points[j].second.y) * 0.5));
            }
        }
    }

    if (paired.size() == 0) {
        printf("  [e] no paired positioning triangles detected. \n");
        printf("  [e] aborting. \n");
        return 0;
    }

    // correct the scale factor zoom

    double avgmark = 0;
    for (int i = 0; i < paired.size(); i++) {
        cv::Point2d v1 = base_vertice[paired[i].first];
        cv::Point2d v2 = base_vertice[paired[i].second];
        avgmark += distance(v1, v2);
    }

    avgmark /= paired.size();
    double original_zoom = zoom;
    printf(
        "corrected zoom: %.4f, %d, %.4f * %.4f \n", 
        avgmark, paired.size(), zoom, (c_scale_width / avgmark)
    );
    zoom *= (c_scale_width / avgmark);

    if (isnan(zoom)) {
        printf("  [e] nan error. \n");
        return 0;
    }

    cv::Mat scaled_gray, scaled_color;
    cv::resize(grayscale, scaled_gray, cv::Size(0, 0), zoom, zoom);
    cv::resize(colored, scaled_color, cv::Size(0, 0), zoom, zoom);

    for(int i = 0; i < meeting_points.size(); i++)
        meeting_points[i].second = cv::Point2d(
            meeting_points[i].second.x * zoom / original_zoom,
            meeting_points[i].second.y * zoom / original_zoom
        );
    
    for(int i = 0; i < base_vertice.size(); i++)
        base_vertice[i] = cv::Point2d(
            base_vertice[i].x * zoom / original_zoom,
            base_vertice[i].y * zoom / original_zoom
        );
    
    for(int i = 0; i < base_meeting.size(); i++)
        base_meeting[i] = cv::Point2d(
            base_meeting[i].x * zoom / original_zoom,
            base_meeting[i].y * zoom / original_zoom
        );

    // calculate the grayscale on the uncorrected base line.

    std::vector<cv::Mat> rois;
    std::vector<cv::Mat> scales;
    std::vector<cv::Mat> usms;
    std::vector<bool> pass1;

    std::vector< cv::Vec2d > dorigins;
    std::vector< cv::Vec2d > dorients;
    std::vector< cv::Vec2d > dbases;
    std::vector< int > dwidths;

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
            cv::Point2d(upx, upy), (distance(vtop, vbottom) / 2 - 3), distance(vtop, vbottom) * 354 / 325.0
            // for a short version. 125.
        );

        scales.push_back(scale_bar);
        
        // search for meeting boundary

        int maximal_search_length = int(100. / zoom);

        // here, the 160 and 180 is associated with the default zoom constant
        // 68.28 (in filter_color) which indicated the zoomed image is set to
        // a uniform length of 20px of the scale bar.

        cv::Point2d orig_b1((origin.x + unifx * c_proximal) / zoom, (origin.y + unify * c_proximal) / zoom); // FIXME: CHANGE
        cv::Point2d orig_b2((origin.x + unifx * c_distal) / zoom, (origin.y + unify * c_distal) / zoom); // FIXME: CHANGE
        
        // for a short version 160 and 180.

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
            dorigins.push_back(cv::Point2d(origin.x / zoom, origin.y / zoom));
            dbases.push_back(orig_b1);
            dorients.push_back(cv::Point2d(corrorientx, corrorienty));
            dwidths.push_back(int(width) * 2 + 1);

            if (corratio < 0.1) { pass1.push_back(true); }
            else { 
                pass1.push_back(false);

                // placeholder to ensure the length of vector
                usms.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
                rois.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
                continue;
            }

            int roih = int(width);
            int roiw = 350;

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

            char roiid[12];
            sprintf(roiid, "%d", usms.size());
            cv::putText(
                annot, roiid,
                cv::Point2d(origin.x / zoom, origin.y / zoom),
                cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(0, 0, 0), 5
            );
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

#define higher_reach 4

        bool detected = false;
        int maxiter = 4 + higher_reach;

        double fthreshs[4 + higher_reach] = {
            0.02,   0.025,  0.032,  0.04,   0.05, 
            0.0625, 0.0781, 0.0977 /*, 0.122,
            0.15,   0.18,   0.22 */
        };

        double cthreshs[4 + higher_reach] = {
            0.045,  0.056,  0.07,   0.09,   0.12,
            0.15,   0.1875, 0.2344 /*, 0.29,
            0.36,   0.5,   0.75 */
        };

        double finethresh = fthreshs[3 + higher_reach];
        double coarsethresh = cthreshs[3 + higher_reach]; 
        double circularity;

        while ((!detected) && maxiter > 0) {

            maxiter -= 1;
            finethresh = fthreshs[maxiter];
            coarsethresh = cthreshs[maxiter];
            bgstrict = cv::Mat::zeros(roi.size(), CV_8U);
            bgloose = cv::Mat::zeros(roi.size(), CV_8U);
            
            cv::cvtColor(roi, ol, cv::COLOR_GRAY2BGR);

            if (show_msg) printf("  [.] performing infection for %d ... \r", croi);
            fflush(stdout);
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

                if (area > 1000 && area < 50000) {

                    fg = cv::Mat::zeros(roi.size(), CV_8U);
                    cv::drawContours(fg, contours, idc, cv::Scalar(255), cv::FILLED);

                    int collapse_right = any_right(fg, fg.cols - 20);
                    if (collapse_right < 10) {
                        cv::drawContours(
                            ol, contours, idc, cv::Scalar(0, 0, 255), 2
                        );
                        circularity = ratio;
                        detected = true;
                        break;
                    }

                } else {
                    cv::drawContours(ol, contours, idc, cv::Scalar(0, 0, 0), 1);
                }

                idc ++;
            }
        }

        // TODO: the higher threshold may be too invasive for the circle detection.
        // however, for some images (where objects are too sticked to the border)
        // such invasiveness is required to strip the subject from the 
        // surroundings. however, these objects may not be round, and may lose
        // the gradients border of natural color. if the effects are mild, we
        // will just solve the problem by the 2 or 3 times of dilation when counting
        // but sometimes the shape itself is far from round and the loss cannot be reversed

        bool nextround = true;
        bool update = false;
        cv::Mat backup_fg, backup_ol;
        fg.copyTo(backup_fg);
        ol.copyTo(backup_ol);

        while (detected && nextround) {
            
            coarsethresh *= 0.64;
            bgloose = cv::Mat::zeros(roi.size(), CV_8U);

            if (show_msg) printf("  [.] correcting infection for %d ... \r", croi);
            fflush(stdout);
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

            int idc = 0;
            bool hasany = false;
            for (auto cont : contours) {
                double lenconts = cv::arcLength(cont, true);
                double area = cv::contourArea(cont, false);
                double ratio = lenconts * lenconts / area;
                
                // the circularity ratio should decrease (more circular)
                // after each iteration.

                if (area > 2000 && area < 50000) {
                    
                    // update the foreground mask.

                    int collapse_right = any_right(fg, fg.cols - 20);
                    if (collapse_right < 10) {
                        if (ratio < circularity * 0.95) {
                            backup_fg = cv::Mat::zeros(roi.size(), CV_8U);
                            cv::drawContours(backup_fg, contours, idc, cv::Scalar(255), cv::FILLED);
                            cv::drawContours(
                                backup_ol, contours, idc, cv::Scalar(0, 255, 0), 2);
                            hasany = true;
                            update = true;
                            circularity = ratio;

                        } else nextround = false;
                        break;
                    }
                }

                idc ++;
            }

            if (!hasany) {
                nextround = false;
            }
        }

        if (update) {
            backup_fg.copyTo(fg);
            backup_ol.copyTo(ol);
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

        cv::threshold(usm, usm, 0, 255, cv::THRESH_OTSU);
        reverse(usm);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(usm, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat darker_mask(sc.size(), CV_8U, cv::Scalar(0));
        cv::Mat lighter_mask(sc.size(), CV_8U, cv::Scalar(255));

        int cid = 0;
        int select_id = -1;
        bool det = false;
        cv::Mat view; sc.copyTo(view);

        for (auto cont : contours) {
            double lenconts = cv::arcLength(cont, true);
            double area = cv::contourArea(cont, false);
            double ratio = lenconts * lenconts / area;

            if (area > 1000) {
                
                det = true;
                select_id = cid;

                // the contour circles the darker part of the image,
                // but need to keep out the red triangles.

                cv::drawContours(
                    view, contours, cid,
                    cv::Scalar(255), cv::FILLED
                );

                cv::drawContours(
                    darker_mask,

                    // relatively shrink the circle to make the darker area more pure.

                    contours, cid,
                    cv::Scalar(255), cv::FILLED
                );

                cv::drawContours(
                    lighter_mask,

                    // relatively extends the circle, note that the two small red
                    // triangle marks lies within lighter mask but with distinct
                    // grayscale compared to the background. we may just use the 
                    // median filter to ignore them.

                    contours, cid,
                    cv::Scalar(0), cv::FILLED
                );
            }

            cid ++;
        }
       
        scale_dark.push_back(quartile(sc, darker_mask, 0.40));
        scale_light.push_back(quartile(sc, lighter_mask, 0.60));

        if (det) {
            scale_success.push_back(true);
            scale_size.push_back(cv::contourArea(contours[select_id], false));
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
    // std::vector< cv::Mat > foreground;    annotated foreground mask
    // std::vector< bool > has_foreground;   has a foreground detection

    printf("\n");

    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double ms = double(duration.count()) * chrono::milliseconds::period::num /
                chrono::milliseconds::period::den;

#ifdef verbose
    for (auto fore : overlap) {
        show(fore, "foreground", 800, 600);
    }
#endif

    if (!args -> fname_as_sample) {
        cv::namedWindow("annotated", cv::WINDOW_NORMAL);
        cv::resizeWindow("annotated", 800, 600);
        cv::imshow("annotated", annot);
        cv::waitKey();
        cv::destroyAllWindows();
    }

    // logging generatrion. 

    char lastname[512] = {0};

    for (int i = 0; i < rois.size(); i++) {

        char name[512] = {0};
        if (!args -> fname_as_sample) {
            printf("  [%2d] input name: ", i + 1);
            scanf("%s", name);
        } else strcpy(name, purefname);

        if (strcmp(name, ".") == 0) strcpy(name, lastname);
        else strcpy(lastname, name);

        char strpass1[2] = ".";
        if (pass1.at(i)) strpass1[0] = 'x';
        else strpass1[0] = '.';

        char strpass2[2] = ".";
        if (scale_success.at(i)) strpass2[0] = 'x';
        else strpass2[0] = 'x';

        char strpass3[3] = ".";
        if (has_foreground.at(i)) strpass3[0] = 'x';
        else strpass3[0] = '.';

        double fm = -1; int fsz = -1;
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
        }

        auto backsmean = cv::mean(rois.at(i), back_strict.at(i));
        auto backlmean = cv::mean(rois.at(i), back_loose.at(i));

        fprintf(
            logfile, "%s\t%d\t%d\t%s\t%s\t%s\t%s\t%.2f\t%d\t%.2f\t%.2f\t%d\t%d\t%.2f\t%.1f\t%.1f\t%.1f\t%.1f\t%d\t%.4f\t%.4f\t%.4f\n",
            file, i + 1, save_count, name, strpass1, strpass2, strpass3,
            fm, fsz, backsmean[0], backlmean[0],
            scale_dark.at(i), scale_light.at(i), scale_size.at(i),
            dorigins.at(i)[0], dorigins.at(i)[1],
            dbases.at(i)[0], dbases.at(i)[1],
            dwidths.at(i), zoom,
            dorients.at(i)[0], dorients.at(i)[1]
        );

        // those with defected detection will not occur in stats.tsv. thus the
        // number of rows may be smaller than the raw.tsv. and we should filter out
        // any values that may crash the application when calculating log(0).

        if (pass1.at(i) && scale_success.at(i) && has_foreground.at(i) &&
            fsz > 0 && fm > 0 && (backsmean[0] - fm) > 0 && 
            scale_light.at(i) > 0 && scale_dark.at(i) > 0 &&
            scale_light.at(i) > scale_dark.at(i) &&
            backlmean[0] > 0 && backsmean[0] > 0) {

            fprintf(
                statfile, "%s\t%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%s\n",
                file, i + 1,
                save_count,                                  // uid
                log((backsmean[0] - fm) * fsz),              // log.abs
                log(scale_light.at(i) - scale_dark.at(i)),   // log.delta
                log(scale_light.at(i)),                      // log.light
                log(scale_dark.at(i)),                       // log.dark
                log(backlmean[0]),                           // log.back
                log(backsmean[0]),                           // log.back.strict
                log(fm),                                     // log.mean
                log(fsz),                                    // log.sz
                name                                         // sample
            );
        }

        fflush(logfile);
        fflush(statfile);
        
        char savefname[1024] = "";
        char fmtstring_src[1024] = "";
        char fmtstring_scale[1024] = "";
        char fmtstring_annot[1024] = "";
        char fmtstring_mask[1024] = "";
        strcpy(fmtstring_src, datapath);
        strcpy(fmtstring_scale, datapath);
        strcpy(fmtstring_annot, datapath);
        strcpy(fmtstring_mask, datapath);

        strcat(fmtstring_src, "/sources/%d.jpg");
        strcat(fmtstring_scale, "/scales/%d.jpg");
        strcat(fmtstring_annot, "/annots/%d.jpg");
        strcat(fmtstring_mask, "/masks/%d.jpg");

        sprintf(savefname, fmtstring_src, save_count);
        cv::imwrite(savefname, rois.at(i));

        sprintf(savefname, fmtstring_scale, save_count);
        cv::imwrite(savefname, scales.at(i));

        sprintf(savefname, fmtstring_annot, save_count);
        cv::imwrite(savefname, overlap.at(i));

        sprintf(savefname, fmtstring_mask, save_count);
        cv::imwrite(savefname, foreground.at(i));

        save_count += 1;
    }

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

    cv::Mat morph = cv::Mat::zeros(usm.size(), CV_8UC1);
    cv::threshold(usm, morph, red_thresh, 255, cv::THRESH_BINARY);

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

            double area = cv::contourArea(contours[i], false);
            double length = cv::arcLength(contours[i], true);
            double ratio = length * length / area;
            
            if (ratio < 15 || ratio > 25)
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
        double area = cv::contourArea(contours[i], false);

        double h = contour_mean[0];
        if (h < 90)
            h += 180;
        double s = contour_mean[1];
        double v = contour_mean[2];

#ifdef filter_color
        if ((h > 140 || h < 220) && s > 80 && v > 30 && area >= size_thresh)
        {
#else  
        if (area >= size_thresh)
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

    if (filter_indices.size() == 0) {
        printf("  [e] no valid positioning angle passed for the color filter\n");
        printf("  [e] this probably because the image is too small, or the redness of triangle being \n");
        printf("  [e] influcenced by the photographing conditions. consider -y and -z options. \n");
        anchors.zoom = -1;
    }

    total_length /= filter_indices.size();
    free(anchors.vertices);

    anchors.detections = filter_indices.size();
    anchors.vertices = array;
    anchors.zoom = ((34.14 * c_scale_factor) / total_length) * zoom;

#ifdef verbose

    show(smaller, "annotated colored", 800, 600);
    printf("designated zoom: %4f\n", anchors.zoom);

#endif
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
        if (c < 80)
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

    int is_dirty = 0;

#define infect_point(_x, _y)                                    \
    {                                                           \
        if (flag[_y][_x] != 0)                                  \
        {                                                       \
        }                                                       \
        else                                                    \
        {                                                       \
            double mbg = mean(bg.size(), bg.data());            \
            double pval = (mbg - (inp[_y][_x] * 1.)) / mbg;     \
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

    if (do_left) {
        infect_point(x - 1, y);
        if (do_top) infect_point(x - 1, y - 1);
        if (do_bottom) infect_point(x - 1, y + 1);
    }

    if (do_right) {
        infect_point(x + 1, y);
        if (do_top) infect_point(x + 1, y - 1);
        if (do_bottom) infect_point(x + 1, y + 1);
    }

    if (do_top) {
        infect_point(x, y - 1);
        if (do_left) infect_point(x - 1, y - 1);
        if (do_right) infect_point(x + 1, y - 1);
    }

    if (do_bottom) {
        infect_point(x, y + 1);
        if (do_left) infect_point(x - 1, y + 1);
        if (do_right) infect_point(x + 1, y + 1);
    }

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
	cv::Mat histo;
	int dims = 1;
	const int histSize[] = { 256 };

	float pranges[] = { 0, 255 }; // for dimension 0.
	const float* ranges[] = { pranges };
	
	cv::calcHist(
        &grayscale, 1, channels, mask, histo, dims,
        histSize, ranges, true, false
    );

    int scale = 2;
	int hist_height = 256;
	cv::Mat hist_img = cv::Mat::zeros(hist_height, 256 * scale, CV_8UC3);
	
    double max_val;
	cv::minMaxLoc(histo, 0, &max_val, 0, 0);

	for (int i = 0; i < 256; i++)
	{
		float bin_val = histo.at<float>(i);
		int intensity = cvRound(bin_val * hist_height / max_val);
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
    
    float sum = 0, accum = 0;
    for (int i = 0; i < 256; i++) sum += histo.at<float>(i);
    for (int i = 0; i < 256; i++) {
        accum += histo.at<float>(i);
        if (accum > sum * lower) return i;
    }

    return 255;
}

int any(cv::Mat &binary) {
    int count = 0;
    for (int r = 0; r < binary.rows; r ++) {
        auto ptr = binary.ptr(r);
        for(int c = 0; c < binary.cols; c ++) {
            if (ptr[c] > 0) count += 1;
        }
    }
    return count;
}

int any_right(cv::Mat &binary, int col) {
    int count = 0;
    for (int r = 0; r < binary.rows; r ++) {
        auto ptr = binary.ptr(r);
        for(int c = col; c < binary.cols; c ++) {
            if (ptr[c] > 0) count += 1;
        }
    }
    return count;
}
