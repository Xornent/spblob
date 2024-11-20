
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

#include "blobroi.h"

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
#else
#include "argparse/argparse.hpp"
#endif

namespace fs = std::filesystem;
namespace chrono = std::chrono;

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

// ============================================================================

// logging

static FILE* logfile = NULL;
static int save_count = 1;
static char logfpath[1024] = "rois.tsv";
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
    "blobroi: detect and extract regions-of-interest from semen patches on test papers. " soft_br
    "this is the first step in the spblob routines (blobroi, blobshed, blobnn). and as the " soft_br
    "shared step for sample image preprocessing. this program reads one or a directory of " soft_br
    "photographs (in *.jpg format) of test papers, detect each paper by the positioning " soft_br
    "triangle, and yields in a specified output folder the faces and scales of the test papers " soft_br
    "with roughly uniform size. the same output folder should be specified as input data for " soft_br
    "later feature extraction routines (blobshed or blobnn, one of them), using methods either " soft_br
    "derived from watershed-like algorithm or neural network model for object segmentation. \n\n"
    "this software is a free software licensed under gnu gplv3. it comes with absolutely " soft_br
    "no warranty. for details, see <https://www.gnu.org/licenses/gpl-3.0.html>";

static char args_doc[] = 
    "[--save-start N] "
    "[--scale SCALE] [--size SIZE] [--proximal PROX] [--distal DIST] "
    "[--posang-size PSIZE] [--posang-thresh PTHRESH] "
    "[-o OUTPUT] [-d] [-f] INPUT";

#ifdef unix
static struct argp_option options[] = {
    { "save-start", 'n', "N", 0, "starting index of the output dataset clips (0)"},
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

const char *argp_program_version = "spblob:blobroi 1.3";
const char *argp_program_bug_address = "yang-z. <xornent@outlook.com>";
bool set_scale_width = false; // a flag if user defined scale widths manually.
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    
    struct arguments *arguments = (struct arguments*) state -> input;

    switch (key) {
        case 'n':
            arguments -> save_count = atoi(arg);
            save_count = atoi(arg);
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
    strcpy(arguments.data_output_path, datapath);
    arguments.scale_factor = c_scale_factor;
    arguments.scale_width = 60.0;
    arguments.proximal = c_proximal;
    arguments.distal = c_distal;
    arguments.directory = false;
    arguments.fname_as_sample = false;
    strcpy(arguments.input, "\0");

#ifdef unix
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
#else

    argparse::ArgumentParser program("blobshed", "1.5");

    program.add_argument("-n", "--save-start")
        .help("starting index of the output dataset clips (0)")
        .metavar("N")
        .default_value(save_count)
        .scan<'i', int>();

    program.add_usage_newline();

    program.add_argument("-x", "--scale")
        .help(
            "the relative scale factor of the output dataset clips (the image dataset " soft_br
            "for later neural-network based detection routine.) this takes the " soft_br
            "perpendicular edge length of the positioning triangle to be unified into " soft_br
            "fold changes from 10px. the default value requires that for each output image, " soft_br
            "the positioning triangle should have an edge length of 26px. (2.6)" )
        .metavar("SCALE")
        .default_value(c_scale_factor)
        .scan<'f', double>();

    program.add_argument("-s", "--size")
        .help("resolution for the final image. stating that every 1 unit " soft_br
              "in --scale should represent 60px in the dataset image. (60.0)")
        .metavar("SIZE")
        .default_value(60.0)
        .scan<'f', double>();

    program.add_argument("-p", "--proximal")
        .help("proximal detetion position (270.0)")
        .metavar("PROX")
        .default_value(c_proximal)
        .scan<'f', double>();
    
    program.add_argument("-t", "--distal")
        .help("distal detetion position (300.0)")
        .metavar("DIST")
        .default_value(c_distal)
        .scan<'f', double>();

    program.add_usage_newline();

    program.add_argument("-y", "--posang-size")
        .help("the minimal size of the positioning angle (50)")
        .metavar("PSIZE")
        .default_value(size_thresh)
        .scan<'i', int>();

    program.add_argument("-z", "--posang-thresh")
        .help("the red visual intensity threshold for the positioning angle (40)")
        .metavar("PTHRESH")
        .default_value(red_thresh)
        .scan<'i', int>();

    program.add_usage_newline();

    program.add_argument("-o", "--output")
        .help("dataset output directory. must exist prior to running")
        .metavar("OUTPUT")
        .default_value(datapath);

    program.add_argument("-d", "--dir")
        .help("input be a directory of images in *.jpg")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-f", "--fas")
        .help("filename as sample name, accept the file name of the image as the sample " soft_br
              "name without prompting the user to enter the sample names manually")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("input")
        .help("the input image, or a directory of images (when specifying -d)")
        .metavar("input");

    program.add_description(doc);

    try { program.parse_args(argc, argv); }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    arguments.save_count = program.get<int>("-n");
    save_count = arguments.save_count;

    arguments.scale_factor = program.get<double>("--scale");
    c_scale_factor = arguments.scale_factor;

    arguments.scale_width = program.get<double>("--size");
    c_scale_width = arguments.scale_width * c_scale_factor;

    arguments.proximal = program.get<double>("--proximal");
    c_proximal = arguments.proximal;

    arguments.distal = program.get<double>("--distal");
    c_distal = arguments.distal;

    size_thresh = program.get<int>("--posang-size");
    red_thresh = program.get<int>("--posang-thresh");

    strcpy(arguments.data_output_path, program.get("--output").c_str());
    strcpy(datapath, program.get("--output").c_str());

    arguments.directory = program.get<bool>("--dir");
    arguments.fname_as_sample = program.get<bool>("--fas");
    strcpy(arguments.input, program.get("input").c_str());

#endif
    
    // make sure the data path exist, and create subdirectories if they are not.

    std::string opath(datapath);
    if (fs::is_directory(opath)) {

        if (!fs::is_directory(opath + "/sources")) fs::create_directories(opath + "/sources");
        if (!fs::is_directory(opath + "/scales")) fs::create_directories(opath + "/scales");
        if (!fs::is_directory(opath + "/scales.annot")) fs::create_directories(opath + "/scales.annot");

        // open the log file and append.
        // the log file of the blobroi routine is automatically set to be {out}/rois.tsv
    
        char logfname[1024] = "\0";
        strcpy(logfname, datapath);
        strcat(logfname, "/");
        strcat(logfname, logfpath);
        strcpy(logfpath, logfname);
        logfile = fopen(logfpath, "a+");

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

            rois.push_back(roi);

            char roiid[12];
            sprintf(roiid, "%d", rois.size());
            cv::putText(
                annot, roiid,
                cv::Point2d(origin.x / zoom, origin.y / zoom),
                cv::FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(0, 0, 0), 5
            );
        }
    }

    // here, we will extract the scale mark and reads some of the critical
    // information from the scale mark image for finer adjustments.

    std::vector< uchar > scale_dark;
    std::vector< uchar > scale_light;
    std::vector< double > scale_size;
    std::vector< bool > scale_success;
    std::vector< cv::Mat > scale_view;

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

            if (area > 1500) { // TODO: THIS 1000 IS UNSTABLE!
                
                det = true;
                select_id = cid;

                // the contour circles the darker part of the image,
                // but need to keep out the red triangles.

                cv::drawContours(
                    view, contours, cid,
                    cv::Scalar(0), 3
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
        scale_view.push_back(view);

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
    // std::vector<cv::Mat> usms;            sharpened raw image *
    // std::vector<bool> pass1;              pass for clipping flank orientation

    // std::vector< uchar > scale_dark;      darker principle grayscale
    // std::vector< uchar > scale_light;     lighter principle grayscale
    // std::vector< double > scale_size;     scale circle size
    // std::vector< bool > scale_success;    no exception in the scale recognition

    // std::vector< cv::Mat > back_strict;   stricter background mask *
    // std::vector< cv::Mat > back_loose;    looser background mask *
    // std::vector< cv::Mat > foreground;    foreground mask *
    // std::vector< bool > has_foreground;   has a foreground detection *

    int nrpass = 0, nspass = 0;
    for (bool j : pass1) if (j) nrpass += 1;
    for (bool j : scale_success) if (j) nspass += 1;
    printf("  [i] detected: [roi] %d/%d  [scale] %d/%d  \n", nrpass, rois.size(), nspass, scales.size());

    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    double ms = double(duration.count()) * chrono::milliseconds::period::num /
                chrono::milliseconds::period::den;

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

        fprintf(
            logfile,
            
            // format string

            "%d\t%s\t%d\t%s\t%s\t%s\t"
            "%d\t%d\t%.2f\t"
            "%.1f\t%.1f\t%.1f\t%.1f\t%d\t%.4f\t"
            "%.4f\t%.4f\n",
            
            // file catalog

            save_count, file, i + 1, name, strpass1, strpass2, // strpass3, <- has_foreground
            
            // in previous versions of this software, the raw detection parameters
            // are recorded in these four fields: foremean, foresize, and two
            // background means. other statistical features are calculated directly
            // from these and the scale parameters. here, the object blob detection
            // is moved to become subroutines either 'blobshed' for conventional
            // algorithm and 'blobnn' for neural network models. so these parameters
            // are removed from the dump roi table.

            // fm, fsz, backsmean[0], backlmean[0],

            scale_dark.at(i), scale_light.at(i), scale_size.at(i),
            dorigins.at(i)[0], dorigins.at(i)[1],
            dbases.at(i)[0], dbases.at(i)[1],
            dwidths.at(i), zoom,
            dorients.at(i)[0], dorients.at(i)[1]
        );

        fflush(logfile);
        
        char savefname[1024] = "";

        // write the sources (face of the test paper) and scales images.

        char fmtstring_src[1024] = "";
        char fmtstring_scale[1024] = "";
        char fmtstring_scale_annot[1024] = "";
        strcpy(fmtstring_src, datapath);
        strcpy(fmtstring_scale, datapath);
        strcpy(fmtstring_scale_annot, datapath);

        strcat(fmtstring_src, "/sources/%d.jpg");
        strcat(fmtstring_scale, "/scales/%d.jpg");
        strcat(fmtstring_scale_annot, "/scales.annot/%d.jpg");

        sprintf(savefname, fmtstring_src, save_count);
        cv::imwrite(savefname, rois.at(i));

        sprintf(savefname, fmtstring_scale, save_count);
        cv::imwrite(savefname, scales.at(i));

        sprintf(savefname, fmtstring_scale_annot, save_count);
        cv::imwrite(savefname, scale_view.at(i));

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
