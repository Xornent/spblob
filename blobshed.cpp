
#include "blobshed.h"

#define higher_reach 4

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

int start_id = 1;
int end_id = INT32_MAX - 10; // we will calculate end_id + 1, no overflow then.
int max_id = 1;

static FILE* rawfile = NULL;
static FILE* statfile = NULL;
static FILE* roifile = NULL;

static std::vector<char*> rawlines;
static std::vector<char*> statlines;
static std::vector<int> rawuids;
static std::vector<int> statuids;

static char rawfpath[1024] = "raw.tsv";
static char statfpath[1024] = "stats.tsv";
static char datapath[1024] = ".";

// ============================================================================

// windows do not support the glibc's getline function, we need to write our
// own version to use it:

#ifndef unix

#define max_line_len 65535

ssize_t getline(char **lineptr, size_t *n, FILE *stream) {
    char* line = (char*) malloc(max_line_len);
    char* result = fgets(line, max_line_len - 1, stream);
    
    if (result == NULL) return -1;
    line[max_line_len - 1] = '\0';
    *n = strlen(line);
    *lineptr = line;
    return strlen(line);
}

#endif

// ============================================================================

// argument parser

static char doc[] = 
    "blobshed: detect the intensity of semen patches from extracted uniform datasets. "
    "this routine runentirely using traditional image segmentation methods with a "
    "watershed-like algorithm. \n\n"
    "this software is a free software licensed under gnu gplv3. it comes with absolutely "
    "no warranty. for details, see <https://www.gnu.org/licenses/gpl-3.0.html>";

static char args_doc[] = 
    "[--start M] [--end N] [SOURCE]";

#ifdef unix
static struct argp_option options[] = {
    { "start", 'm', "M", 0, "starting index (included) of the uid. (0)"},
    { "end", 'n', "N", 0, "ending index (included) of the uid. (int32-max)"},
    { 0 }
};

const char *argp_program_version = "spblob:blobshed 1.5";
const char *argp_program_bug_address = "yang-z. <xornent@outlook.com>";
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    
    switch (key) {
        case 'm':
            start_id = atoi(arg);
            break;
        case 'n': 
            end_id = atoi(arg);
            break;
        case ARGP_KEY_ARG:
            strcpy(datapath, arg);
            break;
        case ARGP_KEY_END:
            if (state -> arg_num != 1) argp_usage(state);
            break;
        default: return ARGP_ERR_UNKNOWN;
    }

    return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };
#endif

int main(int argc, char* argv[]) 
{
    // read the program parameters

#ifdef unix
    argp_parse(&argp, argc, argv, 0, 0, NULL);
#else

    argparse::ArgumentParser program("blobshed", "1.5");

    program.add_argument("-m", "--start")
        .help("starting index (included) of the uid. (0)")
        .metavar("M")
        .default_value(start_id)
        .scan<'i', int>();

    program.add_argument("-n", "--end")
        .help("ending index (included) of the uid. (int32-max)")
        .metavar("N")
        .default_value(end_id)
        .scan<'i', int>();

    program.add_argument("source")
        .help("the directory of blobroi's output, as the input")
        .metavar("SOURCE");

    program.add_description(doc);

    try { program.parse_args(argc, argv); }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    start_id = program.get<int>("--start");
    end_id = program.get<int>("--end");
    strcpy(datapath, program.get("source").c_str());

#endif
    
    // make sure the data path exist, and create subdirectories if they are not.

    std::string opath(datapath);
    if (fs::is_directory(opath)) {

        if (!fs::is_directory(opath + "/annots")) fs::create_directories(opath + "/annots");
        if (!fs::is_directory(opath + "/masks")) fs::create_directories(opath + "/masks");

        // open the log file and append.
        // the log file of the blobroi routine is automatically set to be {out}/rois.tsv
    
        char rfname[1024] = "\0";
        char sfname[1024] = "\0";
        strcpy(rfname, datapath); strcat(rfname, "/"); strcat(rfname, rawfpath);
        strcpy(rawfpath, rfname);
        strcpy(sfname, datapath); strcat(sfname, "/"); strcat(sfname, statfpath);
        strcpy(statfpath, sfname);
        
        // both the rawfile and statfile are automatically maintained. (newer
        // detections will overwrite the older ones, and if not previously detected,
        // then append to the tail of the file. so we will read the old file first,
        // and modify it in every run.

        // this also suggests that NO TWO INSTANCE OF THIS PROGRAM SHOULD BE RUN
        // WITH THE SAME OUTPUT FOLDER! or this will cause edit conflict.

        FILE* read_raw = fopen(rawfpath, "r");
        FILE* read_stat = fopen(statfpath, "r");
        char * line = NULL;

        if (read_raw != NULL) {
            char* line = NULL;
            size_t len = 0;
            ssize_t read;

            // every time getline returns a new allocated pointer. we will use
            // them throughout the program's lifespan, do not free them!

            while ((read = getline(&line, &len, read_raw)) != -1) {
                char* rline = (char*) malloc(len + 1);
                strncpy(rline, line, len);
                rline[len] = '\0';
                rawlines.push_back(rline);
                
                char* first_col = strchr(rline, '\t');
                *first_col = '\0';
                rawuids.push_back(atoi(rline));
                *first_col = '\t';
            }

            fclose(read_raw);
        }

        if (read_stat != NULL) {
            char* line = NULL;
            size_t len = 0;
            ssize_t read;

            while ((read = getline(&line, &len, read_stat)) != -1) {
                char* sline = (char*) malloc(len + 1);
                strncpy(sline, line, len);
                sline[len] = '\0';
                statlines.push_back(sline);

                char* first_col = strchr(sline, '\t');
                *first_col = '\0';
                statuids.push_back(atoi(sline));
                *first_col = '\t';
            }

            fclose(read_stat);
        }

        rawfile = fopen(rawfpath, "w");
        statfile = fopen(statfpath, "w");

    } else {
        printf("[e] data output path do not exist! \n");
        return 1;
    }

    char logfname[1024] = "\0";
    strcpy(logfname, datapath);
    strcat(logfname, "/rois.tsv");
    std::string roifpath(logfname);

    if (fs::is_regular_file(roifpath)) {
        roifile = fopen(logfname, "r");
    } else {
        printf("[e] do not find rois.tsv under the source folder! \n");
        return 1;
    }

    // processing and reading the rois.tsv from output path.

    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    std::vector<char*> sample_names; std::vector<char*> fnames;
    std::vector<int> sid; std::vector<int> uid;
    std::vector<bool> det_success; std::vector<cv::Mat> rois;
    std::vector<bool> scale_success;
    std::vector<int> scale_dark; std::vector<int> scale_light;

    while ((read = getline(&line, &len, roifile)) != -1) {
        
        if(len <= 1) continue;

        char* sline = (char*) malloc(len + 1);
        strncpy(sline, line, len);
        sline[len] = '\0';

        // read column by column ...

        char* col = strchr(sline, '\t'); *col = '\0';
        int uidx = atoi(sline);

        if (uidx >= start_id && uidx <= end_id) {  }
        else { if (uidx > max_id) max_id = uidx; continue; }

        uid.push_back(uidx); sline = col + 1;

        col = strchr(sline, '\t'); *col = '\0';
        fnames.push_back(sline); sline = col + 1;

        col = strchr(sline, '\t'); *col = '\0';
        sid.push_back(atoi(sline)); sline = col + 1;

        col = strchr(sline, '\t'); *col = '\0';
        sample_names.push_back(sline); sline = col + 1;

        col = strchr(sline, '\t'); *col = '\0';
        det_success.push_back(*sline == 'x'); sline = col + 1;

        col = strchr(sline, '\t'); *col = '\0';
        scale_success.push_back(*sline == 'x'); sline = col + 1;

        col = strchr(sline, '\t'); *col = '\0';
        scale_dark.push_back(atoi(sline)); sline = col + 1;

        col = strchr(sline, '\t'); *col = '\0';
        scale_light.push_back(atoi(sline)); sline = col + 1;

        char fmtstring_src[1024] = "";
        char savefname[1024] = "";
        strcpy(fmtstring_src, datapath);
        strcat(fmtstring_src, "/sources/%d.jpg");
        sprintf(savefname, fmtstring_src, uidx);

        cv::Mat src = cv::imread(savefname, cv::IMREAD_GRAYSCALE);
        rois.push_back(src);
    }

    fclose(roifile);

    process(
        true, sample_names, fnames, sid, uid, det_success,
        rois, scale_success, scale_dark, scale_light
    );

    // finalize.

    fclose(rawfile);
    fclose(statfile);
    return 0;
}

int process(bool show_msg,
            std::vector<char*> sample_names, std::vector<char*> fnames,
            std::vector<int> sid, std::vector<int> uid,
            std::vector<bool> det_success, std::vector<cv::Mat> rois,
            std::vector<bool> scale_success,
            std::vector<int> scale_dark, std::vector<int> scale_light)
{
    std::vector< cv::Mat > back_strict;
    std::vector< cv::Mat > back_loose;
    std::vector< cv::Mat > foreground;
    std::vector< cv::Mat > overlap;
    std::vector< bool > has_foreground;
    std::vector<cv::Mat> usms;

    // generate the usm sharpened images from rois:

    int croi = 0;
    for (auto roi : rois)
    {
        if (!det_success.at(croi)) {
            usms.push_back(cv::Mat(cv::Size(3, 3), CV_8U, cv::Scalar(0)));
            croi += 1;
            continue;
        }

        cv::Mat blurred;
        cv::GaussianBlur(roi, blurred, cv::Size(5, 5), 0);

        cv::Mat blur_usm, usm;
        cv::GaussianBlur(blurred, blur_usm, cv::Size(0, 0), 25);
        cv::addWeighted(blurred, 1.5, blur_usm, -0.5, 0, usm);
            
        blur_usm.release();
        usms.push_back(usm);
        croi += 1;
    }

    /* int */ croi = 0;
    for (auto roi : rois)
    {
        if (!det_success.at(croi)) {
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

            if (show_msg) printf("[.] performing infection for %d ... \r", uid.at(croi - 1));
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

            if (show_msg) printf("[.] correcting infection for %d ... \r", uid.at(croi - 1));
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

    printf("\n");

    // logging generatrion. in this step, we should merge the previous file
    // content (in the order of uids) and overwrite duplicated lines. in fact,
    // the rois vector is ordered by uid (inherited from the ordered rois.tsv)
    // so we just test the duplicated items.

    for (int i = 1; i < start_id; i++) {
        for (int j = 0; j < rawuids.size(); j++) if (rawuids.at(j) == i)
            fprintf(rawfile, "%s", rawlines.at(j));
        
        for (int j = 0; j < statuids.size(); j++) if (statuids.at(j) == i)
            fprintf(statfile, "%s", statlines.at(j));
    }

    for (int i = 0; i < rois.size(); i++) {

        char name[512] = {0};
        strcpy(name, sample_names.at(i));

        char strpass1[2] = ".";
        if (det_success.at(i)) strpass1[0] = 'x';
        else strpass1[0] = '.';

        char strpass2[2] = ".";
        if (scale_success.at(i)) strpass2[0] = 'x';
        else strpass2[0] = '.';

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
            rawfile, "%d\t%s\t%d\t%s\t%s\t%s\t%s\t" "%.2f\t%d\t%.2f\t%.2f\t%d\t%d\n",
            uid.at(i), fnames.at(i), sid.at(i), name, strpass1, strpass2, strpass3,
            fm, fsz, backsmean[0], backlmean[0], scale_dark.at(i), scale_light.at(i)
        );

        // those with defected detection will not occur in stats.tsv. thus the
        // number of rows may be smaller than the raw.tsv. and we should filter out
        // any values that may crash the application when calculating log(0).

        if (det_success.at(i) && scale_success.at(i) && has_foreground.at(i) &&
            fsz > 0 && fm > 0 && (backsmean[0] - fm) > 0 && 
            scale_light.at(i) > 0 && scale_dark.at(i) > 0 &&
            scale_light.at(i) > scale_dark.at(i) &&
            backlmean[0] > 0 && backsmean[0] > 0) {

            fprintf(
                statfile, "%d\t%s\t%d\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\t%s\n",
                uid.at(i), fnames.at(i), sid.at(i),
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

        fflush(rawfile);
        fflush(statfile);
        
        char savefname[1024] = "";
        char fmtstring_annot[1024] = "";
        char fmtstring_mask[1024] = "";
        strcpy(fmtstring_annot, datapath);
        strcpy(fmtstring_mask, datapath);

        strcat(fmtstring_annot, "/annots/%d.jpg");
        strcat(fmtstring_mask, "/masks/%d.jpg");

        sprintf(savefname, fmtstring_annot, uid.at(i));
        cv::imwrite(savefname, overlap.at(i));

        sprintf(savefname, fmtstring_mask, uid.at(i));
        cv::imwrite(savefname, foreground.at(i));
    }

    for (int i = end_id + 1; i <= max_id; i++) {
        for (int j = 0; j < rawuids.size(); j++) if (rawuids.at(j) == i)
            fprintf(rawfile, "%s", rawlines.at(j));
        
        for (int j = 0; j < statuids.size(); j++) if (statuids.at(j) == i)
            fprintf(statfile, "%s", statlines.at(j));
    }

    fflush(rawfile);
    fflush(statfile);
    return 0;
}