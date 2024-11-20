
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

#include "blob.h"

struct arguments {
    int save_count;
    char data_output_path[1024];
    double scale_factor;
    double pair_distance;
    double scale_width;
    double proximal;
    double distal;
    char input[1024];
    bool directory;
    bool fname_as_sample;
};

typedef struct anchors {
    int detections;
    int* vertices;
    double zoom;
} anchors_t;

double process(char *file, char* purefname, bool show_msg, struct arguments* args);
void anchor(cv::Mat &image, anchors_t &anchors, double prepzoom);
void filter_mean_color(cv::Mat &colored, anchors_t &anchors);
