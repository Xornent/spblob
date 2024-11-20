
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

int process(
    bool show_msg,
    std::vector<char*> sample_names, std::vector<char*> fnames,
    std::vector<int> sid, std::vector<int> uid,
    std::vector<bool> det_success, std::vector<cv::Mat> rois,
    std::vector<bool> scale_success,
    std::vector<int> scale_dark, std::vector<int> scale_light
);
