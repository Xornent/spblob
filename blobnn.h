
#include "blob.h"

int process(
    bool show_msg,
    std::vector<char*> sample_names, std::vector<char*> fnames,
    std::vector<int> sid, std::vector<int> uid,
    std::vector<bool> det_success, std::vector<cv::Mat> rois,
    std::vector<bool> scale_success,
    std::vector<int> scale_dark, std::vector<int> scale_light
);
