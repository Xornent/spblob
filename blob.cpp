
#include "blob.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

void show(cv::Mat& matrix, const char* window, int width, int height)
{
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, width, height);
    cv::imshow(window, matrix);
    cv::waitKey();
    cv::destroyAllWindows();
}

void reverse(cv::Mat& binary)
{
    int width = binary.size().width;
    int height = binary.size().height;

    for (int line = 0; line < height; line++)
    {
        uchar* row = binary.ptr<uchar>(line);
        for (int col = 1; col < width; col++)
        {
            row[col] = 255 - row[col];
        }
    }
}

// calculate the perceptible color intensity. a simple transformation from
// the hsv colorspace: cos(delta.H) * S * V.
// orient: the hue [0, 360] to extract color intensity, 0 as red.

void color_significance(cv::Mat& hsv, cv::Mat& grayscale, double orient)
{
    int width = hsv.size().width;
    int height = hsv.size().height;

    for (int line = 0; line < height; line++)
    {
        cv::Vec3b* rhsv = hsv.ptr<cv::Vec3b>(line);
        uchar* rgray = grayscale.ptr<uchar>(line);

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

std::vector<std::pair<uchar, int>> extract_line(
    cv::Mat& grayscale, cv::Point start, cv::Point end)
{

    std::vector<std::pair<uchar, int>> result;
    int row = grayscale.rows;
    int col = grayscale.cols;

    int r1 = start.y;
    int c1 = start.x;
    int r2 = end.y;
    int c2 = end.x;

    // distance between the two anchors
    double dist = round(sqrt(pow(float(r2) - float(r1), 2.0) + pow(float(c2) - float(c1), 2.0)));
    if (dist <= 0.00001f)
    {
        // too short distance. return the origin point.
        std::pair<uchar, int> temp;
        temp.first = grayscale.at<uchar>(r1, c1);
        temp.second = 0;
        result.push_back(temp);
        return result;
    }

    double slope_r = (float(r2) - float(r1)) / dist;
    double slope_c = (float(c2) - float(c1)) / dist;

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

double distance(cv::Point2d p1, cv::Point2d p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

int imax(int n, double* arr)
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

double amax(int n, double* arr)
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

int imin(int n, double* arr)
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

double amin(int n, double* arr)
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

double mean(int n, double* arr)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += arr[i];
    double mean = sum / n;
    return mean;
}

int boundary(cv::Mat& grayscale, cv::Point2d origin, cv::Point2d step, int maximal, double tolerance)
{
    double o = grayscale.at<uchar>(int(origin.y), int(origin.x)) * 1.0;
    std::vector<double> may_be_foregrounds;

    // not confirming the foreground immediately, wait a few px.
    // since the decrease along the border may be smoothed.

    int delay = 10;

    for (int i = 0; i <= maximal; ++i)
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

        // here, applying a static number as the boundary threshold is not good enough
        // and may be unfit to the brightless changes. however, those statistical methods
        // seems to be too sensitive. may just try setting a sharp decrease detection
        // with proportions.

        // just ask the test paper manufacturers to address a black border to the 
        // background just outside the test paper may work :)

        // if (c < 80) return i;

        if (i < delay) may_be_foregrounds.push_back(c * 1.0);
        else {
            int delayposy = int(origin.y) + int(round((i - delay) * step.y));
            int delayposx = int(origin.x) + int(round((i - delay) * step.x));
            uchar cd = grayscale.at<uchar>(delayposy, delayposx);
            double mean_fore = mean(static_cast<int>(may_be_foregrounds.size()), may_be_foregrounds.data());
            if (c < 0.70 * mean_fore) return i;
            else may_be_foregrounds.push_back(cd * 1.0);
        }
    }

    return maximal;
}

uchar bilinear(uchar p1, uchar p2, uchar p3, uchar p4, double x, double y)
{
    double x1 = (p1 + (p2 - p1) * x);
    double x2 = (p3 + (p4 - p3) * x);
    return uchar(int(x1 + (x2 - x1) * y));
}

uchar get_bilinear(cv::Mat& grayscale, double x, double y)
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
    cv::Mat& grayscale, cv::Mat& out, cv::Point2d origin,
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
    uchar** ptrs = (uchar**)malloc(sizeof(uchar*) * mat.rows);
    for (int r = 0; r < mat.rows; r++) ptrs[r] = mat.ptr(r);
    return ptrs;
}

// this is not to be called recursively. loops occur directly in infect.
// only extract the logic out to make infect simpler.

int infect_cell(
    uchar** inp, uchar** out, uchar** flag, int width, int height,
    cv::Point center, std::queue<cv::Point>& next,
    std::vector<double>& bg, double cutoff)
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

void infect(cv::Mat& grayscale, cv::Mat& out, cv::Point init, double cutoff)
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

void hist(cv::Mat& grayscale, cv::Mat mask) {

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

int quartile(cv::Mat& grayscale, cv::Mat mask, double lower) {

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

int any(cv::Mat& binary) {
    int count = 0;
    for (int r = 0; r < binary.rows; r++) {
        auto ptr = binary.ptr(r);
        for (int c = 0; c < binary.cols; c++) {
            if (ptr[c] > 0) count += 1;
        }
    }
    return count;
}

int any_right(cv::Mat& binary, int col) {
    int count = 0;
    for (int r = 0; r < binary.rows; r++) {
        auto ptr = binary.ptr(r);
        for (int c = col; c < binary.cols; c++) {
            if (ptr[c] > 0) count += 1;
        }
    }
    return count;
}
