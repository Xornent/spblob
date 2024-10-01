
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

int main(void) {
    cv::Mat image = cv::imread("test/test-2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat smaller;
    cv::resize(image, smaller, cv::Size(0, 0), 0.4, 0.4);

    cv::Mat blurred;
    cv::GaussianBlur(smaller, blurred, cv::Size(5, 5), 0);
    cv::imshow("blurred", blurred);
    cv::waitKey(0);

    cv::Mat blur_usm, usm;
    cv::GaussianBlur(blurred, blur_usm, cv::Size(0, 0), 25);
    cv::addWeighted(blurred, 1.5, blur_usm, -0.5, 0, usm);
    cv::imshow("sharpened", usm);
    cv::waitKey(0);

    cv::Mat edges;
    cv::Canny(usm, edges, 20, 40);
    cv::imshow("edges", edges);
    cv::waitKey(0);

    std::vector<cv::Vec3f> circles;
    int min_radius = 5;
    int max_radius = 20;
    int canny_upper = 40;
    double hough_upper = 20;
    int min_dist = 15;
    cv::HoughCircles(
        usm, circles, cv::HOUGH_GRADIENT, 1,
        min_dist, canny_upper, hough_upper, min_radius, max_radius
    );

    for (int i = 0; i < circles.size(); i++) {
        cv::Point center (cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = circles[i][2];
        cv::circle(smaller, center, radius, cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("annotated", smaller);
    cv::waitKey(0);

    return 0;
}