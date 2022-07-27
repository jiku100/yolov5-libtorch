#include "detection.h"



DetectionDriver::DetectionDriver(int gpuIndex, const cv::Size& image_size){
    this->input_size = image_size;
    this->fake_img = cv::Mat::zeros(input_size, CV_32FC3);
    this->device_string = "cuda:" + std::to_string(gpuIndex);
}


