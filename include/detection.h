# pragma once

#include <memory>
#include <map>
#include <torch/script.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"


class DetectionDriver{
    public:
        cv::Mat fake_img;
        cv::Size input_size;
        std::map<std::string, torch::jit::script::Module> models;
        std::string device_string;
    
        DetectionDriver(int gpuIndex, const cv::Size& image_size = cv::Size(1920, 1088));
        static void load_model(const std::string& modelname);
        static std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size = cv::Size(640, 640));
        static std::vector<std::vector<Detection>> PostProcessing(const torch::Tensor& detections,
                                                              float pad_w, float pad_h, float scale, const cv::Size& img_shape,
                                                              float conf_thres = 0.4, float iou_thres = 0.6);
        static void ScaleCoordinates(std::vector<Detection>& data, float pad_w, float pad_h,
                                 float scale, const cv::Size& img_shape);
        static torch::Tensor xywh2xyxy(const torch::Tensor& x);
        static void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                 const at::TensorAccessor<float, 2>& det,
                                 std::vector<cv::Rect>& offset_box_vec,
                                 std::vector<float>& score_vec);
};