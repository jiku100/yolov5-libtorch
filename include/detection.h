# pragma once

#include <memory>
#include <map>
#include <iostream>
#include <chrono>

#include <torch/script.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


enum Det {
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};

struct Detection {
    cv::Rect bbox;
    float score;
    int class_idx;
};


class DetectionDriver{
    public:

        cv::Size image_input_size;
        cv::Size model_input_size;
        std::vector<std::string> class_names;


        cv::Mat fake_img;
        std::vector<torch::jit::IValue> fake_inputs;
        torch::jit::IValue fake_output;
        torch::Tensor fake_detections;
        std::vector<std::vector<Detection>> fake_result;

        std::map<std::string, torch::jit::script::Module> models;
        std::string device_string;

        float pad_w;
        float pad_h;
        float scale;

        DetectionDriver(int gpuIndex, const std::string coco_name_path, const cv::Size& image_input_size = cv::Size(1920, 1080), const cv::Size& model_input_size = cv::Size(1920, 1088));
        void Load_model(const std::string& model_name, const std::string& model_path);
        std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst);
        std::vector<torch::jit::IValue> PreProcessing(const cv::Mat& img);
        torch::jit::IValue Inference(const std::string& model_name, std::vector<torch::jit::IValue>& inputs);
        void ScaleCoordinates(std::vector<Detection>& data);
        torch::Tensor xywh2xyxy(const torch::Tensor& x);
        void Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                 const at::TensorAccessor<float, 2>& det,
                                 std::vector<cv::Rect>& offset_box_vec,
                                 std::vector<float>& score_vec);
        std::vector<std::vector<Detection>> PostProcessing(const torch::Tensor& detections, float conf_thres=0.4, float iou_thres=0.6);
        void Draw_output(cv::Mat& img, cv::Mat& dst, const std::vector<std::vector<Detection>>& detections, bool label = true);
};