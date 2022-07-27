#include "detection.h"

DetectionDriver::DetectionDriver(int gpuIndex, const std::string coco_name_path, const cv::Size& image_size){
    this->model_input_size = image_size;
    this->fake_img = cv::Mat::zeros(model_input_size, CV_32FC3);
    this->device_string = "cuda:" + std::to_string(gpuIndex);

    // load class names
    std::ifstream infile(coco_name_path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            this->class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }
}

std::vector<float> DetectionDriver::LetterboxImage(const cv::Mat& src, cv::Mat& dst){
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = this->model_input_size.height;
    float out_w = this->model_input_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h)- mid_h + 1) / 2;
    int left = (static_cast<int>(out_w)- mid_w) / 2;
    int right = (static_cast<int>(out_w)- mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}

std::vector<torch::jit::IValue> DetectionDriver::PreProcessing(const cv::Mat& img){
    this->image_input_size = img.size();
    cv::Mat img_input = img.clone();
    
    std::vector<float> pad_info = LetterboxImage(img_input, img_input);

    this->pad_w = pad_info[0];
    this->pad_h = pad_info[1];
    this->scale = pad_info[2];

    cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB); 

    img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f);

    auto tensor_img = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()}).to(this->device_string);

    tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);

    return inputs;
}

torch::jit::IValue DetectionDriver::Inference(const std::string& model_name, std::vector<torch::jit::IValue>& inputs){
    if(this->models.find(model_name) != this->models.end()){
        std::cout << model_name << " inference start !!" << std::endl;
        torch::jit::script::Module module_ = this->models.find(model_name)->second;

        torch::jit::IValue output = module_.forward(inputs);
        return output;
    }
    else{
        throw std::runtime_error("Model not loaded");
    }
}

void DetectionDriver::ScaleCoordinates(std::vector<Detection>& data){
    auto clip = [](float n, float lower, float upper) {
        return std::max(lower, std::min(n, upper));
    };

    std::vector<Detection> detections;
    for (auto & i : data) {
        float x1 = (i.bbox.tl().x - this->pad_w)/this->scale;  // x padding
        float y1 = (i.bbox.tl().y - this->pad_h)/this->scale;  // y padding
        float x2 = (i.bbox.br().x - this->pad_w)/this->scale;  // x padding
        float y2 = (i.bbox.br().y - this->pad_h)/this->scale;  // y padding

        x1 = clip(x1, 0, this->image_input_size.width);
        y1 = clip(y1, 0, this->image_input_size.height);
        x2 = clip(x2, 0, this->image_input_size.width);
        y2 = clip(y2, 0, this->image_input_size.height);

        i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
    }
}

torch::Tensor DetectionDriver::xywh2xyxy(const torch::Tensor& x){
    auto y = torch::zeros_like(x);
    // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
    y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
    y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
    y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
    y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
    return y;
}

void DetectionDriver::Tensor2Detection(const at::TensorAccessor<float, 2>& offset_boxes,
                                const at::TensorAccessor<float, 2>& det,
                                std::vector<cv::Rect>& offset_box_vec,
                                std::vector<float>& score_vec) {

    for (int i = 0; i < offset_boxes.size(0) ; i++) {
        offset_box_vec.emplace_back(
                cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
                         cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y]))
        );
        score_vec.emplace_back(det[i][Det::score]);
    }
}

std::vector<std::vector<Detection>> DetectionDriver::PostProcessing(const torch::Tensor& detections, float conf_thres, float iou_thres){
    constexpr int item_attr_size = 5;
    int batch_size = detections.size(0);
    // number of classes, e.g. 80 for coco dataset
    auto num_classes = detections.size(2) - item_attr_size;

    // get candidates which object confidence > threshold
    auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

    std::vector<std::vector<Detection>> output;
    output.reserve(batch_size);

    // iterating all images in the batch
    for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        // apply constrains to get filtered detections for current image
        auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes + item_attr_size});

        // if none detections remain then skip and start to process next image
        if (0 == det.size(0)) {
            continue;
        }

        // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
        det.slice(1, item_attr_size, item_attr_size + num_classes) *= det.select(1, 4).unsqueeze(1);

        // box (center x, center y, width, height) to (x1, y1, x2, y2)
        torch::Tensor box = this->xywh2xyxy(det.slice(1, 0, 4));

        // [best class only] get the max classes score at each result (e.g. elements 5-84)
        std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

        // class score
        auto max_conf_score = std::get<0>(max_classes);
        // index
        auto max_conf_index = std::get<1>(max_classes);

        max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
        max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

        // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
        det = torch::cat({box.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

        // for batched NMS
        constexpr int max_wh = 4096;
        auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
        auto offset_box = det.slice(1, 0, 4) + c;

        std::vector<cv::Rect> offset_box_vec;
        std::vector<float> score_vec;

        // copy data back to cpu
        auto offset_boxes_cpu = offset_box.cpu();
        auto det_cpu = det.cpu();
        const auto& det_cpu_array = det_cpu.accessor<float, 2>();

        // use accessor to access tensor elements efficiently
        this->Tensor2Detection(offset_boxes_cpu.accessor<float,2>(), det_cpu_array, offset_box_vec, score_vec);

        // run NMS
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);

        std::vector<Detection> det_vec;
        for (int index : nms_indices) {
            Detection t;
            const auto& b = det_cpu_array[index];
            t.bbox =
                    cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]),
                             cv::Point(b[Det::br_x], b[Det::br_y]));
            t.score = det_cpu_array[index][Det::score];
            t.class_idx = det_cpu_array[index][Det::class_idx];
            det_vec.emplace_back(t);
        }

        this->ScaleCoordinates(det_vec);

        // save final detection for the current image
        output.emplace_back(det_vec);
    } // end of batch iterating

    return output;
}

void DetectionDriver::Draw_output(cv::Mat& img, cv::Mat& dst, const std::vector<std::vector<Detection>>& detections, bool label){
    if (!detections.empty()) {
        dst = img.clone();
        for (const auto& detection : detections[0]) {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(dst, box, cv::Scalar(0, 0, 255), 2);

            if (label) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = this->class_names[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline=0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(dst,
                        cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                        cv::Point(box.tl().x + s_size.width, box.tl().y),
                        cv::Scalar(0, 0, 255), -1);
                cv::putText(dst, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
            }
        }
    }
}

void DetectionDriver::Load_model(const std::string& model_name, const std::string& model_path){
    try {
        auto start = std::chrono::high_resolution_clock::now();
        // Deserialize the ScriptModule from a file using torch::jit::load().
        torch::jit::script::Module module_ = torch::jit::load(model_path);
        module_.to(this->device_string);
        module_.eval();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "GPU " << this->device_string << " load " << model_name << " time : " << duration.count() << " us" << std::endl;

        auto fake_infer_start = std::chrono::high_resolution_clock::now();

        this->fake_inputs = this->PreProcessing(this->fake_img);

        for(int i = 0; i<5; i++){
            this->fake_output = module_.forward(this->fake_inputs);
            this->fake_detections = fake_output.toTuple()->elements()[0].toTensor();
            this->fake_result = this->PostProcessing( this->fake_detections, 0.4, 0.6);
        }

        this->models.insert({model_name, module_});

        auto fake_infer_end = std::chrono::high_resolution_clock::now();
        auto fake_infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(fake_infer_end - fake_infer_start);
        std::cout << "GPU " << this->device_string << " fake inference " << model_name << " time : " << fake_infer_duration.count() << " us" << std::endl;
        std::cout << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model!\n";
        std::exit(EXIT_FAILURE);
    }
}