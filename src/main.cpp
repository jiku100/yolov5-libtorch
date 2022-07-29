#include <pthread.h>
#include "detection.h"

void* DetWorker(void* index){

    torch::NoGradGuard no_grad;

    int detGpu = *((int*)index);

    std::string coco_label_path = "../weights/coco.names";

    DetectionDriver detDriver(detGpu, coco_label_path, cv::Size(1920, 1080), cv::Size(1920, 1088));

    std::string model_path_prefix = std::string("/home/seokgyeongshin/torch_models/gpu");
    std::string model_name_prefix = std::string("yolov5");

    std::vector<int> resolutions = {1080};
    std::vector<std::string> models = {"n", "s", "m", "l", "x"};

    std::string model_name;
    std::string model_path;

    for(int resolution: resolutions){
        for(std::string model : models){
            model_name = model_name_prefix + model + std::string("6");
            model_path = model_path_prefix + std::to_string(detGpu) + "/" + std::to_string(resolution) + "p/" + model_name + std::string(".torchscript");
            detDriver.Load_model(model_name, model_path);
        }
    }

    std::string det_test_model_name = "yolov5m6";

    std::string source =std::string("../00013.jpg");
    cv::Mat img = cv::imread(source);
    
    std::vector<torch::jit::IValue> inputs = detDriver.PreProcessing(img);

    torch::jit::IValue output = detDriver.Inference(det_test_model_name, inputs);

    auto detections = output.toTuple()->elements()[0].toTensor();

    auto result = detDriver.PostProcessing(detections, 0.4, 0.6);

    cv::Mat dst;
    detDriver.Draw_output(img, dst, result, true);

    cv::imwrite("../detection_output_gpu" + std::to_string(detGpu) + std::string(".png"), dst);
}


int main(int argc, const char* argv[]) {

    pthread_t workers[8];
    int thr_id;
    std::cout << "Thread Start" << std::endl;

    int a = 1;
    std::cout << "DetWorker "<< a << " Start" << std::endl;
    thr_id = pthread_create(&workers[a-1], NULL, DetWorker, (void*)&a);
    pthread_detach(workers[a-1]);

    int b = 2;
    std::cout << "DetWorker "<< b << " Start" << std::endl;
    thr_id = pthread_create(&workers[b-1], NULL, DetWorker, (void*)&b);
    pthread_detach(workers[b-1]);

    std::cout << "Thread Start End" << std::endl;
    while(true){
    }

    return 0;
}