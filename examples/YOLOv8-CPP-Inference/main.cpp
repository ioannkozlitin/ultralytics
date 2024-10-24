#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "/home/ivan/proj/ultralytics"; // Set your ultralytics base path

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //
    cv::Size frame_size(640, 480);

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/yolov8_0.523.onnx", frame_size, "classes.txt", runOnGPU);
    cv::VideoCapture video_stream;
    video_stream.open("/home/ivan/video/1.mp4", cv::CAP_FFMPEG);

    //std::vector<std::string> imageNames;
    //imageNames.push_back(projectBasePath + "/ultralytics/assets/bus.jpg");
    //imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");

    cv::Mat frame;
    //for (int i = 0; i < imageNames.size(); ++i)
    while (video_stream.read(frame))
    {
        cv::resize(frame, frame, frame_size);
        //cv::Mat frame = cv::imread(imageNames[i]);

        // Inference starts here...
        double t1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        std::vector<Detection> output = inf.runInference(frame);
        double t2 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        std::cout << "Inference time: " << (t2 - t1) * 1000 << "ms\n";

        int detections = output.size();
        //std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        //float scale = 0.8;
        //cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);
        if(cv::waitKey(1) == 27)
            break;
    }
}
