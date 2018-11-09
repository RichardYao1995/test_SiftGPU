#include <SiftGPU.h>
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <chrono>
#include <GL/gl.h>

using namespace std;
using namespace chrono;

int main( int argc, char** argv)
{
    //声明SiftGPU并初始化
    SiftGPU sift;
    char* myargv[5] = { "-m", "-s", "-unpa", "0"};
    //char* myargv[4] = {"-fo", "-1", "-cuda", "0"};
    sift.ParseParam(5, myargv);

    //检查硬件是否支持SiftGPU
    int support = sift.CreateContextGL();
    if ( support != SiftGPU::SIFTGPU_FULL_SUPPORTED )
    {
        std::cerr << "SiftGPU is not supported!" << std::endl;
        return 2;
    }

    sift.ParseParam(5, myargv);
    cv::Mat img = cv::imread("/home/yao/workspace/image_construction/image/2.png");
    int width = img.cols;
    int height = img.rows;
    
    sift.AllocatePyramid(width, height);
    sift.SetTightPyramid(1);
    auto start_siftgpu = std::chrono::system_clock::now();
    sift.RunSIFT(width, height, img.data, GL_RGB, GL_UNSIGNED_BYTE);
    float time_cost = chrono::duration_cast<microseconds>(std::chrono::system_clock::now() - start_siftgpu).count() / 1000.0;
    std::cout << "siftgpu::runSIFT() cost time=" << time_cost << "ms" << std::endl;
    int num = sift.GetFeatureNum();
    std::cout << "Feature number=" << num << std::endl;
    std::vector<float> descriptors(128*num);
    std::vector<SiftGPU::SiftKeypoint> keys(num);
    
    auto start_siftfeature = std::chrono::system_clock::now();
    sift.GetFeatureVector(&keys[0], &descriptors[0]);
   
    return 0;
}
