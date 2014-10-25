#include <iostream>
#include "laneDetection.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	laneDetection Detect;
	
	Mat img = imread("img.png");
	Detect.findCannyEdges(img);
	
	waitKey(100000);

}