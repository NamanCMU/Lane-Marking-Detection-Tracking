#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

class laneDetection{

public:
	laneDetection();
	~laneDetection();
	void findCannyEdges(Mat);
	void LMFiltering(Mat);
	void houghTransform();

protected:
	
	void visualize();
	int kernelSize;
	int lowThreshold;
	int ratio;
	int _width, _height;
	int _LMWidth; // Lane Mark Width
	int _thres;
	Mat _img;
	Mat _detectedEdges;
	Mat _mask;
	vector<Vec2f> _lines;
	float _rho, _theta, _houghThres;
	
};