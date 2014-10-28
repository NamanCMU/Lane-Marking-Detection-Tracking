#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;

class laneDetection{

public:
	laneDetection();
	~laneDetection();
	void findCannyEdges(Mat);
	void LMFiltering(Mat);
	vector<Vec2f> houghTransform();
	Mat drawLines(Mat,vector<Vec2f>);
	Mat _detectedEdges;
	int _width, _height;


protected:
	bool findIntersection(vector<Point>, Point&);
	vector<Point2f> ransac(vector<Point2f>);
	void visualize(Mat);
	int kernelSize;
	int lowThreshold;
	int ratio;
	int _LMWidth; // Lane Mark Width
	int _thres;
	Mat _img;
	vector<Mat> averageFrames;
	Mat _mask;
	vector<Vec2f> _lines;
	float _rho, _theta, _houghThres;
	Point2f VP;
	
};