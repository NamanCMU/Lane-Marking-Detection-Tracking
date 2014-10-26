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
	Mat houghTransform();
	void RANSAC();
	void KF(Mat, Mat);
	Mat _detectedEdges;


protected:
	void intersection(Point2f, Point2f, Point2f, Point2f, Point2f&);
	int findInliers(Point2f);
	void visualize();
	int kernelSize;
	int lowThreshold;
	int ratio;
	int _width, _height;
	int _LMWidth; // Lane Mark Width
	int _thres;
	Mat _img;
	vector<Mat> averageFrames;
	Mat _mask;
	vector<Vec2f> _lines;
	float _rho, _theta, _houghThres;
	Point2f VP;
	
};