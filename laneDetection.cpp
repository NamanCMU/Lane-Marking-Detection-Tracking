#include <iostream>
#include "laneDetection.h"

using namespace cv;

laneDetection::laneDetection(){
	_img = Mat();
	_detectedEdges = Mat();
	kernelSize = 3;
	lowThreshold = 75;
	ratio = 3;
	_width = 800;
	_height = 600;
	_LMWidth = 10;
	_thres = 40;
	_rho = 1;
	_theta = CV_PI/180.0;
	_houghThres = 100;
}

laneDetection::~laneDetection(){

}

// Canny Edge Detector
void laneDetection::findCannyEdges(Mat img){

	///// Generating the mask to mask the top half of the image
	_mask = Mat(img.size(), CV_8UC1, Scalar(1));
	for(int i = 0;i < _mask.rows/2; i++){
		for(int j = 0;j < _mask.cols;j++){
			_mask.at<uchar>(Point(j,i)) = 0;
		}
	}
	img.copyTo(_img,_mask);
	/////

	resize(_img,_img,Size(_width,_height)); // resizing to half (only so that image fits on the screen)
	Canny(_img,_detectedEdges,lowThreshold,lowThreshold*ratio,kernelSize); // Canny Edge Detector
	
	visualize();
}
//


//////////
// Filter to detect lane Markings
// This is just one approach. Other approaches can be Edge detection, Connected Components, etc.
void laneDetection::LMFiltering(Mat src){

	
	///// Generating the mask to mask the top half of the image
	_mask = Mat(src.size(), CV_8UC1, Scalar(1));
	for(int i = 0;i < _mask.rows/2; i++){
		for(int j = 0;j < _mask.cols;j++){
			_mask.at<uchar>(Point(j,i)) = 0;
		}
	}
	src.copyTo(_img,_mask);
	/////

	resize(_img,_img,Size(_width,_height)); // Resizing the image
	
	_detectedEdges = Mat(_img.size(),CV_8UC1); // detectedEdges
	_detectedEdges.setTo(0);

	int val = 0;
	// iterating through each row
	for (int j = 0;j<_img.rows;j++){
		unsigned char *imgptr = _img.ptr<uchar>(j);
		unsigned char *detEdgesptr = _detectedEdges.ptr<uchar>(j);

		// iterating through each column seeing the difference among columns which are "width" apart
		for (int i = _LMWidth;i < _img.cols - _LMWidth; ++i){
			if(imgptr[i]!= 0){
				val = 2*imgptr[i];
				val += -imgptr[i - _LMWidth];
				val += -imgptr[i + _LMWidth];
				val += -abs((int)(imgptr[i - _LMWidth] - imgptr[i + _LMWidth]));

				val = (val<0)?0:val;
				val = (val>255)?255:val;

				detEdgesptr[i] = (unsigned char) val;
			}
		}
	}

	// Thresholding
	threshold(_detectedEdges,_detectedEdges,_thres,255,0);

	visualize();
}
//////////

// Performing Hough Transform
void laneDetection::houghTransform(){

	Mat _detectedEdgesRGB;
	cvtColor(_detectedEdges,_detectedEdgesRGB, CV_GRAY2BGR);
	HoughLines(_detectedEdges,_lines,_rho,_theta,_houghThres);
	
	for (int i = 0;i < _lines.size();i++){
		float r = _lines[i][0];
		float t = _lines[i][1];

		float x = r*cos(t);
		float y = r*sin(t);

		Point p1(cvRound(x - 1.0*sin(t)*1000), cvRound(y + cos(t)*1000));
		Point p2(cvRound(x + 1.0*sin(t)*1000), cvRound(y - cos(t)*1000));

		clipLine(_detectedEdges.size(),p1,p2);

		
		line(_detectedEdgesRGB,p1,p2,Scalar(0,0,255),2);

	}
	imwrite("test.png",_detectedEdgesRGB);
	cout << _lines.size() << endl;
}

// Visualize
void laneDetection::visualize(){
	
	namedWindow("Canny");
	namedWindow("Original");

	imshow("Canny",_detectedEdges); // Detected Edges
	imshow("Original",_img); // Original Image

}

#ifdef LaneTest
int main()
{
	Mat img = imread("img.png",0); // Read the image
	
	laneDetection detect; // Make the object
	detect.LMFiltering(img); // Filtering to detect Lane Markings
	// detect.findCannyEdges(img);
	detect.houghTransform(); // Hough Transform

	waitKey(100000);
}
#endif