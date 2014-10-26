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
	
	// Averaging over last 5 frames
	averageFrames.push_back(_detectedEdges);
	if (averageFrames.size() > 5) averageFrames.erase(averageFrames.begin());
	for (int c = 0;c < averageFrames.size() - 1; c++)
		_detectedEdges += averageFrames[c];
	_detectedEdges /= (1.0* averageFrames.size());
	//

	visualize();
}
//////////

// Performing Hough Transform
Mat laneDetection::houghTransform(){

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
	return _detectedEdges;
	// RANSAC();
}

///// RANSAC to remove Outliers and improve the estimate of the Vanishing Point
// and hence the Lane Markings
void laneDetection::RANSAC(){
	
	int minInliers = 0,num = 0;
	for (int i = 0;i < _lines.size(); i++){
		float x1 = _lines[i][0] * cos(_lines[i][1]);
		float y1 = _lines[i][0] * sin(_lines[i][1]);

		Point p11(cvRound(x1 - 1.0*sin(_lines[i][1])*1000), cvRound(y1 + cos(_lines[i][1])*1000));
		Point p12(cvRound(x1 + 1.0*sin(_lines[i][1])*1000), cvRound(y1 - cos(_lines[i][1])*1000));

		clipLine(_detectedEdges.size(),p11,p12);
		
		for (int j = i + 1;j < _lines.size(); j++){
			float x2 = _lines[j][0] * cos(_lines[j][1]);
			float y2 = _lines[j][0] * sin(_lines[j][1]);

			Point p21(cvRound(x2 - 1.0*sin(_lines[j][1])*1000), cvRound(y2 + cos(_lines[j][1])*1000));
			Point p22(cvRound(x2 + 1.0*sin(_lines[j][1])*1000), cvRound(y2 - cos(_lines[j][1])*1000));

			clipLine(_detectedEdges.size(),p21,p22);

			Point2f r;
			intersection(p11,p12,p21,p22,r);
			num = findInliers(r);
			if (num >= minInliers) VP = r;
		}
	}
}

void laneDetection::intersection(Point2f p11, Point2f p12, Point2f p21, Point2f p22, Point2f &r){
	Point2f x = p21 - p11;
	Point2f vec1 = p12 - p11;
	Point2f vec2 = p22 - p21;

	float cross = vec1.x*vec2.y - vec1.y*vec2.x;
	double t = (x.x * vec2.y - x.y * vec2.x) / cross;
	r = p11 + vec1 * t;
}

int laneDetection::findInliers(Point2f rOrig){

	int num = 0;
	for (int i = 0;i < _lines.size(); i++){
		float x1 = _lines[i][0] * cos(_lines[i][1]);
		float y1 = _lines[i][0] * sin(_lines[i][1]);

		Point p11(cvRound(x1 - 1.0*sin(_lines[i][1])*1000), cvRound(y1 + cos(_lines[i][1])*1000));
		Point p12(cvRound(x1 + 1.0*sin(_lines[i][1])*1000), cvRound(y1 - cos(_lines[i][1])*1000));

		clipLine(_detectedEdges.size(),p11,p12);
		
		for (int j = i + 1;j < _lines.size(); j++){
			float x2 = _lines[j][0] * cos(_lines[j][1]);
			float y2 = _lines[j][0] * sin(_lines[j][1]);

			Point p21(cvRound(x2 - 1.0*sin(_lines[j][1])*1000), cvRound(y2 + cos(_lines[j][1])*1000));
			Point p22(cvRound(x2 + 1.0*sin(_lines[j][1])*1000), cvRound(y2 - cos(_lines[j][1])*1000));

			clipLine(_detectedEdges.size(),p21,p22);

			Point2f r;
			intersection(p11,p12,p21,p22,r);
			int dist = sqrt((r.x - rOrig.x)*(r.x - rOrig.x) + (r.y - rOrig.y)*(r.y - rOrig.y));
			if (dist < 20) num++;
		}
	}
	return num;

}
/////

// Visualize
void laneDetection::visualize(){
	
	namedWindow("Canny");
	namedWindow("Original");

	imshow("Canny",_detectedEdges); // Detected Edges
	imshow("Original",_img); // Original Image

}

void laneDetection::KF(Mat im1, Mat im2){

	vector<KalmanFilter> kk;
	resize(im1,im1,Size(_width,_height));
	resize(im2,im2,Size(_width,_height));
	
	KalmanFilter KF(4,2,0);
	Mat_<float> state(4,1);
	Mat_<float> measurement(2,1);
	measurement.setTo(Scalar(0));

	KF.transitionMatrix = *(Mat_<float>(4,4) << 1,0,1,0,  0,1,0,1,  0,0,1,0,  0,0,0,1);

	KF.statePre.at<float>(0) = 100;
	KF.statePre.at<float>(1) = 100;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;

	setIdentity(KF.measurementMatrix);
	setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(KF.errorCovPost, Scalar::all(1));


	while(1){
		Mat prediction = KF.predict();
		Point predictedPoint(prediction.at<float>(0), prediction.at<float>(1));

		// Getting the next measurement using Optical Flow
		vector<Point2f> prevPoints, nextPoints;
		vector<uchar> status;
		vector<float> err;
		prevPoints.push_back(Point(100,100));
		
		calcOpticalFlowPyrLK(im1,im2,prevPoints,nextPoints,status,err);
		measurement(0) = nextPoints[0].x;
		measurement(1) = nextPoints[0].y;
		//

		// The update phase
 		Mat estimated = KF.correct(measurement);
 		
 		Point statePoint(estimated.at<float>(0),estimated.at<float>(1));
 		Point measurementPoint(measurement(0),measurement(1));

 		cout << "M: " << measurementPoint << " S: " << statePoint << endl;
		
 		circle(im2,statePoint,2,Scalar(0,0,255),-1);
 		circle(im1,Point(100,100),2,Scalar(0,0,255),-1);
 		imshow("Original",im1);
 		imshow("Kalman",im2);

 		waitKey(100000);

	}

	

}

#ifdef LaneTest
int main()
{
	laneDetection detect; // Make the object
	
	int i = 1;
	char ipname[10], opname[10];
	Mat img1 = imread("./images/mono6.png");
	Mat img2 = imread("./images/mono7.png");

	detect.KF(img1,img2);
	// while(i <= 674){
	// 	sprintf(ipname,"./images/mono%d.png",i);
	// 	Mat img = imread(ipname,0); // Read the image
	// 	detect.LMFiltering(img); // Filtering to detect Lane Markings
	// 	Mat opImg = detect.houghTransform(); // Hough Transform
		
	// 	sprintf(opname,"./output/mono%d.png",i);
	// 	imwrite(opname,opImg);
	// 	i++;
	// 	waitKey(100);
	// }
}
#endif