#include <iostream>
#include "laneDetection.h"
#include "CKalmanFilter.h"

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
	_houghThres =100;
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
	
	_detectedEdges = Mat(_img.size(),CV_8UC1); // detectedEdges
	_detectedEdges.setTo(0);

	int val = 0;
	// iterating through each row
	for (int j = _img.rows/2;j<_img.rows;j++){
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
vector<Vec2f> laneDetection::houghTransform(){

	Mat _detectedEdgesRGB;
	cvtColor(_detectedEdges,_detectedEdgesRGB, CV_GRAY2BGR);
	HoughLines(_detectedEdges,_lines,_rho,_theta,_houghThres);
	vector<Vec2f> retVar;
	
	if (_lines.size() > 1){
		Mat labels,centers;
		Mat samples = Mat(_lines.size(),2,CV_32F);

		for (int i = 0;i < _lines.size();i++){
			samples.at<float>(i,0) = _lines[i][0];
			samples.at<float>(i,1) = _lines[i][1];
		}
		// K means to get two lines
		kmeans(samples, 2, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1000, 0.001), 5, KMEANS_PP_CENTERS, centers );

		////////////////// Using RANSAC to get rid of outliers
		_lines.clear();
		
		vector<Point2f> left;
		vector<Point2f> right;
		for(int i = 0;i < labels.rows; i++){
			if (labels.at<int>(i) == 0) left.push_back(Point2f(samples.at<float>(i,0), samples.at<float>(i,1)));
			else right.push_back(Point2f(samples.at<float>(i,0), samples.at<float>(i,1)));
		}
		vector<Point2f> leftR = ransac(left);
		vector<Point2f> rightR = ransac(right);
		if (leftR.size() < 2 || rightR.size() < 2) return retVar;
		
		////////////////

		// // Removing bad lines
		// bool check = (((centers.at<float>(0,1) * 180.0 / CV_PI) < 110 && (centers.at<float>(0,1) * 180.0 / CV_PI) > 70)
		// 			|| ((centers.at<float>(1,1) * 180.0 / CV_PI) < 110 && (centers.at<float>(1,1) * 180.0 / CV_PI) > 70));

		if ((float)(cos((leftR[0].y + leftR[1].y)/2) * cos((rightR[0].y + rightR[1].y)/2)) >= 0) return retVar;

		_lines.push_back(Vec2f((leftR[0].x + leftR[1].x)/2, (leftR[0].y + leftR[1].y)/2));
		_lines.push_back(Vec2f((rightR[0].x + rightR[1].x)/2, (rightR[0].y + rightR[1].y)/2));

	}


	return _lines;
}

// Implementing RANSAC to remove outlier lines
// TO DO: Better implementation 
vector<Point2f> laneDetection::ransac(vector<Point2f> data){

	vector<Point2f> res;
	int maxInliers = 0;

	for(int i = 0;i < data.size();i++){
		Point2f p1 = data[i];

		for(int j = i + 1;j < data.size();j++){
			Point2f p2 = data[j];
			int n = 0;
			
			for (int k = 0;k < data.size();k++){
				Point2f p3 = data[k];
				float normalLength = norm(p2 - p1);
				float distance = abs((float)((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / normalLength);
				if (distance < 0.01) n++;
			}
			
			if (n > maxInliers) {
				res.clear();
				maxInliers = n;			
				res.push_back(p1);
				res.push_back(p2);
			}
		
		}
		
	}

	return res;
}


// Visualize
void laneDetection::visualize(){
	
	namedWindow("Filter");
	namedWindow("Original");

	imshow("Filter",_detectedEdges); // Detected Edges
	imshow("Original",_img); // Original Image

}

// Draw Lines on the image
Mat laneDetection::drawLines(Mat img, vector<Vec2f> lines){

	Mat imgRGB;
	cvtColor(img,imgRGB,CV_GRAY2RGB);

	for (int i = 0;i < lines.size();i++){
		float r = lines[i][0];
		float t = lines[i][1];
		
		float x = r*cos(t);
		float y = r*sin(t);

		Point p1(cvRound(x - 1.0*sin(t)*1000), cvRound(y + cos(t)*1000));
		Point p2(cvRound(x + 1.0*sin(t)*1000), cvRound(y - cos(t)*1000));

		clipLine(img.size(),p1,p2);

		line(imgRGB,p1,p2,Scalar(0,0,255),2);

	}

	return imgRGB;
}

#ifdef LaneTest
int main()
{
	laneDetection detect; // Make the object
	
	int i = 1;
	char ipname[10], opname[10];
	
	sprintf(ipname,"./images/mono%d.png",i);
	Mat img1 = imread(ipname,0); // Read the image
	resize(img1,img1,Size(detect._width,detect._height));
	detect.LMFiltering(img1); // Filtering to detect Lane Markings
	vector<Vec2f> lines = detect.houghTransform(); // Hough Transform
	Mat imgFinal = detect.drawLines(img1, lines);
	i++;
	
	sprintf(ipname,"./output/mono%d.png",i - 1);
	imwrite(ipname,imgFinal); 
	while(i <= 674){
		
		sprintf(ipname,"./images/mono%d.png",i);
		Mat img2 = imread(ipname,0); // Read the image
		resize(img2,img2,Size(detect._width,detect._height));
		i++;
		detect.LMFiltering(img2); // Filtering to detect Lane Markings
		
		vector<Vec2f> lines2 = detect.houghTransform(); // Hough Transform
		if (lines2.size() < 2) {
			imgFinal = detect.drawLines(img2,lines);
			sprintf(opname,"./output/mono%d.png",i - 1);
			imwrite(opname,imgFinal); 
			continue;
		}
		
		CKalmanFilter KF2(lines);
		vector<Vec2f> pp = KF2.predict();

		vector<Vec2f> lines2Final = KF2.update(lines2);
		lines = lines2Final;
		imgFinal = detect.drawLines(img2,lines2);
		
		sprintf(opname,"./output/mono%d.png",i - 1);
		imwrite(opname,imgFinal);

		waitKey(100);
	}
	

}
#endif