#include <iostream>
#include "laneDetection.h"
#include "CKalmanFilter.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	laneDetection detect;

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

	}

}