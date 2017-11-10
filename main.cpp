// Christian Medina Armas
// CC3056
// CUDA
// Apply a filter to an image

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdlib.h>

#include "kernel.h"

using namespace cv;
using namespace std;

int main(){

	Mat input_img;
	input_img = imread("ramphastosSulphuratus.jpeg", CV_LOAD_IMAGE_GRAYSCALE);

	if(! input_img.data ){
		cout<< "Failed to open the image!"<< endl;
		return -1;
	}

	// create a zero filled Mat of the input image size
	Mat output_img = Mat::zeros(Size(input_img.cols, input_img.rows), CV_8UC1);

	// compute filter
	filter_gpu(input_img, output_img);

	imwrite("output.jpg", output_img);
	return 0;
}
