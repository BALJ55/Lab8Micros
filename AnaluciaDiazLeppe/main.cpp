// Christian Medina Armas
//modificado por Ana lucia Diaz Leppe 151378
// CC3056
// CUDA
// filtro de memoria compartida 
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
	//el main necesita calcular el tiempo
	double t2 = (double) getTicketCount();
	// compute filter
	//necesitamos agregar el tiempo t2 y definir la ejecucion del tiempo
	filter_gpu(input_img, output_img);
	t2= ((double) getTicketCount()-t2) /getTicketFrequency();
	count << "tiempo de ejecucion: " << t2 << " s" << end1;
	
	imwrite("filter_AnaluciaDiaz.jpeg", output_img);
	cout <<"La imagen saliooo." <<end1;
	return 0;
}
