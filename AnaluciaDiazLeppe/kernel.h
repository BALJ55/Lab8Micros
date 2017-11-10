#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
#ifndef KERNEL_H
#define KERNEL_H

// compute distance of array to ref in GPU
void wrapper_gpu(Mat input);

#endif
