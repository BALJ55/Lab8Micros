#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

#ifndef KERNEL_H
#define KERNEL_H

// execute a filter operation
void filter_gpu(Mat input, Mat output);

#endif
