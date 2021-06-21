#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/ros.h>
#include <ros/message_forward.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include "tf/tfMessage.h"
#include "sensor_msgs/CameraInfo.h"
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

#include <string>
#include <sstream>
#include <iostream>
#include <math.h>

namespace EdgeDetector{
using namespace cv;


typedef struct { float x, y, z, w; } Point4f;

void applyLinearNormalAveraging(Mat &normals, Mat &avgNormals, int normalAveragingRange, int height, int width);

float flipAngle(float angle);

float maxAngle(float snr, float snl, float snt, float snb, float snbl, float snbr, float sntl, float sntr);

float scalar(Point4f &a, Point4f &b);

void applyImageBinarization(Mat &binarizedImage, Mat angleImage, int height, int width, double binarizationThreshold);

void applyAngleImageCalculation(Mat &angleI, Mat &norm, int height, int width, int neighborhoodRange, int neighborhoodMode);

void applyGaussianNormalSmoothing(Mat &normalImage, Size size);

void applyNormalCalculation(Mat &filteredImage, Mat &normals, int normalRange, int normalAveragingRange, bool gauss, bool average, int height, int width);

std::string type2str(int type);

Mat preSeg_calculate(Mat &depthImage, bool filter, bool average, bool gauss);
}