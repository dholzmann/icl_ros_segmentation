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

#include <ICLGeom/PointCloudObject.h>
#include <ICLGeom/PointCloudCreator.h>
#include <ICLUtils/Configurable.h>
#include <ICLCore/DataSegment.h>
#include <ICLGeom/Camera.h>

namespace EdgeDetector{
using namespace cv;


typedef struct { float x, y, z, w; } Point4f;

struct Data {
        int medianFilterSize = 3;
        int normalRange = 2;
        int normalAveragingRange = 1;
        int neighborhoodMode = 0;
        int neighborhoodRange = 3;
        float binarizationThreshold = 0.89;
        bool useNormalAveraging = true;
        bool useGaussSmoothing = false;
        bool usedFilterFlag = true;
        bool usedSmoothingFlag = true;
        int height;
        int width;
        Mat normals;
        Mat avgNormals;
        Mat filteredImage;
        Mat angleImage;
        Mat binarizedImage;
        Mat normalImage;

		Data( std::string usedFilter, bool useAveraging, std::string usedAngle, std::string usedSmoothing, int normalrange,
            int neighbrange, float threshold, int avgrange, int h, int w){
            
            if(usedFilter.compare("median3x3")==0){ //median 3x3
                medianFilterSize = 3;
            }
            else if(usedFilter.compare("median5x5")==0){ //median 5x5
                medianFilterSize = 5;
            }else{
                usedFilterFlag = false;
            }

            normalRange = normalrange;	
            normalAveragingRange = avgrange;	
                
            if(usedAngle.compare("max")==0){//max
                neighborhoodRange = 0;
            }
            else if(usedAngle.compare("mean")==0){//mean
                neighborhoodRange = 1;
            }
            
            if(usedSmoothing.compare("linear")==0){//linear
                usedSmoothingFlag = false;
            }
            else if(usedSmoothing.compare("gaussian")==0){//gauss
                usedSmoothingFlag = true;
            }
            height = h;
            width = w;
            normals = Mat(height, width, CV_32FC4);
            avgNormals = Mat(height, width, CV_32FC4);
            //Mat worldNormals;
            //Mat normalsA;
            //Mat avgNormalsA;
            //Mat worldNormalsA;
            //Mat rawImage;
            filteredImage = Mat::zeros(height, width, CV_32FC1);
            angleImage = Mat(height, width, CV_32FC1);
            binarizedImage = Mat(height, width, CV_32FC1);
            normalImage;
	    }

	    ~Data() {
	    }
};

float flipAngle(float angle);

float maxAngle(float snr, float snl, float snt, float snb, float snbl, float snbr, float sntl, float sntr);

float scalar(Point4f &a, Point4f &b);

void applyLinearNormalAveraging(Data &data);

void applyImageBinarization(Data &data);

void applyAngleImageCalculation(Data &data);

void applyGaussianNormalSmoothing(Data &data);

void applyNormalCalculation(Data &data);
/*
void applyImageBinarization(Mat &binarizedImage, Mat angleImage, int height, int width, double binarizationThreshold);

void applyAngleImageCalculation(Mat &angleI, Mat &norm, int height, int width, int neighborhoodRange, int neighborhoodMode);

void applyLinearNormalAveraging(Mat &normals, Mat &avgNormals, int normalAveragingRange, int height, int width);

//void applyGaussianNormalSmoothing(Mat &normalImage, Size size);

void applyNormalCalculation(Mat &filteredImage, Mat &normals, int normalRange, int normalAveragingRange, bool gauss, bool average, int height, int width);
*/
std::string type2str(int type);

Mat ICLImg_to_Mat(icl::core::Img32f &image, int numChannel);

Mat ICLImg_to_Mat(icl::core::Img8u &image, int numChannel);

void Mat_to_ICLImg(icl::core::Img32f &image, Mat mat, int numChannel);

void Mat_to_ICLImg(icl::core::Img8u &image, Mat mat, int numChannel);

//Mat preSeg_calculate(Mat &depthImage, bool filter, bool average, bool gauss);

Mat preSeg_calculate(Mat &depthImage, Data &data);
}