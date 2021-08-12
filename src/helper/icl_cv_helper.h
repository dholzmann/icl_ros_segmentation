#include <ICLGeom/PointCloudObject.h>
#include <ICLGeom/PointCloudCreator.h>
#include <ICLUtils/Configurable.h>
#include <ICLCore/DataSegment.h>
#include <ICLGeom/Camera.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

using namespace cv;

Mat ICLImg_to_Mat(icl::core::Img32f &image, Mat &mat, int numChannel);

Mat ICLImg_to_Mat(icl::core::Img8u &image, Mat &mat, int numChannel);

void Mat_to_ICLImg(icl::core::Img32f &image, Mat &mat, int numChannel);

void Mat_to_ICLImg(icl::core::Img8u &image, Mat &mat, int numChannel);

void Mat_to_DataSegment(icl::core::DataSegment<float, 4> &image, Mat &mat, int h, int w);

void DataSegment_to_Mat(icl::core::DataSegment<float, 4> &image, Mat &mat, int h, int w);