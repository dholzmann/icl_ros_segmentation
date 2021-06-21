// https://docs.ros.org/en/api/rosbag_storage/html/c++/
// http://wiki.ros.org/ros_type_introspection/Tutorials/ReadingRosbagUsingIntrospection
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a952ef1a85d70a510240cb645a90efc0d
// https://github.com/ccny-ros-pkg/ccny_rgbd_tools/blob/0.1.1/ccny_rgbd/src/rgbd_util.cpp#L308



#include "preSegmentation.h"

using namespace cv;
namespace EdgeDetector{



Mat getPCfromDepth(Mat &depth_image, Mat &calibration, int &height, int &width){
    Mat pointCloud = Mat::zeros(depth_image.size(), CV_32FC3);
    float fx_inv = 1.0 / calibration.at<float>(0,0);
    float fy_inv = 1.0 / calibration.at<float>(1,1);
    float cx = calibration.at<float>(2,0);
    float cy = calibration.at<float>(2,1);

    for(int i=0; i < height; i++){
        for(int j=0; j < width; j++){
            float z = depth_image.at<float>(i, j);
            Point3f* p = pointCloud.ptr<Point3f>(i, j);
            if(z != 0) {
                float z_metric = z / (1.0 + pow((i-cx) * fx_inv, 2) + pow((j-cy) * fy_inv, 2));
                p->x = z_metric * (i-cx) * fx_inv;
                p->y = z_metric * (j-cy) * fy_inv;
                p->z = z_metric;
            } else {
                p->x = p->y = p->z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    return pointCloud;
}

void applyWorldNormalCalculation(Mat &normalImage, Mat &avgNormals, Mat &worldNormals, Mat &rawImage, Mat &R, Mat &T2, int &height, int &width, bool useNormalAveraging) {
	//FixedMatrix<float, 3, 3> R = T.part<0, 0, 3, 3>();
	//Mat T2 = R.transp().resize<4, 4>(0);
	//T2(3, 3) = 1;
	Mat d = rawImage;
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			if (d.at<float>(y, x) == 2047) {
				normalImage.at<int>(x, y, 0) = 0;
				normalImage.at<int>(x, y, 1) = 0;
				normalImage.at<int>(x, y, 2) = 0;
			} else {
				Mat pWN;
				if (useNormalAveraging == true) {
					pWN = T2 * Mat(avgNormals.at<Point3f>(y, x));
				} else {
					pWN = T2 * Mat(normalImage.at<Point3f>(y, x));
				}
				worldNormals.at<Point4f>(y, x).x = -pWN.at<float>(0);
				worldNormals.at<Point4f>(y, x).y = -pWN.at<float>(1);
				worldNormals.at<Point4f>(y, x).z = -pWN.at<float>(2);
				worldNormals.at<Point4f>(y, x).w = 1.;

                normalImage.at<int>(x, y, 0) = (int) std::abs(pWN.at<float>(0) * 255.);
				normalImage.at<int>(x, y, 1) = (int) std::abs(pWN.at<float>(1) * 255.);
				normalImage.at<int>(x, y, 2) = (int) std::abs(pWN.at<float>(2) * 255.);
			}
		}
	}
}

float flipAngle(float angle){
    if (angle < cos(M_PI / 2)){
	    angle = cos(M_PI - acos(angle));
	}
	return angle;
}

float maxAngle(float snr, float snl, float snt, float snb,
                                      float snbl, float snbr, float sntl, float sntr){
    float max = snr;
	if (max > snl) {
		max = snl;
	}
	if (max > snt) {
		max = snt;
	}
	if (max > snb) {
		max = snb;
	}
	if (max > snbl) {
		max = snbl;
	}
	if (max > snbr) {
		max = snbr;
	}
	if (max > sntl) {
		max = sntl;
	}
	if (max > sntr) {
		max = sntr;
	}
	return max;
}

float scalar(Point4f &a, Point4f &b){
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

void applyNormalCalculation(Data &data) {
    int r = data.normalRange;
    for (int y = 0; y < data.height; y++) {
        for (int x = 0; x < data.width; x++) {
            //int i = x + width * y;
            //std::cout << y << "," << x << std::endl;
            //std::cout << filteredImage.at<float>(532+r, 2-r) << std::endl;
            //std::cout << filteredImage.at<float>(532-r, 2-r) << std::endl;
            if (y < r || y >= data.height- r || x < r || x >= data.width - r){
                Point4f* p = data.normals.ptr<Point4f>(y, x);
                p->x = 0;
                p->y = 0;
                p->z = 0;
            } else {
                // cross product normal determination
                
                float fa1[3], fb1[3], n1[3], n01[3];
                
                fa1[0] = (x + r) - (x - r);
				fa1[1] = (y - r) - (y - r);
				//fa1[2] = filteredImage.at<float>(x + r, y - r)
				//		- filteredImage.at<float>(x - r, y - r);
                fa1[2] = data.filteredImage.at<float>(y - r, x + r)
						- data.filteredImage.at<float>(y - r, x - r);
                
				fb1[0] = (x) - (x - r);
				fb1[1] = (y + r) - (y - r);
				//fb1[2] = filteredImage.at<float>(x, y + r)
				//		- filteredImage.at<float>(x - r, y - r);
                fb1[2] = data.filteredImage.at<float>(y + r, x)
						- data.filteredImage.at<float>(y - r, x - r);
                
				n1[0] = fa1[1] * fb1[2] - fa1[2] * fb1[1];
				n1[1] = fa1[2] * fb1[0] - fa1[0] * fb1[2];
				n1[2] = fa1[0] * fb1[1] - fa1[1] * fb1[0];
                
                float norm = sqrt(pow(n1[0],2)+ pow(n1[1],2) + pow(n1[2],2));
				n01[0] = n1[0] / norm;
				n01[1] = n1[1] / norm;
				n01[2] = n1[2] / norm;
                
                data.normals.ptr<Point4f>(y, x)->x = n01[0];
			    data.normals.ptr<Point4f>(y, x)->y = n01[1];
				data.normals.ptr<Point4f>(y, x)->z = n01[2];
				data.normals.ptr<Point4f>(y, x)->w = 1;
            }
        }
    }
    if (data.useNormalAveraging && !data.useGaussSmoothing) {
        applyLinearNormalAveraging(data);
    } else if (data.useNormalAveraging && data.useGaussSmoothing) {
        GaussianBlur(data.normals, data.avgNormals, Size(data.normalAveragingRange,data.normalAveragingRange), 0);
    }
}

void applyLinearNormalAveraging(Data &data){
    const int r = data.normalAveragingRange;
    
	for (int y = 0; y < data.height; y++) {
		for (int x = 0; x < data.width; x++) {
			if (y < r || y >= data.height - r || x < r
					|| x >= data.width - r) {
				data.avgNormals.at<Point4f>(y, x) = data.normals.at<Point4f>(y, x);
			} else {
				Point4f avg;
				avg.x = 0, avg.y = 0, avg.z = 0, avg.w = 0;
				for (int sx = -r; sx <= r; sx++) {
					for (int sy = -r; sy <= r; sy++) {
						avg.x += data.normals.at<Point4f>((x + sx), (y + sy)).x;
                        avg.y += data.normals.at<Point4f>((x + sx), (y + sy)).y;
						avg.z += data.normals.at<Point4f>((x + sx), (y + sy)).z;
					}
				}
				avg.x /= ((1 + 2 * r) * (1 + 2 * r));
				avg.y /= ((1 + 2 * r) * (1 + 2 * r));
				avg.z /= ((1 + 2 * r) * (1 + 2 * r));
				avg.w = 1;
				data.avgNormals.at<Point4f>(y, x) = avg;
			}
		}
	}
}

void applyImageBinarization(Data &data) {
	for (int y = 0; y < data.height; y++) {
		for (int x = 0; x < data.width; x++) {
			if (data.angleImage.at<float>(x, y) > data.binarizationThreshold) {
				data.binarizedImage.at<float>(x, y) = 255;
			} else {
				data.binarizedImage.at<float>(x, y) = 0;
			}
		}
	}
}

void applyAngleImageCalculation(Data &data) {
	const int w = data.width, h = data.height;
    Mat norm;
    if (data.useNormalAveraging == true) {
		norm = data.avgNormals;
	} else {
		norm = data.normals;
	}
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (y < data.neighborhoodRange
					|| y >= h - (data.neighborhoodRange)
					|| x < data.neighborhoodRange
					|| x >= w - (data.neighborhoodRange)) {
				data.angleImage.at<float>(y, x) = 0;
			} else {
				float snr = 0; //sum right
				float snl = 0; //sum left
				float snt = 0; //sum top
				float snb = 0; //sum bottom
				float sntr = 0; //sum top-right
				float sntl = 0; //sum top-left
				float snbr = 0; //sum bottom-right
				float snbl = 0; //sum bottom-left
				for (int z = 1; z <= data.neighborhoodRange; z++) {
					//angle between normals
                    snr += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y, x+z))));
					snl += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y, x-z))));
					snt += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y+z, x))));
					snb += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y-z, x))));
					sntr += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y+z, x+z))));
					sntl += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y+z, x-z))));
					snbr += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y-z, x+z))));
					snbl += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y-z, x-z))));
				}
				snr /= data.neighborhoodRange;
				snl /= data.neighborhoodRange;
				snt /= data.neighborhoodRange;
				snb /= data.neighborhoodRange;
				sntr /= data.neighborhoodRange;
				sntl /= data.neighborhoodRange;
				snbr /= data.neighborhoodRange;
				snbl /= data.neighborhoodRange;

				if (data.neighborhoodMode == 0) {//max
					data.angleImage.at<float>(y, x) = maxAngle(snr, snl, snt, snb,
                                            snbl, snbr, sntl, sntr);
				} else if (data.neighborhoodMode == 1) {//mean
					data.angleImage.at<float>(y, x) = (snr + snl + snt + snb
							+ sntr + sntl + snbr + snbl) / 8;
				} else {
				}
			}
		}
	}
}

void applyImageBinarization(Mat &binarizedImage, Mat angleImage, int height, int width, double binarizationThreshold) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (angleImage.at<float>(x, y) > binarizationThreshold) {
				binarizedImage.at<float>(x, y) = 255;
			} else {
				binarizedImage.at<float>(x, y) = 0;
			}
		}
	}
}

void applyAngleImageCalculation(Mat &angleI, Mat &norm, int height, int width, int neighborhoodRange, int neighborhoodMode) {
	const int w = width, h = height;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			//int i = x + w * y;
            //std::cout << y << "," << x << std::endl;
			if (y < neighborhoodRange
					|| y >= h - (neighborhoodRange)
					|| x < neighborhoodRange
					|| x >= w - (neighborhoodRange)) {
				angleI.at<float>(y, x) = 0;
			} else {
                //std::cout << 1 << std::endl;
				float snr = 0; //sum right
				float snl = 0; //sum left
				float snt = 0; //sum top
				float snb = 0; //sum bottom
				float sntr = 0; //sum top-right
				float sntl = 0; //sum top-left
				float snbr = 0; //sum bottom-right
				float snbl = 0; //sum bottom-left
                //std::cout << 2 << std::endl;
				for (int z = 1; z <= neighborhoodRange; z++) {
                    
					//angle between normals
					//flip if angle is bigger than 90°
                    //Mat d = norm.at<Mat>(y, x);
                    //std::cout << 1 << std::endl;
                    //Mat f = norm.at<Mat>(y, x+z);
                    //std::cout << 2 << std::endl;
                    //std::cout << norm.rows << ":" << y << "," << x+z << ":" << norm.cols << std::endl;
                    snr += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y, x+z))));
					snl += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y, x-z))));
					snt += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y+z, x))));
					snb += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y-z, x))));
					sntr += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y+z, x+z))));
					sntl += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y+z, x-z))));
					snbr += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y-z, x+z))));
					snbl += flipAngle(scalar(norm.at<Point4f>(y, x),(norm.at<Point4f>(y-z, x-z))));
				}
				snr /= neighborhoodRange;
				snl /= neighborhoodRange;
				snt /= neighborhoodRange;
				snb /= neighborhoodRange;
				sntr /= neighborhoodRange;
				sntl /= neighborhoodRange;
				snbr /= neighborhoodRange;
				snbl /= neighborhoodRange;

				if (neighborhoodMode == 0) {//max
					angleI.at<float>(y, x) = maxAngle(snr, snl, snt, snb,
                                            snbl, snbr, sntl, sntr);
				} else if (neighborhoodMode == 1) {//mean
					angleI.at<float>(y, x) = (snr + snl + snt + snb
							+ sntr + sntl + snbr + snbl) / 8;
				} else {
				}
			}
		}
	}
}

void applyLinearNormalAveraging(Mat &normals, Mat &avgNormals, int normalAveragingRange, int height, int width){
    const int r = normalAveragingRange;
    
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (y < r || y >= height - r || x < r
					|| x >= width - r) {
				avgNormals.at<Point4f>(y, x) = normals.at<Point4f>(y, x);
			} else {
				Point4f avg;
				avg.x = 0, avg.y = 0, avg.z = 0, avg.w = 0;
				for (int sx = -r; sx <= r; sx++) {
					for (int sy = -r; sy <= r; sy++) {
						avg.x += normals.at<Point4f>((x + sx), (y + sy)).x;
                        avg.y += normals.at<Point4f>((x + sx), (y + sy)).y;
						avg.z += normals.at<Point4f>((x + sx), (y + sy)).z;
					}
				}
				avg.x /= ((1 + 2 * r) * (1 + 2 * r));
				avg.y /= ((1 + 2 * r) * (1 + 2 * r));
				avg.z /= ((1 + 2 * r) * (1 + 2 * r));
				avg.w = 1;
				avgNormals.at<Point4f>(y, x) = avg;
			}
		}
	}
}

void applyNormalCalculation(Mat &filteredImage, Mat &normals, int normalRange, int normalAveragingRange, bool gauss, bool average, int height, int width)  {
    int r = normalRange;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            //int i = x + width * y;
            //std::cout << y << "," << x << std::endl;
            //std::cout << filteredImage.at<float>(532+r, 2-r) << std::endl;
            //std::cout << filteredImage.at<float>(532-r, 2-r) << std::endl;
            if (y < normalRange || y >= height- normalRange || x < normalRange || x >= width - normalRange){
                Point4f* p = normals.ptr<Point4f>(y, x);
                p->x = 0;
                p->y = 0;
                p->z = 0;
            } else {
                // cross product normal determination
                
                float fa1[3], fb1[3], n1[3], n01[3];
                
                fa1[0] = (x + r) - (x - r);
				fa1[1] = (y - r) - (y - r);
				//fa1[2] = filteredImage.at<float>(x + r, y - r)
				//		- filteredImage.at<float>(x - r, y - r);
                fa1[2] = filteredImage.at<float>(y - r, x + r)
						- filteredImage.at<float>(y - r, x - r);
                
				fb1[0] = (x) - (x - r);
				fb1[1] = (y + r) - (y - r);
				//fb1[2] = filteredImage.at<float>(x, y + r)
				//		- filteredImage.at<float>(x - r, y - r);
                fb1[2] = filteredImage.at<float>(y + r, x)
						- filteredImage.at<float>(y - r, x - r);
                
				n1[0] = fa1[1] * fb1[2] - fa1[2] * fb1[1];
				n1[1] = fa1[2] * fb1[0] - fa1[0] * fb1[2];
				n1[2] = fa1[0] * fb1[1] - fa1[1] * fb1[0];
                
                float norm = sqrt(pow(n1[0],2)+ pow(n1[1],2) + pow(n1[2],2));
				n01[0] = n1[0] / norm;
				n01[1] = n1[1] / norm;
				n01[2] = n1[2] / norm;
                
                normals.ptr<Point4f>(y, x)->x = n01[0];
			    normals.ptr<Point4f>(y, x)->y = n01[1];
				normals.ptr<Point4f>(y, x)->z = n01[2];
				normals.ptr<Point4f>(y, x)->w = 1;
            }
        }
    }
    if (average && !gauss) {
        applyLinearNormalAveraging(normals, normals, normalAveragingRange, height, width);
    } else if (average && gauss) {
        GaussianBlur(normals, normals, Size(normalAveragingRange,normalAveragingRange), 0);
    }
}

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

Mat ICLImg_to_Mat(const icl::core::Img32f &image, int numChannel){
    int h = image.getHeight();
    int w = image.getWidth();
    Mat mat = Mat(h, w, CV_MAKE_TYPE(CV_32F, numChannel));
    for(int i=0; i < h; i++){
        for(int j=0; j<w; j++){
            for(int k=0; k<numChannel; k++) {
                mat.at<float>(i, j, 0) = image(i, j,0);
            }
        }
      }
    return mat;
}

Mat ICLImg_to_Mat(const icl::core::Img8u &image, int numChannel){
    int h = image.getHeight();
    int w = image.getWidth();
    Mat mat = Mat(h, w, CV_MAKE_TYPE(CV_8U, numChannel));
    for(int i=0; i < h; i++){
        for(int j=0; j<w; j++){
            for(int k=0; k<numChannel; k++) {
                mat.at<int8_t>(i, j, 0) = image(i, j,0);
            }
        }
      }
    return mat;
}

void Mat_to_ICLImg(icl::core::Img32f &image, Mat mat, int numChannel){
    int h = image.getHeight();
    int w = image.getWidth();
    for(int i=0; i < h; i++){
        for(int j=0; j<w; j++){
            for(int k = 0; k < numChannel; k++){
                image(i, j, 0) = mat.at<float>(i, j);
            }
        }
      }
}

void Mat_to_ICLImg(icl::core::Img8u &image, Mat mat, int numChannel){
    int h = image.getHeight();
    int w = image.getWidth();
    for(int i=0; i < h; i++){
        for(int j=0; j<w; j++){
            for(int k = 0; k < numChannel; k++){
                image(i, j, 0) = mat.at<int8_t>(i, j);
            }
        }
      }
}

Mat preSeg_calculate(Mat &depthImage, Data &data){
    std::vector<int> size = {data.height, data.width, 3};
    /*
    Mat normals = Mat(height, width, CV_32FC4);
    Mat angleImage = Mat(height, width, CV_32FC1);
    Mat binarizedImage = Mat(height, width, CV_32FC1);
    Mat filteredImage = Mat::zeros(height, width, CV_32FC1);
    */
    if(data.usedFilterFlag){
        medianBlur(depthImage, data.filteredImage, data.medianFilterSize);
    }

    applyNormalCalculation(data);
    applyAngleImageCalculation(data);
	//applyImageBinarization(binarizedImage, angleImage, height, width, binarizationThreshold);
    cv::threshold(data.angleImage, data.binarizedImage, data.binarizationThreshold, 255, CV_THRESH_BINARY);
    cv::rotate(data.normals, data.normals, cv::ROTATE_90_COUNTERCLOCKWISE);
    imshow("normal", data.normals);
    cv::waitKey(1);
    data.binarizedImage.convertTo(data.binarizedImage, CV_8UC1);
    
    return data.binarizedImage;
}

Mat preSeg_calculate(Mat &depthImage, bool filter, bool average, bool gauss){
    
    int medianFilterSize = 3;
    int normalRange = 2;
    int normalAveragingRange = 5;
	int neighborhoodMode = 0;
	int neighborhoodRange = 3;
	double binarizationThreshold = 0.89;
	int useNormalAveraging = true;
	int useGaussSmoothing = false;
    

    int height = depthImage.rows;
    int width = depthImage.cols;
    std::vector<int> size = {height, width, 3};

    Mat normals = Mat(height, width, CV_32FC4);
    Mat angleImage = Mat(height, width, CV_32FC1);
    Mat binarizedImage = Mat(height, width, CV_32FC1);
    Mat filteredImage = Mat::zeros(height, width, CV_32FC1);

    if(filter){
        medianBlur(depthImage, filteredImage, medianFilterSize);
    }

    applyNormalCalculation(filteredImage, normals, normalRange, normalAveragingRange, gauss, average, height, width);
    cv::rotate(normals, normals, cv::ROTATE_90_COUNTERCLOCKWISE);
    imshow("normal", normals);
    applyAngleImageCalculation(angleImage, normals, height, width, neighborhoodRange, neighborhoodMode);
	//applyImageBinarization(binarizedImage, angleImage, height, width, binarizationThreshold);
    cv::threshold(angleImage, binarizedImage, binarizationThreshold, 255, CV_THRESH_BINARY);

    
    cv::waitKey(1);
    binarizedImage.convertTo(binarizedImage, CV_8UC1);
    
    return binarizedImage;
}


int main(int argc, char *argv[]) {
}

}