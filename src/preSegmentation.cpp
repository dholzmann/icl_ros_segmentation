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
				worldNormals.at<Vec4f>(y, x)[0] = -pWN.at<float>(0);
				worldNormals.at<Vec4f>(y, x)[1] = -pWN.at<float>(1);
				worldNormals.at<Vec4f>(y, x)[2] = -pWN.at<float>(2);
				worldNormals.at<Vec4f>(y, x)[3] = 1.;

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

float scalar(Vec4f &a, Vec4f &b){
    return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
}

void applyNormalCalculation(Data &data) {
    int r = data.normalRange;
    for (int y = 0; y < data.height; y++) {
        for (int x = 0; x < data.width; x++) {
            //int i = x + width * y;
            //std::cout << y << "," << x << std::endl;
            if (y < r || y >= data.height- r || x < r || x >= data.width - r){
                Vec4f p = data.normals.at<Vec4f>(y, x);
                p[0] = 0;
                p[1] = 0;
                p[2] = 0;
				p[3] = 0;
            } else {
                // cross product normal determination
                
                float fa1[3], fb1[3], n1[3], n01[3];
                
                fa1[0] = (x + r) - (x - r);
				fa1[1] = (y - r) - (y - r);
                fa1[2] = data.filteredImage.at<float>(y - r, x + r)
						- data.filteredImage.at<float>(y - r, x - r);
				fb1[0] = (x) - (x - r);
				fb1[1] = (y + r) - (y - r);
                fb1[2] = data.filteredImage.at<float>(y + r, x)
						- data.filteredImage.at<float>(y - r, x - r);
                
				n1[0] = fa1[1] * fb1[2] - fa1[2] * fb1[1];
				n1[1] = fa1[2] * fb1[0] - fa1[0] * fb1[2];
				n1[2] = fa1[0] * fb1[1] - fa1[1] * fb1[0];
                
                float norm = sqrt(pow(n1[0],2)+ pow(n1[1],2) + pow(n1[2],2));
				n01[0] = n1[0] / norm;
				n01[1] = n1[1] / norm;
				n01[2] = n1[2] / norm;
                
                data.normals.at<Vec4f>(y, x)[0] = n01[0];
			    data.normals.at<Vec4f>(y, x)[1] = n01[1];
				data.normals.at<Vec4f>(y, x)[2] = n01[2];
				data.normals.at<Vec4f>(y, x)[3] = 1;
            }
        }
    }
	//std::cout << data.normalAveragingRange << std::endl;
	GaussianBlur(data.normals, data.avgNormals, Size(data.normalAveragingRange,data.normalAveragingRange), 0, 0, BORDER_ISOLATED);
	//applyGaussianNormalSmoothing(data);
	/*
    if (data.useNormalAveraging && !data.useGaussSmoothing) {
        applyLinearNormalAveraging(data);
    } else if (data.useNormalAveraging && data.useGaussSmoothing) {
        GaussianBlur(data.normals, data.avgNormals, Size(data.normalAveragingRange,data.normalAveragingRange), 0, 0, BORDER_ISOLATED);
		//applyGaussianNormalSmoothing(data);
    }*/
}

void applyLinearNormalAveraging(Data &data){
    int r = data.normalAveragingRange;
    
	for (int y = 0; y < data.height; y++) {
		for (int x = 0; x < data.width; x++) {
			if (y < r || y >= data.height - r || x < r
					|| x >= data.width - r) {
				data.avgNormals.at<Vec4f>(y, x) = data.normals.at<Vec4f>(y, x);
			} else {
				Vec4f avg(0,0,0,0);
				for (int sx = -r; sx <= r; sx++) {
					for (int sy = -r; sy <= r; sy++) {
						//avg[0] += data.normals.at<Vec4f>((y + sy), (x + sx))[0];
                        //avg[1] += data.normals.at<Vec4f>((y + sy), (x + sx))[1];
						//avg[2] += data.normals.at<Vec4f>((y + sy), (x + sx))[2];
						avg += data.normals.at<Vec4f>((y + sy), (x + sx));
					}
				}
				avg /= ((1 + 2 * r) * (1 + 2 * r));
				avg[3] = 1;
				data.avgNormals.at<Vec4f>(y, x) = avg;
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
	int w = data.width, h = data.height;
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
                    snr += flipAngle(scalar(norm.at<Vec4f>(y, x),(norm.at<Vec4f>(y, x+z))));
					snl += flipAngle(scalar(norm.at<Vec4f>(y, x),(norm.at<Vec4f>(y, x-z))));
					snt += flipAngle(scalar(norm.at<Vec4f>(y, x),(norm.at<Vec4f>(y+z, x))));
					snb += flipAngle(scalar(norm.at<Vec4f>(y, x),(norm.at<Vec4f>(y-z, x))));
					sntr += flipAngle(scalar(norm.at<Vec4f>(y, x),(norm.at<Vec4f>(y+z, x+z))));
					sntl += flipAngle(scalar(norm.at<Vec4f>(y, x),(norm.at<Vec4f>(y+z, x-z))));
					snbr += flipAngle(scalar(norm.at<Vec4f>(y, x),(norm.at<Vec4f>(y-z, x+z))));
					snbl += flipAngle(scalar(norm.at<Vec4f>(y, x),(norm.at<Vec4f>(y-z, x-z))));
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

void applyGaussianNormalSmoothing(Data &data) {
	Mat kernel = data.kernel;
	float norm = data.kNorm;
	int l = data.kL;
	std::cout << kernel << std::endl;
	Mat res;
	GaussianBlur(data.normals, res, Size(5,5),-1,0.0,BORDER_ISOLATED);
	imshow("gaus", res);
	cv::waitKey(1);
    for (int y = 0; y < data.height; y++) {
	    for (int x = 0; x < data.width; x++) {
		    int i = x + data.width * y;
		    if (y < l || y >= data.height - l || x < l || x >= data.width - l
				    || l == 0) {
			    data.avgNormals.at<Vec4f>(y, x) = data.normals.at<Vec4f>(y, x);
		    } else {
			    Vec4f avg(0,0,0,0);
			    for (int sx = -l; sx <= l; sx++) {
				    for (int sy = -l; sy <= l; sy++) {
					    avg[0] += data.normals.at<Vec4f>(y + sy, x + sx)[0]
							    * kernel.at<float>(sx + l, sy + l);
					    avg[1] += data.normals.at<Vec4f>(y + sy, x + sx)[1]
							    * kernel.at<float>(sx + l, sy + l);
					    avg[2] += data.normals.at<Vec4f>(y + sy, x + sx)[2]
							    * kernel.at<float>(sx + l, sy + l);
				    }
			    }
			    //avg.x /= norm;
			    //avg.y /= norm;
			    //avg.z /= norm;
				avg /= norm;
			    avg[3] = 1;
			    data.avgNormals.at<Vec4f>(y,x) = avg;
		    }
	    }
    }
}

void preSeg_calculate(Mat &depthImage, Data &data){
    if (data.usedFilterFlag == false) {
		data.filteredImage = depthImage;
	} else {
        medianBlur(depthImage, data.filteredImage, data.medianFilterSize);
		data.rawImage = depthImage;
	}
    applyNormalCalculation(data);
    applyAngleImageCalculation(data);
    cv::threshold(data.angleImage, data.binarizedImage, data.binarizationThreshold, 255, CV_THRESH_BINARY);  
    data.binarizedImage.convertTo(data.binarizedImage, CV_8UC1);
}

//int main(int argc, char *argv[]) {
//}
}