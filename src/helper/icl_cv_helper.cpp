#include "icl_cv_helper.h"

Mat ICLImg_to_Mat(icl::core::Img32f &image, Mat &mat, int numChannel){
    int h = image.getHeight();
    int w = image.getWidth();
	//std::cout << h << ", " << w << std::endl;
    //Mat mat = Mat(w, h, CV_MAKE_TYPE(CV_32F, numChannel));
    for(int i=0; i < h; i++){
        for(int j=0; j<w; j++){
            for(int k=0; k<numChannel; k++) {
                //mat.at<float>(j, i, k) = image(i, j, k);
				mat.at<float>(i, j, k) = image[k][j+w*i];
            }
        }
      }
    return mat;
}

Mat ICLImg_to_Mat(icl::core::Img8u &image, Mat &mat, int numChannel){
    int h = image.getHeight();
    int w = image.getWidth();
    //Mat mat = Mat(h, w, CV_MAKE_TYPE(CV_8U, numChannel));
    for(int i=0; i < h; i++){
        for(int j=0; j<w; j++){
            for(int k=0; k<numChannel; k++) {
                //mat.at<int>(i, j, k) = image(i, j,k);
				mat.at<int>(i, j, k) = image[k][j+w*i];
            }
        }
      }
    return mat;
}

void Mat_to_ICLImg(icl::core::Img32f &image, Mat &mat, int numChannel){
    int h = image.getHeight();
    int w = image.getWidth();
    for(int i=0; i < h; i++){
        for(int j=0; j<w; j++){
            for(int k = 0; k < numChannel; k++){
                //image(i, j, k) = mat.at<float>(i, j, k);
				image[k][j+w*i] = mat.at<float>(i, j, k);
            }
        }
      }
}

void Mat_to_ICLImg(icl::core::Img8u &image, Mat &mat, int numChannel){
    int h = image.getHeight();
    int w = image.getWidth();
    for(int i=0; i < h; i++){
        for(int j=0; j<w; j++){
            for(int k = 0; k < numChannel; k++){
                //image(i, j, k) = mat.at<int>(i, j, k);
				image[k][j+w*i] = mat.at<int>(i, j, k);
            }
        }
      }
}

void Mat_to_DataSegment(icl::core::DataSegment<float, 4> &image, Mat &mat, int h, int w){
	for(int y=0; y<h; y++){
		for(int x=0; x<w; x++){
			int i = x + w * y;
			Vec4f p = mat.at<Vec4f>(y, x);
			image[i].x = p[0];
			image[i].y = p[1];
			image[i].z = p[2];
			image[i].w = p[3];
		}
	}
}

void DataSegment_to_Mat(icl::core::DataSegment<float, 4> &image, Mat &mat, int h, int w){
	for(int y=0; y<h; y++){
		for(int x=0; x<w; x++){
			int i = x + w * y;
			Vec4f p = mat.at<Vec4f>(y, x);
			p[0] = image[i].x;
			p[1] = image[i].y;
			p[2] = image[i].z;
			p[3] = image[i].w;
		}
	}
}
