#include "HighLevelObjectSegmentation.h"

namespace ObjectSegmenter{
    using namespace cv;

    Mat apply(Data &data, Mat xyz, const Mat &edgeImage, const Mat &depthImg, Mat normals, bool stabilize, bool useROI, bool useCutfreeAdjacency, 
        bool useCoplanarity, bool useCurvature, bool useRemainingPoints){
            surfaceSegmentation(data, xyz, edgeImage, depthImg, data.minSurfaceSize, useROI);

            if(data.surfaceSize>0){
              Mat initialMatrix = m_data->segUtils->edgePointAssignmentAndAdjacencyMatrix(xyz, m_data->labelImage, 
                                m_data->maskImage, m_data->assignmentRadius, m_data->assignmentDistance, m_data->surfaces.size());
            }
    }

    void surfaceSegmentation(Data &data, Mat &xyz, const Mat &edgeImg, const Mat &depthImg, int minSurfaceSize, bool useROI){
        data.surfaces.clear();

        if(useROI){//create mask
            data.maskImage = data.segUtils->createROIMask(xyz, depthImg, data.xMinROI, 
                    data.xMaxROI, data.yMinROI, data.yMaxROI, data.zMinROI, data.zMaxROI);
        }else{
            data.maskImage = data.segUtils->createMask(depthImg);
        }

        for(int y=0; y<depthImg.size[0]; y++){
          for(int x=0; x<depthImg.size[1]; x++){
            if(depthImg.at<float>(x,y,0)==0){
              data.maskImage.at<float>(x,y,0) = 1;
            }
          }
        }

        for(int x=0; x<data.ulCorner; x++){
          for(int y=0; y<data.ulCorner-x; y++){
            data.maskImage.at<float>(x,y,0) = 1;
          }
        }
        
        for(int x=0; x<data.urCorner; x++){
          for(int y=0; y<data.urCorner-x; y++){
            data.maskImage.at<float>(data.maskImage.size().width-x-1, y, 0) = 1;
          }
        }
        for(int x=0; x<data.llCorner; x++){
          for(int y=0; y<data.llCorner-x; y++){
            data.maskImage.at<float>(x,data.maskImage.size().height-y-1,0)=1;
          }
        }
        for(int x=0; x<data.lrCorner; x++){
          for(int y=0; y<data.lrCorner-x; y++){
            data.maskImage.at<float>(data.maskImage.size().width-x-1,data.maskImage.size().height-y-1,0)=1;
          }
        }

        data.labelImage = Mat::zeros(edgeImg.size(),CV_32FC1);
        
        //core::Channel8u maskImageC = data.maskImage[0];
        //core::Channel32s labelImageC = data.labelImage[0];
        int w = edgeImg.size().width;
        
        //mask edge image
        Mat edgeImgMasked(edgeImg.size(), CV_8UC1);

        //core::Channel8u edgeImgMaskedC = edgeImgMasked[0];
        //core::Channel8u edgeImgC = edgeImg[0];
        Size size = edgeImg.size();
        for(int y=0; y<size.height; y++){
          for(int x=0; x<size.width; x++){
            if(data.maskImage.at<int>(x,y)==1){
              edgeImgMasked.at<int>(x,y)=0;
            }else{
              edgeImgMasked.at<int>(x,y)=edgeImg.at<int>(x,y);
            }
          }
        }
        /*
        int numCluster=0;      
        data.region->setConstraints (minSurfaceSize, 4000000, 254, 255);
        std::vector<ImageRegion> regions;
        regions = data.region->detect(&edgeImgMasked); 	
        for(unsigned int i=0; i<regions.size(); i++){
          std::vector<utils::Point> ps = regions[i].getPixels();
          if((int)ps.size()>=minSurfaceSize){
            numCluster++;
            std::vector<int> dat;
            for(unsigned int j=0; j<ps.size(); j++){
              int px = ps[j][0];
              int py = ps[j][1];
              int v=px+w*py;
              if(data.maskImage.at<int>(px,py)==0){
                data.labelImage.at<int>(px,py)=numCluster;
                data.maskImage.at<int>(px,py)=1;
                dat.push_back(v);
              }
            }
            data.surfaces.push_back(dat);
          }        
        }
        std::vector<std::vector<Point> > contours;
        std::vector<Vec4i> hierarchy;
        findContours(edgeImgMasked, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        data.surfaceSize = contours.size();
        for(int i = 0; i < contours.size(); i++){
          for(int j=0; j < contours[i].size(); j++){
            Point p = contours[i][j];
            data.labelImage.at<int>(p.y, p.x) = i;
          }
        }*/
        Mat labelImageC;
        int num_labels = connectedComponents(edgeImgMasked, labelImageC, 4, CV_32S);
    }
}