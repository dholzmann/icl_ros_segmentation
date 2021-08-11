#include "HighLevelObjectSegmentation.h"

using namespace cv;

Mat ObjectSegmenter::apply(Data &data, Mat xyz, const Mat &edgeImage, Mat &depthImg, Mat normals, bool stabilize, bool useROI, bool useCutfreeAdjacency, 
    bool useCoplanarity, bool useCurvature, bool useRemainingPoints){
        ObjectSegmenter::surfaceSegmentation(data, xyz, edgeImage, depthImg, data.minSurfaceSize, useROI);

        if(data.surfaceSize>0){
          data.features=SurfaceFeatureExtract::apply(data.labelImage, xyz, normals, SurfaceFeatureExtract::ALL);
    
          Mat initialMatrix = data.segUtils->edgePointAssignmentAndAdjacencyMatrix(xyz, data.labelImage, 
                                  data.maskImage, data.assignmentRadius, data.assignmentDistance, data.surfaces.size());
          
          Mat resultMatrix(data.surfaces.size(), data.surfaces.size(), false);
          
          if(useCutfreeAdjacency){
            Mat cutfreeMatrix = data.cutfree->apply(xyz, 
                      data.surfaces, initialMatrix, data.cutfreeRansacEuclideanDistance, 
              data.cutfreeRansacPasses, data.cutfreeRansacTolerance, data.labelImage, data.features, data.cutfreeMinAngle);
            //math::GraphCutter::mergeMatrix(resultMatrix, cutfreeMatrix);
          }
        }
    return Mat();
}

void ObjectSegmenter::surfaceSegmentation(Data &data, Mat &xyz, const Mat &edgeImg, Mat &depthImg, int minSurfaceSize, bool useROI){
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

    int w = edgeImg.size().width;
    
    //mask edge image
    Mat edgeImgMasked(edgeImg.size(), CV_8UC1);
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
    int num_labels = connectedComponents(edgeImgMasked, data.labelImage, 4, CV_32S);
}
