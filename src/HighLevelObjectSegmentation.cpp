#include "HighLevelObjectSegmentation.h"

using namespace cv;

cv::Mat ObjectSegmenter::apply(ObjectSegmenter::Data &data, cv::Mat xyz, const Mat &edgeImage, Mat &depthImg, Mat normals, bool stabilize, bool useROI, bool useCutfreeAdjacency, 
    bool useCoplanarity, bool useCurvature, bool useRemainingPoints){
        ObjectSegmenter::surfaceSegmentation(data, xyz, edgeImage, depthImg, data.minSurfaceSize, useROI);
        Mat cutfreeMatrix;
        Mat coplanMatrix; 
        Mat curveMatrix;
        Mat remainingMatrix;
        if(data.surfaceSize>0){
          data.features=SurfaceFeatureExtract::apply(data.labelImage, xyz, normals, SurfaceFeatureExtract::ALL);
    
          Mat initialMatrix = data.segUtils->edgePointAssignmentAndAdjacencyMatrix(xyz, data.labelImage, 
                                  data.maskImage, data.assignmentRadius, data.assignmentDistance, data.surfaces.size());
          
          Mat resultMatrix(data.surfaces.size(), data.surfaces.size(), false);
          
          if(useCutfreeAdjacency){
            cutfreeMatrix = data.cutfree->apply(xyz, 
                      data.surfaces, initialMatrix, data.cutfreeRansacEuclideanDistance, 
              data.cutfreeRansacPasses, data.cutfreeRansacTolerance, data.labelImage, data.features, data.cutfreeMinAngle);
            GraphCut::mergeMatrix(resultMatrix, cutfreeMatrix);
          }
          if(useCoplanarity){
		        coplanMatrix = CoPlanarityFeature::apply(initialMatrix, data.features, depthImg, data.surfaces, data.coplanarityMaxAngle,
                            data.coplanarityDistanceTolerance, data.coplanarityOutlierTolerance, data.coplanarityNumTriangles, data.coplanarityNumScanlines);
      	    GraphCut::mergeMatrix(resultMatrix, coplanMatrix);
          }

          if(useCurvature){
		        curveMatrix = CurvatureFeature::apply(depthImg, xyz, initialMatrix, data.features, data.surfaces, normals,
                                              data.curvatureUseOpenObjects, data.curvatureUseOccludedObjects, data.curvatureHistogramSimilarity, 
                                              data.curvatureMaxDistance, data.curvatureMaxError, data.curvatureRansacPasses, data.curvatureDistanceTolerance, 
                                              data.curvatureOutlierTolerance);
            GraphCut::mergeMatrix(resultMatrix, curveMatrix);
          }
          data.surfaces = data.segUtils->createLabelVectors(data.labelImage);
	      
	        data.segments = GraphCut::thresholdCut(resultMatrix, data.graphCutThreshold);  
        }	    
	      if(useRemainingPoints){
	        remainingMatrix = RemainingPointsFeature::apply(xyz, depthImg, data.labelImage, data.maskImage, 
                            data.surfaces, data.remainingMinSize, data.remainingEuclideanDistance, data.remainingRadius, data.remainingAssignEuclideanDistance);
	      }
	      
	      data.segUtils->relabel(data.labelImage, data.segments, data.surfaces.size());
      
	      return getColoredLabelImage(data, stabilize);
	    }   

cv::Mat ObjectSegmenter::applyHierarchy(ObjectSegmenter::Data &data, cv::Mat xyz, const Mat &edgeImage, Mat &depthImg, Mat normals, bool stabilize, bool useROI, bool useCutfreeAdjacency, 
    bool useCoplanarity, bool useCurvature, bool useRemainingPoints,
    float weightCutfreeAdjacency, float weightCoplanarity, float weightCurvature, float weightRemainingPoints){
        ObjectSegmenter::surfaceSegmentation(data, xyz, edgeImage, depthImg, data.minSurfaceSize, useROI);
        Mat cutfreeMatrix;
        Mat coplanMatrix; 
        Mat curveMatrix;
        Mat remainingMatrix;
        Mat probabilityMatrix;
        if(data.surfaceSize>0){
          data.features=SurfaceFeatureExtract::apply(data.labelImage, xyz, normals, SurfaceFeatureExtract::ALL);
    
          Mat initialMatrix = data.segUtils->edgePointAssignmentAndAdjacencyMatrix(xyz, data.labelImage, 
                                  data.maskImage, data.assignmentRadius, data.assignmentDistance, data.surfaces.size());
          
          Mat resultMatrix(data.surfaces.size(), data.surfaces.size(), false);
          
          if(useCutfreeAdjacency){
            cutfreeMatrix = data.cutfree->apply(xyz, 
                      data.surfaces, initialMatrix, data.cutfreeRansacEuclideanDistance, 
              data.cutfreeRansacPasses, data.cutfreeRansacTolerance, data.labelImage, data.features, data.cutfreeMinAngle);
            GraphCut::mergeMatrix(resultMatrix, cutfreeMatrix);
          }
          if(useCoplanarity){
		        coplanMatrix = CoPlanarityFeature::apply(initialMatrix, data.features, depthImg, data.surfaces, data.coplanarityMaxAngle,
                            data.coplanarityDistanceTolerance, data.coplanarityOutlierTolerance, data.coplanarityNumTriangles, data.coplanarityNumScanlines);
      	    GraphCut::mergeMatrix(resultMatrix, coplanMatrix);
          }

          if(useCurvature){
		        curveMatrix = CurvatureFeature::apply(depthImg, xyz, initialMatrix, data.features, data.surfaces, normals,
                                              data.curvatureUseOpenObjects, data.curvatureUseOccludedObjects, data.curvatureHistogramSimilarity, 
                                              data.curvatureMaxDistance, data.curvatureMaxError, data.curvatureRansacPasses, data.curvatureDistanceTolerance, 
                                              data.curvatureOutlierTolerance);
            GraphCut::mergeMatrix(resultMatrix, curveMatrix);
        }	    
	      if(useRemainingPoints){
	        remainingMatrix = RemainingPointsFeature::apply(xyz, depthImg, data.labelImage, data.maskImage, 
                            data.surfaces, data.remainingMinSize, data.remainingEuclideanDistance, data.remainingRadius, data.remainingAssignEuclideanDistance);
                            
          GraphCut::mergeMatrix(resultMatrix, remainingMatrix);
	      }
	      
	      probabilityMatrix = GraphCut::calculateProbabilityMatrix(resultMatrix, true);
	      
	      GraphCut::weightMatrix(probabilityMatrix, cutfreeMatrix, weightCutfreeAdjacency);
	      GraphCut::weightMatrix(probabilityMatrix, coplanMatrix, weightCoplanarity);
	      GraphCut::weightMatrix(probabilityMatrix, curveMatrix, weightCurvature);
	      GraphCut::weightMatrix(probabilityMatrix, remainingMatrix, weightRemainingPoints);
	    
	    }
	      
	    data.segments.clear();
	    
	    //return createHierarchy(probabilityMatrix, xyz, rgb);
      return probabilityMatrix;
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


void ObjectSegmenter::setROI(Data &data, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax) {
  data.xMinROI = xMin;
  data.xMaxROI = xMax;
  data.yMinROI = yMin;
  data.yMaxROI = yMax;
  data.zMinROI = zMin;
  data.zMaxROI = zMax;
}


void ObjectSegmenter::setCornerRemoval (Data &data, int ll, int lr, int ul, int ur){
  data.llCorner=ll;
  data.lrCorner=lr;
  data.ulCorner=ul;
  data.urCorner=ur;
}


void ObjectSegmenter::setMinSurfaceSize(Data &data, unsigned int size) {
  data.minSurfaceSize = size;
}


void ObjectSegmenter::setAssignmentParams(Data &data, float distance, int radius){
  data.assignmentRadius=radius;
  data.assignmentDistance=distance;
}


void ObjectSegmenter::setCutfreeParams(Data &data, float euclideanDistance, int passes, int tolerance, float minAngle){
  data.cutfreeRansacEuclideanDistance=euclideanDistance;
  data.cutfreeRansacPasses=passes;
  data.cutfreeRansacTolerance=tolerance;
  data.cutfreeMinAngle=minAngle;
}


void ObjectSegmenter::setCoplanarityParams(Data &data, float maxAngle, float distanceTolerance, float outlierTolerance, int numTriangles, int numScanlines){
  data.coplanarityMaxAngle=maxAngle;
  data.coplanarityDistanceTolerance=distanceTolerance;
  data.coplanarityOutlierTolerance=outlierTolerance;
  data.coplanarityNumTriangles=numTriangles;
  data.coplanarityNumScanlines=numScanlines;
}
  
  
void ObjectSegmenter::setCurvatureParams(Data &data, float histogramSimilarity, bool useOpenObjects, int maxDistance, bool useOccludedObjects, float maxError, 
                        int ransacPasses, float distanceTolerance, float outlierTolerance){
  data.curvatureHistogramSimilarity=histogramSimilarity;
  data.curvatureUseOpenObjects=useOpenObjects;
  data.curvatureMaxDistance=maxDistance;
  data.curvatureUseOccludedObjects=useOccludedObjects;
  data.curvatureMaxError=maxError;
  data.curvatureRansacPasses=ransacPasses;
  data.curvatureDistanceTolerance=distanceTolerance;
  data.curvatureOutlierTolerance=outlierTolerance;                        
}


void ObjectSegmenter::setRemainingPointsParams(Data &data, int minSize, float euclideanDistance, int radius, float assignEuclideanDistance, int supportTolerance){
  data.remainingMinSize=minSize;
  data.remainingEuclideanDistance=euclideanDistance;
  data.remainingRadius=radius;
  data.remainingAssignEuclideanDistance=assignEuclideanDistance;
  data.remainingSupportTolerance=supportTolerance;
}


void ObjectSegmenter::setGraphCutThreshold(Data &data, float threshold){
  data.graphCutThreshold=threshold;
}

std::vector<SurfaceFeatureExtract::SurfaceRegionFeature> ObjectSegmenter::getSurfaceFeatures(Data &data) {
return data.features;
}

std::vector<std::vector<int> > ObjectSegmenter::getSegments(Data &data) {
  return data.segments;
}


std::vector<std::vector<int> > ObjectSegmenter::getSurfaces(Data &data) {
  return data.surfaces;
}


Mat ObjectSegmenter::getLabelImage(Data &data, bool stabelize) {
  if(stabelize){
    return data.segUtils->stabelizeSegmentation(data.labelImage);
  }
  return data.labelImage;
}


Mat ObjectSegmenter::getColoredLabelImage(Data &data, bool stabelize) {
  Mat lI=getLabelImage(data, stabelize);
  return data.segUtils->createColorImage(lI);
}


Mat ObjectSegmenter::getMaskImage(Data &data){
  return data.maskImage;
}

/*
std::vector<PointCloudSegmentPtr> ObjectSegmenter::createHierarchy(Mat &probabilityMatrix, core::DataSegment<float,4> &xyz, core::DataSegment<float,4> &rgb){
  std::vector<GraphCut::CutNode> cutNodes = GraphCut::hierarchicalCut(probabilityMatrix);
  std::vector<PointCloudSegmentPtr> results;

  //create point cloud segments (incl. leafs with data)
  std::vector<PointCloudSegmentPtr> pointCloudSegments(cutNodes.size());
  for(unsigned int i=0; i<cutNodes.size(); i++){//create segments
    if(cutNodes[i].children.size()==0){//leaf (no children)
      pointCloudSegments[i]=new PointCloudSegment(data.surfaces[cutNodes[i].subset[0]].size(), true);
      core::DataSegment<float,4> dst_xyzh=pointCloudSegments[i]->selectXYZH();
      core::DataSegment<float,4> dst_rgba=pointCloudSegments[i]->selectRGBA32f();
      for(unsigned int j=0; j<data.surfaces[cutNodes[i].subset[0]].size(); j++){
        dst_xyzh[j]=xyz[data.surfaces[cutNodes[i].subset[0]][j]];
        dst_rgba[j]=rgb[data.surfaces[cutNodes[i].subset[0]][j]];
      }
      pointCloudSegments[i]->cutCost=0;//cost 0
    }
    else{//parent node
      pointCloudSegments[i]=new PointCloudSegment(0,false);
      pointCloudSegments[i]->cutCost=cutNodes[i].cost;//cost x
    }
  }

  //create hierarchy
  for(unsigned int i=0; i<cutNodes.size(); i++){//create segments
    if(cutNodes[i].children.size()>0){
      for(unsigned int j=0; j<cutNodes[i].children.size(); j++){
        pointCloudSegments[i]->addChild(pointCloudSegments[cutNodes[i].children[j]]);
      }
    }
  }

  //add root nodes to result
  for(unsigned int i=0; i<cutNodes.size(); i++){//create segments
    if(cutNodes[i].parent==-1){
      results.push_back(pointCloudSegments[i]);
      results.back()->updateFeatures();//new          
    }
  }
  
  return results; 
}*/
      

