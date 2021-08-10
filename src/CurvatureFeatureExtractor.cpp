#include "CurvatureFeatureExtractor.h"

Mat_<bool> CurvatureFeatureExtractor::apply(Mat &depthImg, Mat &xyz, Mat_<bool> &initialMatrix,
                      std::vector<SurfaceFeatureExtractor::SurfaceFeature> features,
                      std::vector<std::vector<int> > &surfaces, Mat &normals, bool useOpenObjects, bool useOccludedObjects,
                      float histogramSimilarity, int distance, float maxError, int ransacPasses, float distanceTolerance, float outlierTolerance){
  int w = depthImg.size().width;
  Mat_<bool> curvature = Mat_<bool>(initialMatrix.rows,initialMatrix.rows, true);//result matrix
  //initialize
  for(size_t i=0; i<initialMatrix.rows; i++){
    for(size_t j=0; j<initialMatrix.rows; j++){
      //only test pairs of non-adjacent curved surfaces
      if(initialMatrix(i,j)==true ||
          (features[i].curvatureFactor!=SurfaceFeatureExtractor::CURVED_1D && features[i].curvatureFactor!=SurfaceFeatureExtractor::CURVED_2D) ||
          (features[j].curvatureFactor!=SurfaceFeatureExtractor::CURVED_1D && features[j].curvatureFactor!=SurfaceFeatureExtractor::CURVED_2D) ){
        curvature(i,j)=false;
      }
    }
  }
  for(size_t i=0; i<curvature.rows; i++){
    for(size_t j=i+1; j<curvature.cols; j++){//dont check pairs twice
      if(curvature(i,j)==true){//candidate
        bool proceed=true;

        //joint criterion: similar surface shape and orientation (normal histogram matching)
        float similarityScore = SurfaceFeatureExtractor::matchNormalHistograms(features[i].normalHistogram, features[j].normalHistogram);
        if(similarityScore<histogramSimilarity){
          proceed=false;
        }

        //compute cases
        if(proceed){
          proceed=false;
          if(useOpenObjects){
            proceed = computeOpenObject(normals, features[i], features[j],
                                surfaces[i], surfaces[j], distance, w);
          }
          if(useOccludedObjects && !proceed){//only if first case does not match
            proceed = computeOccludedObject(depthImg, xyz, normals, features[i], features[j],
                                surfaces[i], surfaces[j], w, maxError, ransacPasses, distanceTolerance, outlierTolerance);
          }
        }

        if(!proceed){//remove if no case succeeded
          curvature(i,j)=false;
          curvature(j,i)=false;
        }
      }
    }
  }

  return curvature;
}


bool CurvatureFeatureExtractor::computeOpenObject(Mat &normals, SurfaceFeatureExtractor::SurfaceFeature feature1, SurfaceFeatureExtractor::SurfaceFeature feature2,
                                std::vector<int> &surface1, std::vector<int> &surface2, int distance, int w){
  //1. neighbouring in image space
  std::pair<Point,Point> bBox1 = feature1.boundingBox2D; //min, max
  std::pair<Point,Point> bBox2 = feature2.boundingBox2D;
  bool proceed=false;
  if(bBox1.second.x>bBox2.first.x && bBox2.second.x>bBox1.first.x){ //overlap in x
    proceed=true;
  }
  if(proceed){//maximum distance in y
    proceed=false;
    if(bBox1.first.y-bBox2.second.y<distance && bBox2.first.y-bBox1.second.y<distance){
      proceed=true;
    }
  }
  //2. one surface concave and one surface convex
  if(proceed){
    float direction1 = computeConvexity(normals, feature1, surface1, w);
    float direction2 = computeConvexity(normals, feature2, surface2, w);
    //>=0 convex (front), <0 concave (back)
    if((direction1>=0 && direction2<0 && bBox2.second.y>bBox1.second.y) || (direction1<0 && direction2>=0 && bBox1.second.y>bBox2.second.y)){
      return true;
    }
  }

  return false;
}


bool CurvatureFeatureExtractor::computeOccludedObject(Mat &depthImg, Mat &xyz, Mat &normals,
                                SurfaceFeatureExtractor::SurfaceFeature feature1, SurfaceFeatureExtractor::SurfaceFeature feature2,
                                std::vector<int> &surface1, std::vector<int> &surface2, int w, float maxError, int ransacPasses, float distanceTolerance, float outlierTolerance){
  //select most populated bin (same bin for both histograms)
  float maxBinValue=0;
  Point maxBin(0,0);
  for(int y=0; y<feature1.normalHistogram.size().height; y++){
    for(int x=0; x<feature1.normalHistogram.size().width; x++){
      float binValue = std::min(feature1.normalHistogram.at<float>(x,y), feature2.normalHistogram.at<float>(x,y));
      if(binValue>maxBinValue){
        maxBinValue=binValue;
        maxBin.x=x;
        maxBin.y=y;
      }
    }
  }

  //backproject the points
  std::vector<int> pointIDs1 = backprojectPointIDs(normals, maxBin, surface1);
  std::vector<int> pointIDs2 = backprojectPointIDs(normals, maxBin, surface2);
  std::vector<Vec3f> points1 = createPointsFromIDs(xyz, pointIDs1);
  std::vector<Vec3f> points2 = createPointsFromIDs(xyz, pointIDs2);

  //fit line with RANSAC (faster than linear regression)
  float minError = 100000;
  std::pair<Point,Point> pointPairImg;
  for(int i=0; i<ransacPasses; i++){
    float currentError=0;
    std::pair<Vec3f,Vec3f> currentPointPair;
    std::pair<int,int> currentPointPairID;
    currentPointPairID.first = pointIDs1[rand()%pointIDs1.size()];
    currentPointPairID.second = pointIDs2[rand()%pointIDs2.size()];
    currentPointPair.first = xyz.at<float>(currentPointPairID.first);
    currentPointPair.second = xyz.at<float>(currentPointPairID.second);
    for(unsigned int i=0; i<points1.size(); i++){
      currentError+=linePointDistance(currentPointPair, points1[i]);
    }
    for(unsigned int i=0; i<points2.size(); i++){
      currentError+=linePointDistance(currentPointPair, points2[i]);
    }
    currentError/=points1.size()+points2.size();
    if(currentError<minError){
      minError=currentError;
      pointPairImg.first=idToPoint(currentPointPairID.first,w);
      pointPairImg.second=idToPoint(currentPointPairID.second,w);
    }
  }

  //occlusion check
  if(minError<maxError){
    return SegmenterUtils::occlusionCheck(depthImg, pointPairImg.first, pointPairImg.second, distanceTolerance, outlierTolerance);
  }
  return false;
}


float CurvatureFeatureExtractor::computeConvexity(Mat &normals, SurfaceFeatureExtractor::SurfaceFeature feature, std::vector<int> &surface, int w){
  //select extremal bins in histogram
  std::pair<Point,Point> histoExtremalBins = computeExtremalBins(feature);

  //backproject to image space and calculate mean
  std::pair<Point,Point> imgBackproject = backproject(normals, histoExtremalBins, surface, w);

  //scalar product to determine concave and convex
  float direction = computeConvexity(histoExtremalBins, imgBackproject);

  return direction;
}


std::pair<Point,Point> CurvatureFeatureExtractor::computeExtremalBins(SurfaceFeatureExtractor::SurfaceFeature feature){
  //normal histogram bounding box
  std::pair<Point,Point> histoBBox;
  histoBBox.first.x=1000;
  histoBBox.first.y=1000;
  histoBBox.second.x=-1000;
  histoBBox.second.y=-1000;
  for(int y=0; y<feature.normalHistogram.size().height; y++){
    for(int x=0; x<feature.normalHistogram.size().width; x++){
      if(feature.normalHistogram.at<float>(x,y)>=0.005){
        if(x<histoBBox.first.x) histoBBox.first.x=x;
        if(y<histoBBox.first.y) histoBBox.first.y=y;
        if(x>histoBBox.second.x) histoBBox.second.x=x;
        if(y>histoBBox.second.y) histoBBox.second.y=y;
      }
    }
  }
  //normal histogram extremal bins
  std::pair<Point,Point> histoExtremalBins; //min, max
  if(histoBBox.second.x-histoBBox.first.x>=histoBBox.second.y-histoBBox.first.y){//sample x
    int x1 = histoBBox.first.x;
    histoExtremalBins.first.x=x1;
    int x2 = histoBBox.second.x;
    histoExtremalBins.second.x=x2;
    for(int y=0; y<feature.normalHistogram.size().height; y++){
      if(feature.normalHistogram.at<float>(x1,y)>=0.005){
        histoExtremalBins.first.y=y;
      }
      if(feature.normalHistogram.at<float>(x2,y)>=0.005){
        histoExtremalBins.second.y=y;
      }
    }
  }else{//sampleY
    int y1 = histoBBox.first.y;
    histoExtremalBins.first.y=y1;
    int y2 = histoBBox.second.y;
    histoExtremalBins.second.y=y2;
    for(int x=0; x<feature.normalHistogram.size().width; x++){
      if(feature.normalHistogram.at<float>(x,y1)>=0.005){
        histoExtremalBins.first.x=x;
      }
      if(feature.normalHistogram.at<float>(x,y2)>=0.005){
        histoExtremalBins.second.x=x;
      }
    }
  }
  return histoExtremalBins;
}


std::pair<Point,Point> CurvatureFeatureExtractor::backproject(Mat &normals,
                    std::pair<Point,Point> &histoExtremalBins, std::vector<int> &surface, int w){
  std::vector<int> imgMinPoints=backprojectPointIDs(normals, histoExtremalBins.first, surface);
  std::vector<int> imgMaxPoints=backprojectPointIDs(normals, histoExtremalBins.second, surface);

  std::pair<Point,Point> imgMeans;
  imgMeans.first = computeMean(imgMinPoints, w);
  imgMeans.second = computeMean(imgMaxPoints, w);
  return imgMeans;
}


std::vector<int> CurvatureFeatureExtractor::backprojectPointIDs(Mat &normals, Point bin, std::vector<int> &surface){
  std::vector<int> pointIDs;
  for(unsigned int i=0; i<surface.size(); i++){
    if(round(normals.at<Vec4f>(surface[i])[0]*5.0+5.0)==bin.x && round(normals.at<Vec4f>(surface[i])[1]*5.0+5.0)==bin.y){
      pointIDs.push_back(surface[i]);
    }
  }
  return pointIDs;
}


Point CurvatureFeatureExtractor::computeMean(std::vector<int> &imgIDs, int w){
  std::vector<Point> points = createPointsFromIDs(imgIDs, w);
  Point imgMean(0,0);

  for(unsigned int i=0; i<points.size(); i++){
    imgMean.x+=points[i].x;
    imgMean.y+=points[i].y;
  }
  imgMean.x/=points.size();
  imgMean.y/=points.size();
  return imgMean;
}


std::vector<Point> CurvatureFeatureExtractor::createPointsFromIDs(std::vector<int> &imgIDs, int w){
  std::vector<Point> points(imgIDs.size());
  for(unsigned int i=0; i<imgIDs.size(); i++){
    int id = imgIDs[i];
    int y = (int)floor((float)id/(float)w);
    int x = id-y*w;
    points[i]=Point(x,y);
  }
  return points;
}


std::vector<Vec3f> CurvatureFeatureExtractor::createPointsFromIDs(Mat &xyz, std::vector<int> &imgIDs){
  std::vector<Vec3f> points(imgIDs.size());
  for(unsigned int i=0; i<imgIDs.size(); i++){
    points[i]=xyz.at<Vec3f>(imgIDs[i]);
  }
  return points;
}


float CurvatureFeatureExtractor::computeConvexity(std::pair<Point,Point> histoExtremalBins, std::pair<Point,Point> imgBackproject){
  Point histoVec;
  Point imgVec;
  histoVec.x=histoExtremalBins.second.x-histoExtremalBins.first.x;
  histoVec.y=histoExtremalBins.second.y-histoExtremalBins.first.y;
  float lengthHisto = sqrt(histoVec.x*histoVec.x+histoVec.y*histoVec.y);//normalize
  imgVec.x=imgBackproject.second.x-imgBackproject.first.x;
  imgVec.y=imgBackproject.second.y-imgBackproject.first.y;
  float lengthImg = sqrt(imgVec.x*imgVec.x+imgVec.y*imgVec.y);
  float direction = (histoVec.x/lengthHisto)*(imgVec.x/lengthImg)+(histoVec.y/lengthHisto)*(imgVec.y/lengthImg);
  return direction;
}


float CurvatureFeatureExtractor::linePointDistance(std::pair<Vec3f,Vec3f> line, Vec3f point){
  float d = norm(Point3f(point-line.first).cross(Point3f(point-line.second)))/norm(line.second-line.first);
  return d;
}


Point CurvatureFeatureExtractor::idToPoint(int id, int w){
  int y = (int)floor((float)id/(float)w);
  int x = id-y*w;
  return Point(x,y);
}
