#include <SurfaceFeatureExtractor.h>
SurfaceFeatureExtractor::SurfaceFeature SurfaceFeatureExtractor::apply(
                                        std::vector<Vec> &points, std::vector<Vec> &normals, int mode){
  SurfaceFeatureExtractor::SurfaceFeature feature = getInitializedStruct();
  if(normals.size()!=points.size()){
    throw utils::ICLException("points size != normals size");
  }
  feature.numPoints=points.size();
  for(unsigned int i=0; i<points.size(); i++){
    update(normals[i], points[i], feature, mode);
  }
  finish(feature, mode);
  return feature;
}


std::vector<SurfaceFeature> SurfaceFeatureExtractor::apply (Mat labelImage, Mat &xyzh, Mat &normals, int mode){
  unsigned int w = labelImage.size().width;
  unsigned int h = labelImage.size().height;

  std::vector<SurfaceFeatureExtractionSurfaceFeature> features;
  for(unsigned int y=0; y<h; y++){
    for(unsigned int x=0; x<w; x++){
      while((int)features.size() < labelImage.at<int>(x,y)){
        features.push_back(getInitializedFeature());
      }
      if(labelImage.at<int>(x,y)>0){
        features.at(labelImage.at<int>(x,y)-1).numPoints++;
        update(normals.at<Point4f>(x, y), xyzh.at<Point4f>(x, y), features.at(labelImage.at<int>(x,y)-1), mode, x, y);
      }
    }
  }
  for(unsigned int i=0; i<features.size(); i++){
    finish(features.at(i), mode);
  }
  return features;
}


void SurfaceFeatureExtractor::finish(SurfaceFeature &feature, int mode){
  feature.meanNormal/=feature.numPoints;
  feature.meanPosition/=feature.numPoints;
  if(mode&CURVATURE_FACTOR || mode&NORMAL_HISTOGRAM){
    int numOfBinsBigger05=0;
    for(unsigned int y = 0; y<11; y++){
      for(unsigned int x = 0; x<11; x++){
        feature.normalHistogram.at<float>(x,y)/=feature.numPoints;//normalized histogram
        if(feature.normalHistogram.at<float>(x,y)>=0.005) numOfBinsBigger05++;
      }
    }
    if(numOfBinsBigger05==0) feature.curvatureFactor = UNDEFINED;
    else if(numOfBinsBigger05<8) feature.curvatureFactor = PLANAR;
    else if(numOfBinsBigger05<20) feature.curvatureFactor = CURVED_1D;
    else feature.curvatureFactor = CURVED_2D;
  }
  if(mode&BOUNDING_BOX_3D){
    Point4f min = feature.boundingBox3D.first;
    Point4f max = feature.boundingBox3D.second;
    feature.volume = (max[0]-min[0])*(max[1]-min[1])*(max[2]-min[2]);
  }
}

void SurfaceFeatureExtractor::update(Point4f &normal, Point4f &point, SurfaceFeature &feature, int mode, int x, int y){
    if(mode&NORMAL_HISTOGRAM){
      int xx = round(normal.x*5.0+5.0);//-1 -> 0, 0 -> 5, 1 -> 10
      int yy = round(normal.y*5.0+5.0);
      feature.normalHistogram.at<float>(xx,yy)++;
    }
    if(mode&MEAN_NORMAL){
      feature.meanNormal+=normal;
    }
    if(mode&MEAN_POSITION){
      feature.meanPosition+=point;
    }
    if(mode&BOUNDING_BOX_3D){
      if(point[0]<feature.boundingBox3D.first[0]) feature.boundingBox3D.first.x=point[0];//min
      if(point[1]<feature.boundingBox3D.first[1]) feature.boundingBox3D.first.y=(point[1]);
      if(point[2]<feature.boundingBox3D.first[2]) feature.boundingBox3D.first.z=point[2];
      if(point[0]>feature.boundingBox3D.second[0]) feature.boundingBox3D.second.x=point[0];//max
      if(point[1]>feature.boundingBox3D.second[1]) feature.boundingBox3D.second.y=point[1];
      if(point[2]>feature.boundingBox3D.second[2]) feature.boundingBox3D.second.z=point[2];
    }
    if(mode&BOUNDING_BOX_2D){
      if(x<feature.boundingBox2D.first.x) feature.boundingBox2D.first.x=x;//min
      if(y<feature.boundingBox2D.first.y) feature.boundingBox2D.first.y=y;
      if(x>feature.boundingBox2D.second.x) feature.boundingBox2D.second.x=x;//max
      if(y>feature.boundingBox2D.second.y) feature.boundingBox2D.second.y=y;
    }
  }

  SurfaceFeature SurfaceFeatureExtractor::getInitializedFeature(){
    SurfaceFeature feature;
    feature.numPoints=0;
    //feature.normalHistogram.setSize(utils::Size(11,11));
    //feature.normalHistogram.setChannels(1);
    //feature.normalHistogram.setFormat(core::formatMatrix);
    //feature.normalHistogram.fill(0);
    feature.normalHistogram = Mat::zeros(Size(11,11), CV_32FC(6));
    //feature.normalHistogramChannel = feature.normalHistogram[0];
    feature.meanNormal=Point4f();
    feature.meanPosition=Point4f();
    //feature.curvatureFactor=SurfaceFeatureExtractor::UNDEFINED; == 0
    feature.curvatureFactor = UNDEFINED;
    //feature.boundingBox3D.first = Vec(1000000, 1000000, 1000000, 0);
    feature.boundingBox3D.first = Point4f(1000000, 1000000, 1000000, 0);
    feature.boundingBox3D.second = Point4f(-1000000, -1000000, -1000000, 0);
    feature.boundingBox2D.first = Point(1000000,1000000);
    feature.boundingBox2D.second = Point(-1000000, -1000000);
    feature.volume=0;
    return feature;
  }
}


float SurfaceFeatureExtractor::matchNormalHistograms(core::Img32f &a, core::Img32f &b){
  float sum=0;
  core::Channel32f aC = a[0];
  core::Channel32f bC = b[0];
  for(size_t i=0; i<11; i++){
    for(size_t j=0; j<11; j++){
      sum+=std::min(aC(i,j),bC(i,j));
    }
  }
  return sum;
}