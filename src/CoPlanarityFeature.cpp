#include "CoPlanarityFeature.h"

Mat CoPlanarityFeature::apply(Mat &initialMatrix, std::vector<SurfaceFeatureExtract::SurfaceRegionFeature> features,
                      const Mat &depthImage, std::vector<std::vector<int> > &surfaces, float maxAngle,
                      float distanceTolerance, float outlierTolerance, int triangles, int scanlines){
  Mat coplanar = Mat(initialMatrix.size().height,initialMatrix.size().height, CV_8SC1);//result matrix
  //initialize
  for(size_t i=0; i<initialMatrix.size().height; i++){
    for(size_t j=0; j<initialMatrix.size().height; j++){
      //only test pairs of non-adjacent planar surfaces
      if(initialMatrix.at<bool>(i,j)==true || features[i].curvatureFactor!=SurfaceFeatureExtract::PLANAR || features[j].curvatureFactor!=SurfaceFeatureExtract::PLANAR){
        coplanar.at<bool>(i,j)=false;
      }
    }
  }
  for(size_t i=0; i<coplanar.size().height; i++){
    for(size_t j=i+1; j<coplanar.size().width; j++){//dont check pairs twice
      if(coplanar.at<bool>(i,j)==true){//candidate
        bool proceed=true;

        //criterion 1: both surfaces have similar mean normal (same orientation)
        proceed=criterion1(features[i].meanNormal, features[j].meanNormal, maxAngle);

        //criterion 2: both surfaces have the same level (combined surface has similar normal)
        if(proceed){
          proceed = criterion2(depthImage, surfaces[i], surfaces[j], features[i].meanNormal, features[j].meanNormal, maxAngle, triangles);
        }

        //criterion3: both surfaces separated by occlusion
        if(proceed){
          proceed=criterion3(depthImage, surfaces[i], surfaces[j], distanceTolerance, outlierTolerance, scanlines);
        }

        if(!proceed){//remove if one of the criterions failed
          coplanar.at<bool>(i,j)=false;
          coplanar.at<bool>(j,i)=false;
        }
      }
    }
  }
  return coplanar;
}


float CoPlanarityFeature::getAngle(Vec4f n1, Vec4f n2){
  float a1=(n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2]);
  float angle=acos(a1)*180.0/M_PI;//angle between the surface normals
  return angle;
}


Point CoPlanarityFeature::getRandomPoint(std::vector<int> surface, int imgWidth){
  int id=surface[rand()%surface.size()];
  int y = (int)floor((float)id/(float)imgWidth);
  int x = id-y*imgWidth;
  Point p(x,y);
  return p;
}


Vec4f CoPlanarityFeature::getNormal(Vec4f p0, Vec4f p1, Vec4f p2){
  Vec4f fa=p1-p0;
  Vec4f fb=p2-p0;
  Vec4f n1(fa[1]*fb[2]-fa[2]*fb[1],//normal
          fa[2]*fb[0]-fa[0]*fb[2],
          fa[0]*fb[1]-fa[1]*fb[0],
          0);
  Vec4f n01=n1/norm(n1);//normalized normal
  return n01;
}


bool CoPlanarityFeature::criterion1(Vec4f n1, Vec4f n2, float maxAngle){
  float angle=getAngle(n1,n2);
  if(angle>90) angle=180.-angle;//flip

  if(angle>maxAngle){
    return false;
  }
  return true;
}


bool CoPlanarityFeature::criterion2(const Mat &depthImage, std::vector<int> &surface1, std::vector<int> &surface2,
                                            Vec4f n1, Vec4f n2, float maxAngle, int triangles){
  std::vector<int> a,b;
  int w = depthImage.size().width;
  if(surface1.size()>surface2.size()){//find bigger surface
    b=surface2;
    a=surface1;
  }else{
    a=surface2;
    b=surface1;
  }

  Vec4f meanNormal((n1[0]+n2[0])/2.,
                  (n1[1]+n2[1])/2.,
                  (n1[2]+n2[2])/2.,
                  0);//mean of both surface normals
  Vec4f meanComb(0,0,0,0);//mean of combined surface normals

  for(int p=0; p<triangles; p++){
    //random combined plane normals
    Point p0=getRandomPoint(b, w);
    Point p1=getRandomPoint(a, w);
    Point p2=getRandomPoint(a, w);

    while (p1.x==p2.x && p1.y==p2.y){//not the same point
      p2=getRandomPoint(a, w);
    }

    Vec4f a(p0.x,p0.y,depthImage.at<float>(p0.x,p0.y),0);
    Vec4f b(p1.x,p1.y,depthImage.at<float>(p1.x,p1.y),0);
    Vec4f c(p2.x,p2.y,depthImage.at<float>(p2.x,p2.y),0);

    Vec4f n01 = getNormal(a,b,c);

    float ang = getAngle(n01,meanNormal);
    if(ang>90.){//flip
      n01*=-1;
      ang=180.-ang;
    }
    meanComb+=n01;
  }
  meanComb/=triangles;

  float ang=getAngle(meanNormal,meanComb);

  if(ang>maxAngle){
    return false;
  }
  return true;
}


bool CoPlanarityFeature::criterion3(const Mat &depthImage, std::vector<int> &surface1, std::vector<int> &surface2,
                                            float distanceTolerance, float outlierTolerance, int scanlines){
  int w = depthImage.size().width;
  int occlusions=0;
  for(int l=0; l<scanlines; l++){
    Point p1=getRandomPoint(surface1, w);
    Point p2=getRandomPoint(surface2, w);

    if(SegmenterHelper::occlusionCheck((Mat&)depthImage, p1, p2, distanceTolerance, outlierTolerance)){
      occlusions++;
    }
  }

  if(occlusions<0.8*scanlines){
    return false;
  }
  return true;
}