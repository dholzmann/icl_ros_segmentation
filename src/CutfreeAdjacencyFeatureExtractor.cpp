#include "CutfreeAdjacencyFeatureExtractor.h"

struct CutfreeAdjacencyFeatureExtractor::Data {
  Data(Mode mode) {
    if(mode==BEST || mode==GPU){
      ransac=new PlanarRansacEstimator(PlanarRansacEstimator::GPU);
    }else{
      ransac=new PlanarRansacEstimator(PlanarRansacEstimator::CPU);
    }
  }

  ~Data() {
  }

  PlanarRansacEstimator* ransac;
};


CutfreeAdjacencyFeatureExtractor::CutfreeAdjacencyFeatureExtractor(Mode mode) :
  m_data(new Data(mode)) {
}


CutfreeAdjacencyFeatureExtractor::~CutfreeAdjacencyFeatureExtractor() {
  delete m_data;
}


Mat CutfreeAdjacencyFeatureExtractor::apply(Mat &xyzh,
            std::vector<std::vector<int> > &surfaces, Mat &testMatrix, float euclideanDistance,
            int passes, int tolerance, Mat labelImage){
  Mat cutfreeMatrix(testMatrix);
  for(unsigned int x=0; x<cutfreeMatrix.size().height; x++){
    cutfreeMatrix.at<int>(x,x)=false;
  }
  std::vector<std::vector<PlanarRansacEstimator::Result>> result = m_data->ransac->apply(xyzh, surfaces,
                cutfreeMatrix, euclideanDistance, passes, tolerance,
                PlanarRansacEstimator::ON_ONE_SIDE, labelImage);

  for(unsigned int x=0; x<static_cast<int>(result.size()); x++){
    for(unsigned int y=0; y<static_cast<int>(result[0].size()); y++){
      if(result[x][y].nacc>=result[x][y].acc){
        cutfreeMatrix.at<int>(x,y)=false;
        cutfreeMatrix.at<int>(y,x)=false;
      }
    }
  }
  return cutfreeMatrix;
}


Mat CutfreeAdjacencyFeatureExtractor::apply(Mat &xyzh,
            std::vector<std::vector<int> > &surfaces, Mat &testMatrix, float euclideanDistance,
            int passes, int tolerance, Mat labelImage,
            std::vector<SurfaceFeatureExtractor::SurfaceFeature> feature, float minAngle){
  Mat cutfreeMatrix(testMatrix);
  for(unsigned int x=0; x<cutfreeMatrix.size().height; x++){
    cutfreeMatrix.at<int>(x,x)=false;
    for(unsigned int y=0; y<cutfreeMatrix.size().width; y++){
      if(cutfreeMatrix.at<int>(x,y)==true || cutfreeMatrix.at<int>(y,x)==true){
        Vec4f n1=feature[x].meanNormal;
        Vec4f n2=feature[y].meanNormal;
        float a1 = (n1[0] * n2[0]+ n1[1] * n2[1]+ n1[2] * n2[2]);
        float ang=acos(a1)*180./M_PI;
        if(ang<minAngle){
          cutfreeMatrix.at<int>(x,y)=false;
          cutfreeMatrix.at<int>(y,x)=false;
        }
      }
    }
  }
  std::vector<std::vector<PlanarRansacEstimator::Result>> result = m_data->ransac->apply(xyzh, surfaces,
                cutfreeMatrix, euclideanDistance, passes, tolerance,
                PlanarRansacEstimator::ON_ONE_SIDE, labelImage);

  for(unsigned int x=0; x<static_cast<int>(result.size()); x++){
    for(unsigned int y=0; y<static_cast<int>(result[0].size()); y++){
      if(result[x][y].nacc>=result[x][y].acc){
        cutfreeMatrix.at<int>(x,y)=false;
        cutfreeMatrix.at<int>(y,x)=false;
      }
    }
  }
  return cutfreeMatrix;
}