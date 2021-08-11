#include "PlanarRansac.h"

struct PlanarRansac::Data {
  Data(Mode mode) {
    clReady = false;

    if(mode==BEST || mode==GPU){
      useCL=true;
    }else{
      useCL=false;
    }
  }

  ~Data() {
  }

  bool clReady;
  bool useCL;


};


PlanarRansac::PlanarRansac(Mode mode) :
  m_data(new Data(mode)) {
}


PlanarRansac::~PlanarRansac() {
  delete m_data;
}


PlanarRansac::RansacResult PlanarRansac::apply(Mat &xyzh,
            std::vector<int> &srcIDs, std::vector<int> &dstIDs, float threshold, int passes,
            int subset, int tolerance, int optimization){

  std::vector<Vec4f> srcPoints(srcIDs.size());
  std::vector<Vec4f> dstPoints(dstIDs.size());
  for(unsigned int i=0; i<srcIDs.size(); i++){
    srcPoints[i]=xyzh.at<Vec4f>(srcIDs.at(i));
  }
  for(unsigned int i=0; i<dstIDs.size(); i++){
    dstPoints[i]=xyzh.at<Vec4f>(dstIDs.at(i));
  }
  return apply(srcPoints, dstPoints, threshold, passes, subset, tolerance, optimization);
}


PlanarRansac::RansacResult PlanarRansac::apply(std::vector<Vec4f> &srcPoints,
            std::vector<Vec4f> &dstPoints, float threshold, int passes, int subset, int tolerance, int optimization){

  int numPoints=dstPoints.size();
  std::vector<Vec4f> n0(passes);
  std::vector<float> dist(passes);
  std::vector<int> cAbove(passes,0);
  std::vector<int> cBelow(passes,0);
  std::vector<int> cOn(passes,0);

  calculateRandomModels(srcPoints, n0, dist, passes);

  calculateSingleCPU(dstPoints, threshold, passes, subset, n0, dist, cAbove, cBelow, cOn);

  return createResult(n0, dist, cAbove, cBelow, cOn, threshold, passes, tolerance, optimization, numPoints);
}


std::vector<std::vector<PlanarRansac::RansacResult>> PlanarRansac::apply(Mat &xyzh,
            std::vector<std::vector<int> > &pointIDs, Mat &testMatrix, float threshold,
            int passes, int tolerance, int optimization, Mat labelImage){

  std::vector<std::vector<Vec4f> > n0Pre(testMatrix.size().height, std::vector<Vec4f>(passes));
  std::vector<std::vector<float> > distPre(testMatrix.size().height, std::vector<float>(passes));
  std::vector<Vec4f> n0(testMatrix.size().height*passes);
  std::vector<float> dist(testMatrix.size().height*passes);
  std::vector<int> cAbove;
  std::vector<int> cBelow;
  std::vector<int> cOn;

  std::vector<int> adjs;

  std::vector<int> start(testMatrix.size().height);
  std::vector<int> end(testMatrix.size().height);

  int count=0;
  for(size_t i=0; i<testMatrix.size().height; i++){

    start[i]=count;

    calculateRandomModels(xyzh, pointIDs.at(i), n0Pre.at(i), distPre.at(i), passes);

    for(int k=0; k<passes; k++){
      n0[i*passes+k]=n0Pre.at(i)[k];
      dist[i*passes+k]=distPre.at(i)[k];
    }

    for(size_t j=0; j<testMatrix.size().height; j++){
      if(testMatrix.at<int>(i,j)==1){
        adjs.push_back(j);
        count++;
        for(int k=0; k<passes; k++){
          cAbove.push_back(0);
          cBelow.push_back(0);
          cOn.push_back(0);
        }
      }
    }

    end[i]=count;
  }

  calculateMultiCPU(xyzh, pointIDs, testMatrix, threshold, passes, n0Pre, distPre, cAbove, cBelow, cOn, adjs, start, end);

  return createResultMatrix(testMatrix, start, end, adjs, cAbove, cBelow, cOn, pointIDs, n0Pre, distPre, threshold, passes, tolerance, optimization);
}


void PlanarRansac::relabel(Mat &xyzh, Mat &newMask, Mat &oldLabel,
                            Mat &newLabel, int desiredID, int srcID, float threshold, RansacResult &result){

  Size size = newMask.size();
  int w = size.width;
  int h = size.height;

  //if(m_data->useCL==true && m_data->clReady==true){
  //  relabelCL(xyzh, newMask, oldLabel, newLabel, desiredID, srcID, threshold, result, w, h);
  //}else{//CPU
    relabelCPU(xyzh, newMask, oldLabel, newLabel, desiredID, srcID, threshold, result, w, h);
  //}
}

void PlanarRansac::calculateMultiCPU(Mat &xyzh, std::vector<std::vector<int> > &pointIDs, Mat &testMatrix,
                float threshold, int passes, std::vector<std::vector<Vec4f> > &n0Pre, std::vector<std::vector<float> > &distPre, std::vector<int> &cAbove,
                std::vector<int> &cBelow, std::vector<int> &cOn, std::vector<int> &adjs, std::vector<int> &start, std::vector<int> &end){
  for(size_t i=0; i<testMatrix.size().height; i++){
    for(int j=start[i]; j<end[i]; j++){
      int k=adjs[j];
      for(unsigned int m=0; m<pointIDs.at(k).size(); m++){
        for(int l=0; l<passes; l++){
          Vec4f n01 = n0Pre.at(i).at(l);
          Vec4f point = xyzh.at<Vec4f>(pointIDs.at(k).at(m));
          float s1 = (point[0]*n01[0]+point[1]*n01[1]+point[2]*n01[2])-distPre.at(i).at(l);
          if((s1>=-threshold && s1<=threshold)){
            cOn[j*passes+l]++;
          }else if(s1>threshold){
            cAbove[j*passes+l]++;
          }else if(s1<threshold){
            cBelow[j*passes+l]++;
          }
        }
      }
    }
  }
}

void PlanarRansac::calculateSingleCPU(std::vector<Vec4f> &dstPoints, float threshold, int passes, int subset,
            std::vector<Vec4f> &n0, std::vector<float> &dist, std::vector<int> &cAbove, std::vector<int> &cBelow, std::vector<int> &cOn){
  for(int p=0; p<passes; p++){
    for(unsigned int q=0; q<dstPoints.size(); q+=subset){
      Vec4f n01 = n0[p];
      Vec4f point = dstPoints.at(q);
      float s1 = (point[0]*n01[0]+point[1]*n01[1]+point[2]*n01[2])-dist[p];
      if((s1>=-threshold && s1<=threshold)){
        cOn[p]++;
      }else if(s1>threshold){
        cAbove[p]++;
      }else if(s1<threshold){
        cBelow[p]++;
      }
    }
  }
}

void PlanarRansac::calculateRandomModels(std::vector<Vec4f> &srcPoints, std::vector<Vec4f> &n0, std::vector<float> &dist, int passes){
  for(int i=0; i<passes; i++){
    Vec4f p0i=srcPoints.at(rand()%srcPoints.size());
    Vec4f p1i=srcPoints.at(rand()%srcPoints.size());
    Vec4f p2i=srcPoints.at(rand()%srcPoints.size());
    Vec4f fa = p1i-p0i;
    Vec4f fb = p2i-p0i;
    Vec4f rPoint = p0i;

    calculateModel(fa, fb, rPoint, n0[i], dist[i]);
  }
}


void PlanarRansac::calculateRandomModels(Mat &xyzh, std::vector<int> &srcPoints, std::vector<Vec4f> &n0, std::vector<float> &dist, int passes){
  for(int i=0; i<passes; i++){
    int p0i=srcPoints.at(rand()%srcPoints.size());
    int p1i=srcPoints.at(rand()%srcPoints.size());
    int p2i=srcPoints.at(rand()%srcPoints.size());
    Vec4f fa = xyzh.at<Vec4f>(p1i)-xyzh.at<Vec4f>(p0i);
    Vec4f fb = xyzh.at<Vec4f>(p2i)-xyzh.at<Vec4f>(p0i);
    Vec4f rPoint = xyzh.at<Vec4f>(p0i);

    calculateModel(fa, fb, rPoint, n0[i], dist[i]);
  }
}


PlanarRansac::RansacResult PlanarRansac::createResult(std::vector<Vec4f> &n0, std::vector<float> &dist, std::vector<int> &cAbove,
                std::vector<int> &cBelow, std::vector<int> &cOn, float threshold, int passes, int tolerance, int optimization, int numPoints){
  int maxMatch=0;
  int maxMatchID=0;
  int countAcc=0;
  int countNAcc=0;
  for(int i=0;i<passes; i++){
    if(optimization==ON_ONE_SIDE){
      if(cAbove[i]<tolerance || cBelow[i]<tolerance){
        countAcc++;
      }else{
        countNAcc++;
      }
    }
    if(optimization==MAX_ON){
      if(cOn[i]>maxMatch){
        maxMatch=cOn[i];
        maxMatchID=i;
      }
    }else if(optimization==ON_ONE_SIDE){
      if(cBelow[i]>maxMatch){
        maxMatch=cBelow[i];
        maxMatchID=i;
      }
      if(cAbove[i]>maxMatch){
        maxMatch=cAbove[i];
        maxMatchID=i;
      }
    }
  }

  RansacResult result;
  result.numPoints=numPoints;
  result.countOn=cOn[maxMatchID];
  result.countAbove=cAbove[maxMatchID];
  result.countAbove=cBelow[maxMatchID];
  result.euclideanThreshold=threshold;
  result.n0=n0[maxMatchID];
  result.dist=dist[maxMatchID];
  result.tolerance=tolerance;
  result.acc=countAcc;
  result.nacc=countNAcc;
  return result;
}


std::vector<std::vector<PlanarRansac::RansacResult>> PlanarRansac::createResultMatrix(Mat &testMatrix, std::vector<int> &start,
                std::vector<int> &end, std::vector<int> &adjs, std::vector<int> &cAbove, std::vector<int> &cBelow, std::vector<int> &cOn,
                std::vector<std::vector<int> > &pointIDs, std::vector<std::vector<Vec4f> > &n0Pre, std::vector<std::vector<float> > &distPre,
                float threshold, int passes, int tolerance, int optimization){
  RansacResult init;
  std::vector<std::vector<PlanarRansac::RansacResult>> result(testMatrix.size().height, std::vector<RansacResult>(testMatrix.size().width));

  for(size_t i=0; i<testMatrix.size().height; i++){
    for(int j=start[i]; j<end[i]; j++){
      int k=adjs[j];
      std::vector<int> above(passes);
      std::vector<int> below(passes);
      std::vector<int> on(passes);
      for(int l=0; l<passes; l++){
        above[l]=cAbove[j*passes+l];
        below[l]=cBelow[j*passes+l];
        on[l]=cOn[j*passes+l];
      }
      int numPoints = pointIDs.at(k).size();
      RansacResult res = createResult(n0Pre[i], distPre[i], above, below, on, threshold, passes, tolerance, optimization, numPoints);
      result[i][k]=res;
    }
  }

  return result;
}


void PlanarRansac::calculateModel(Vec4f &fa, Vec4f &fb, Vec4f &rPoint, Vec4f &n0, float &dist){
  Vec4f n1;
  n1[0]=fa[1]*fb[2]-fa[2]*fb[1];
  n1[1]=fa[2]*fb[0]-fa[0]*fb[2];
  n1[2]=fa[0]*fb[1]-fa[1]*fb[0];
  n0[0]=n1[0]/norm(n1);
  n0[1]=n1[1]/norm(n1);
  n0[2]=n1[2]/norm(n1);
  dist = rPoint[0]*n0[0]+rPoint[1]*n0[1]+ rPoint[2]*n0[2];
}

void PlanarRansac::relabelCPU(Mat &xyzh, Mat &newMask, Mat &oldLabel, Mat &newLabel,
                  int desiredID, int srcID, float threshold, RansacResult &result, int w, int h){
  for(int y=0; y<h; y++){
    for(int x=0; x<w; x++){
      int i=x+y*w;
      if(newMask.at<int>(x,y,0)==0){
        if(oldLabel.at<int>(x,y,0)==srcID){
          newLabel.at<int>(x,y,0)=desiredID;
          newMask.at<int>(x,y,0)=1;
        }else{
          Vec4f n01 = result.n0;
          float dist = result.dist;
          float s1 = (xyzh.at<Vec4f>(i)[0]*n01[0]+xyzh.at<Vec4f>(i)[1]*n01[1]+xyzh.at<Vec4f>(i)[2]*n01[2])-dist;
          if((s1>=-threshold && s1<=threshold) && newMask.at<int>(x,y,0)==0){
            newLabel.at<int>(x,y,0)=desiredID;
            newMask.at<int>(x,y,0)=1;
          }else{
            newLabel.at<int>(x,y,0)=0;
            newMask.at<int>(x,y,0)=0;
          }
        }
      }else{
        newLabel.at<int>(x,y,0)=oldLabel.at<int>(x,y,0);
      }
    }
  }
}