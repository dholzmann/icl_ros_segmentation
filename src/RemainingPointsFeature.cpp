#include "RemainingPointsFeature.h"

void RemainingPointsFeature::apply(Mat &xyz, const Mat &depthImage, Mat &labelImage, Mat &maskImage,
                      std::vector<std::vector<int> > &surfaces, std::vector<std::vector<int> > &segments, int minSize, float euclideanDistance, int radius, float assignEuclideanDistance, int supportTolerance){
  calculateLocalMinima(depthImage, maskImage, radius);
  apply(xyz, labelImage, maskImage, surfaces, segments, minSize, euclideanDistance, assignEuclideanDistance, supportTolerance);
}


void RemainingPointsFeature::apply(Mat &xyz, Mat &labelImage, Mat &maskImage,
                  std::vector<std::vector<int> > &surfaces, std::vector<std::vector<int> > &segments, int minSize, float euclideanDistance, float assignEuclideanDistance, int supportTolerance){
  int numCluster=surfaces.size();
  clusterRemainingPoints(xyz, surfaces, labelImage, maskImage, minSize, euclideanDistance, numCluster);
  std::vector<std::vector<int> > neighbours;
  std::vector<std::vector<int> > neighboursPoints;
  detectNeighbours(xyz, surfaces, labelImage, neighbours, neighboursPoints, numCluster, assignEuclideanDistance);

  ruleBasedAssignment(xyz, labelImage, surfaces, segments, neighbours, neighboursPoints, numCluster, supportTolerance);
}


Mat_<bool> RemainingPointsFeature::apply(Mat &xyz, const Mat &depthImage, Mat &labelImage, Mat &maskImage,
                  std::vector<std::vector<int> > &surfaces, int minSize, float euclideanDistance, int radius, float assignEuclideanDistance){
  calculateLocalMinima(depthImage, maskImage, radius);
  return apply(xyz, labelImage, maskImage, surfaces, minSize, euclideanDistance, assignEuclideanDistance);
}


Mat_<bool> RemainingPointsFeature::apply(Mat &xyz, Mat &labelImage, Mat &maskImage,
                  std::vector<std::vector<int> > &surfaces, int minSize, float euclideanDistance, float assignEuclideanDistance){
  int numCluster=surfaces.size();
  clusterRemainingPoints(xyz, surfaces, labelImage, maskImage, minSize, euclideanDistance, numCluster);
  std::vector<std::vector<int> > neighbours;
  std::vector<std::vector<int> > neighboursPoints;
  detectNeighbours(xyz, surfaces, labelImage, neighbours, neighboursPoints, numCluster, assignEuclideanDistance);

  //create Matrix
  Mat_<bool> remainingMatrix(surfaces.size(), surfaces.size(), false);
  for(unsigned int x=numCluster; x<surfaces.size(); x++){
    std::vector<int> nb = neighbours[x-numCluster];
    for(unsigned int y=0; y<nb.size(); y++){
      remainingMatrix(x,nb[y])=true;//nb-1
      remainingMatrix(nb[y],x)=true;//nb-1
    }
  }
  return remainingMatrix;
}


void RemainingPointsFeature::calculateLocalMinima(const Mat &depthImage, Mat &maskImage, int radius){
  int w=depthImage.size().width;
  int h=depthImage.size().height;
  int ii=radius;
  for(int y=0; y<h; y++){
    for(int x=0; x<w; x++){
      if(maskImage.at<int>(x,y)==0){
        bool localMin=false;
        if(x>=ii && x<w-ii){
          localMin=(depthImage.at<float>(x-ii,y)<depthImage.at<float>(x,y) && depthImage.at<float>(x+ii,y)<depthImage.at<float>(x,y));
        }else if(y>=ii && y<h-ii && localMin==false){
            localMin=(depthImage.at<float>(x,y-ii)<depthImage.at<float>(x,y) && depthImage.at<float>(x,y+ii)<depthImage.at<float>(x,y));
        }else if(x>=ii && x<w-ii && y>=ii && y<h-ii && localMin==false){
            localMin=(depthImage.at<float>(x-ii,y+ii)<depthImage.at<float>(x,y) && depthImage.at<float>(x+ii,y-ii)<depthImage.at<float>(x,y));
        }else if(x>=ii && x<w-ii && y>=ii && y<h-ii && localMin==false){
            localMin=(depthImage.at<float>(x+ii,y+ii)<depthImage.at<float>(x,y) && depthImage.at<float>(x-ii,y-ii)<depthImage.at<float>(x,y));
        }
        if(localMin){
          maskImage.at<int>(x,y)=1;
        }
      }
    }
  }
}


void RemainingPointsFeature::clusterRemainingPoints(Mat &xyz, std::vector<std::vector<int> > &surfaces, Mat &labelImage, Mat &maskImage,
                                        int minSize, float euclideanDistance, int numCluster){
  //cluster remaining points by euclidean distance
  RegionGrowing rg;
  const Mat &result = rg.applyFloat4EuclideanDistance(xyz, maskImage, euclideanDistance, minSize, numCluster+1);
  std::vector<std::vector<int> > regions=rg.getRegions();
  surfaces.insert(surfaces.end(), regions.begin(), regions.end());

  //relabel the label image
  Size s=labelImage.size();
  for(int y=0; y<s.height; y++){
    for(int x=0; x<s.width; x++){
      if(labelImage.at<int>(x,y)==0){
        labelImage.at<int>(x,y)=result.at<int>(x,y);
      }
    }
  }
}

void RemainingPointsFeature::detectNeighbours(Mat &xyz, std::vector<std::vector<int> > &surfaces, Mat &labelImage, std::vector<std::vector<int> > &neighbours,
                                  std::vector<std::vector<int> > &neighboursPoints, int numCluster, float assignEuclideanDistance){
  Size s = labelImage.size();
  //determine neighbouring surfaces
  for(unsigned int x=numCluster; x<surfaces.size(); x++){
    std::vector<int> nb;//neighbours
    std::vector<int> nbPoints;//number of connecting points
    for(unsigned int y=0; y<surfaces[x].size(); y++){
      for(int p=-1; p<=1; p++){//all 8 neighbours
        for(int q=-1; q<=1; q++){
          int p1 = surfaces[x][y];
          int p2 = surfaces[x][y]+p+s.width*q;
          if(p2>=0 && p2<s.width*s.height && p1!=p2 && labelImage.at<int>(p1)>labelImage.at<int>(p2) && labelImage.at<int>(p2)!=0){//bounds, id, value, not 0
            if(checkNotExist(labelImage.at<int>(p2)-1, nb, nbPoints) && norm(xyz.at<Vec3f>(p1), xyz.at<Vec3f>(p2), NORM_L2)<assignEuclideanDistance){// /4.
              nb.push_back(labelImage.at<int>(p2)-1);//id, not label-value
              nbPoints.push_back(1);
            }
          }
        }
      }
    }
    neighbours.push_back(nb);
    neighboursPoints.push_back(nbPoints);
  }
}


bool RemainingPointsFeature::checkNotExist(int zw, std::vector<int> &nb, std::vector<int> &nbPoints){
  if(zw!=0){
    for(unsigned int z=0; z<nb.size(); z++){
      if(nb[z]==zw){
        nbPoints[z]++;
        return false;
      }
    }
    return true;
  }
  return false;
}


void RemainingPointsFeature::ruleBasedAssignment(Mat &xyz, Mat &labelImage, std::vector<std::vector<int> > &surfaces, std::vector<std::vector<int> > &segments,
                                    std::vector<std::vector<int> > &neighbours, std::vector<std::vector<int> > &neighboursPoints, int numCluster, int supportTolerance){

  std::vector<int> assignment = segmentMapping(segments, surfaces.size());

  for(unsigned int x=numCluster; x<surfaces.size(); x++){
    std::vector<int> nb = neighbours[x-numCluster];
    std::vector<int> nbPoints = neighboursPoints[x-numCluster];
    if(nb.size()==0){ //no neighbours -> new segment
      std::vector<int> seg;
      seg.push_back(x);
      segments.push_back(seg);
      assignment[x] = segments.size()-1;
    }
    else if(nb.size()==1 && surfaces[x].size()<15){ //very small -> assign
      segments[assignment[nb[0]]].push_back(x);
      assignment[x]=assignment[nb[0]];
    }
    else if(nb.size()==1){
      bool supported = checkSupport(labelImage, surfaces[x], nb[0], supportTolerance);
      if(supported){ //new blob
      //if(nbPoints[0]<9){ //new blob (weak connectivity)
        std::vector<int> seg;
        seg.push_back(x);
        segments.push_back(seg);
        assignment[x] = segments.size()-1;
      }
      else{ //assign
        segments[assignment[nb[0]]].push_back(x);
        assignment[x]=assignment[nb[0]];
      }
    }
    else if(nb.size()>1){
      bool same=true;
      for(unsigned int a=1; a<nb.size(); a++){
        if(assignment[nb[a]]!=assignment[nb[0]]){
          same=false;
        }
      }
      if(same==true){ //same blob->assign
        for(unsigned int p=0; p<nb.size(); p++){
          segments[assignment[nb[p]]].push_back(x);
          assignment[x]=assignment[nb[p]];
        }
      }
      else{ //different blob -> determine best match by RANSAC
        int bestNeighbourID = ransacAssignment(xyz, surfaces, nb, x);
        //assign to neighbour with smallest error
        segments[assignment[nb[bestNeighbourID]]].push_back(x);
        assignment[x]=assignment[nb[bestNeighbourID]];
      }
    }
  }
}


std::vector<int> RemainingPointsFeature::segmentMapping(std::vector<std::vector<int> > &segments, int numSurfaces){
  //mapping for faster calculation
  std::vector<int> assignment (numSurfaces,0);
  for(unsigned int i=0; i<segments.size(); i++){
    for(unsigned int j=0; j<segments[i].size(); j++){
      assignment[segments[i][j]]=i;
    }
  }
  return assignment;
}


int RemainingPointsFeature::ransacAssignment(Mat &xyz, std::vector<std::vector<int> > &surfaces, std::vector<int> &nb, int x){
  std::vector<std::vector<Vec4f> > n0;
  std::vector<std::vector<float> > dist;
  for(unsigned int i=0; i<nb.size(); i++){//calculate RANSAC models on neighbours
    std::vector<Vec4f> n00(10);
    std::vector<float> dist0(10);

    PlanarRansac::calculateRandomModels(xyz, surfaces[nb[i]], n00, dist0, 10);
    n0.push_back(n00);
    dist.push_back(dist0);
  }
  float bestNeighbourScore=10000000;
  int bestNeighbourID=0;
  for(unsigned int i=0; i<n0.size(); i++){//neighbours
    float bestPassScore=10000000;
    for(unsigned int j=0; j<n0[i].size(); j++){//passes
      float passScore=0;
      for(unsigned int p=0; p<surfaces[x].size(); p++){
        float s1 = (xyz.at<Vec4f>(surfaces[x][p])[0]*n0[i][j][0]+
                    xyz.at<Vec4f>(surfaces[x][p])[1]*n0[i][j][1]+
                    xyz.at<Vec4f>(surfaces[x][p])[2]*n0[i][j][2])-dist[i][j];
        passScore+=fabs(s1);
      }
      passScore/=surfaces[x].size();
      if(passScore<bestPassScore){
        bestPassScore=passScore;
      }
    }
    if(bestPassScore<bestNeighbourScore){
      bestNeighbourScore=bestPassScore;
      bestNeighbourID=i;
    }
  }
  return bestNeighbourID;
}

bool RemainingPointsFeature::checkSupport(Mat &labelImage, std::vector<int> &surface, int neighbourID, int supportTolerance){
  int count=0;
  Size s = labelImage.size();
  for(unsigned int y=0; y<surface.size(); y++){
    for(int p=-1; p<=1; p++){//all 8 neighbours
      for(int q=-1; q<=1; q++){
        int p1 = surface[y];
        int p2 = surface[y]+p+s.width*q;
        if(p2>=0 && p2<s.width*s.height && p1!=p2 && labelImage.at<int>(p1)!=labelImage.at<int>(p2) && labelImage.at<int>(p2)!=0 && labelImage.at<int>(p2)-1!=neighbourID){//bounds, id-self, value-self, not 0, value-neighb.
          if(count<supportTolerance){
            count++;
          }else{
            return false;
          }
        }
      }
    }
  }
  return true;
}
