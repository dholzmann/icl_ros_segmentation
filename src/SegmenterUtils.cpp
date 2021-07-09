/********************************************************************
**                Image Component Library (ICL)                    **
**                                                                 **
** Copyright (C) 2006-2013 CITEC, University of Bielefeld          **
**                         Neuroinformatics Group                  **
** Website: www.iclcv.org and                                      **
**          http://opensource.cit-ec.de/projects/icl               **
**                                                                 **
** File   : ICLGeom/src/ICLGeom/SegmenterUtils.cpp                 **
** Module : ICLGeom                                                **
** Authors: Andre Ueckermann                                       **
**                                                                 **
**                                                                 **
** GNU LESSER GENERAL PUBLIC LICENSE                               **
** This file may be used under the terms of the GNU Lesser General **
** Public License version 3.0 as published by the                  **
**                                                                 **
** Free Software Foundation and appearing in the file LICENSE.LGPL **
** included in the packaging of this file.  Please review the      **
** following information to ensure the license requirements will   **
** be met: http://www.gnu.org/licenses/lgpl-3.0.txt                **
**                                                                 **
** The development of this software was supported by the           **
** Excellence Cluster EXC 277 Cognitive Interaction Technology.    **
** The Excellence Cluster EXC 277 is a grant of the Deutsche       **
** Forschungsgemeinschaft (DFG) in the context of the German       **
** Excellence Initiative.                                          **
**                                                                 **
********************************************************************/

#include "SegmenterUtils.h"

#ifdef ICL_HAVE_OPENCL
//OpenCL kernel code
static char utilsKernel[] =
  "  #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable                                                           \n"
  "__kernel void                                                                                                                  \n"
  "segmentColoring(__global int const * assignment, __global uchar * colorR, __global uchar * colorG, __global uchar * colorB)    \n"
  "{                                                                                                                              \n"
  "  size_t id =  get_global_id(0);                                                                                               \n"
  "  if(assignment[id]==0)                                                                                                        \n"
  "  {                                                                                                                            \n"
  "    colorR[id]=128;                                                                                                            \n"
  "    colorG[id]=128;                                                                                                            \n"
  "    colorB[id]=128;                                                                                                            \n"
  "  }                                                                                                                            \n"
  "  else                                                                                                                         \n"
  "  {                                                                                                                            \n"
  "    int H=(int)(assignment[id]*35.)%360;                                                                                       \n"
  "    float S=1.0-assignment[id]*0.01;                                                                                           \n"
  "    float hi=floor((float)H/60.);                                                                                              \n"
  "	   float f=((float)H/60.)-hi;                                                                                                 \n"
  "	   float pp=1.0-S;                                                                                                            \n"
  "	   float qq=1.0-S*f;                                                                                                          \n"
  "	   float tt=1.0-S*(1.-f);                                                                                                     \n"
  "	   float newR=0;                                                                                                              \n"
  "	   float newG=0;                                                                                                              \n"
  "	   float newB=0;                                                                                                              \n"
  "	   if((int)hi==0 || (int)hi==6){                                                                                              \n"
  "	  	 newR=1.0;                                                                                                                \n"
  "	     newG=tt;                                                                                                                 \n"
  "		   newB=pp;                                                                                                                 \n"
  "	   }else if((int)hi==1){                                                                                                      \n"
  "		   newR=qq;                                                                                                                 \n"
  "		   newG=1.0;                                                                                                                \n"
  "		   newB=pp;                                                                                                                 \n"
  "	   }else if((int)hi==2){                                                                                                      \n"
  "		   newR=pp;                                                                                                                 \n"
  "		   newG=1.0;                                                                                                                \n"
  "	     newB=tt;                                                                                                                 \n"
  "	   }else if((int)hi==3){                                                                                                      \n"
  "		   newR=pp;                                                                                                                 \n"
  "	     newG=qq;                                                                                                                 \n"
  "	     newB=1.0;                                                                                                                \n"
  "	   }else if((int)hi==4){                                                                                                      \n"
  "		   newR=tt;                                                                                                                 \n"
  "		   newG=pp;                                                                                                                 \n"
  "		   newB=1.0;                                                                                                                \n"
  "	   }else if((int)hi==5){                                                                                                      \n"
  "	     newR=1.0;                                                                                                                \n"
  "		   newG=pp;                                                                                                                 \n"
  "		   newB=qq;                                                                                                                 \n"
  "	   }                                                                                                                          \n"
  "    colorR[id]=(unsigned char)(newR*255.);                                                                                     \n"
  "    colorG[id]=(unsigned char)(newG*255.);                                                                                     \n"
  "    colorB[id]=(unsigned char)(newB*255.);                                                                                     \n"
  "  }                                                                                                                            \n"
  "}                                                                                                                              \n"
  "__kernel void                                                                                                                  \n"
  "calculatePointAssignment(__global float4 const * xyz, __global uchar * mask, __global int const * assignment,                  \n"
  "                         int const radius, int const numFaces, __global uchar * neighbours, __global int * assignmentOut,       \n"
  "                         int const w, int const h, float const maxDist)                                                        \n"
  "{                                                                                                                              \n"
  "  int x = get_global_id(0);                                                                                                    \n"
  "  int y = get_global_id(1);                                                                                                    \n"
  "  size_t id = x+y*w;                                                                                                           \n"
  "  float dist=100000;                                                                                                           \n"
  "  int ass=0;                                                                                                                   \n"
  "  bool assigned=false;                                                                                                         \n"
  "  if(mask[id]==0 && assignment[id]==0){                                                                                        \n"
  "    bool adj[100];                                                                                                             \n"
  "    for(int a=0; a<numFaces; a++){                                                                                             \n"
  "      adj[a]=false;                                                                                                            \n"
  "    }                                                                                                                          \n"
  "    for(int xx=-radius; xx<=radius; xx++){                                                                                     \n"
  "      for(int yy=-radius; yy<=radius; yy++){                                                                                   \n"
  "        if(x+xx>=0 && x+xx<w && y+yy>=0 && y+yy<h && assignment[(x+xx)+w*(y+yy)]!=0){                                          \n"
  "          float4 pointa=xyz[id];                                                                                               \n"
  "          pointa.w=1.0;                                                                                                        \n"
  "          float4 pointb=xyz[(x+xx)+w*(y+yy)];                                                                                  \n"
  "          pointb.w=1.0;                                                                                                        \n"
  "          float dist3 = distance(pointa,pointb);                                                                               \n"
  "          if(dist3<maxDist){                                                                                                   \n"
  "            adj[assignment[(x+xx)+w*(y+yy)]-1]=true;                                                                           \n"
  "          }                                                                                                                    \n"
  "          if(dist3<dist && dist3<maxDist){                                                                                     \n"
  "            dist=dist3;                                                                                                        \n"
  "            ass=assignment[(x+xx)+w*(y+yy)];                                                                                   \n"
  "            assigned=true;                                                                                                     \n"
  "          }                                                                                                                    \n"
  "        }                                                                                                                      \n"
  "      }                                                                                                                        \n"
  "    }                                                                                                                          \n"
  "    for(int a=0; a<numFaces-1; a++){                                                                                           \n"
  "      for (int b=a+1; b<numFaces; b++){                                                                                        \n"
  "        if(adj[a]==true && adj[b]==true){                                                                                      \n"
  "          neighbours[a*numFaces+b]=1;                                                                                          \n"
  "          neighbours[b*numFaces+a]=1;                                                                                          \n"
  "        }                                                                                                                      \n"
  "      }                                                                                                                        \n"
  "    }                                                                                                                          \n"
  "    if(assigned==true){                                                                                                        \n"
  "      assignmentOut[id]=ass;                                                                                                   \n"
  "      mask[id]=1;                                                                                                              \n"
  "    }                                                                                                                          \n"
  "    else                                                                                                                       \n"
  "    {                                                                                                                          \n"
  "      assignmentOut[id]=assignment[id];                                                                                        \n"
  "    }                                                                                                                          \n"
  "  }                                                                                                                            \n"
  "  else                                                                                                                         \n"
  "  {                                                                                                                            \n"
  "    assignmentOut[id]=assignment[id];                                                                                          \n"
  "  }                                                                                                                            \n"
  "}                                                                                                                              \n"
;
#endif

using namespace cv;

struct SegmenterUtils::Data {
  Data(Mode mode) {
    clReady = false;
    kernelSegmentColoringInitialized=false;
    kernelPointAssignmentInitialized=false;
    size = Size(0,0);
    stabelizeCounter=0;

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

  Size size;
  bool kernelSegmentColoringInitialized;
  bool kernelPointAssignmentInitialized;

  int stabelizeCounter;
  //core::Img32s lastLabelImage;
  Mat lastLabelImage;

  #ifdef ICL_HAVE_OPENCL
    //OpenCL data
    std::vector<unsigned char> segmentColorImageRArray;
    std::vector<unsigned char> segmentColorImageGArray;
    std::vector<unsigned char> segmentColorImageBArray;

    std::vector<unsigned char> maskArray;
    std::vector<int> assignmentArray;

    //OpenCL
    utils::CLProgram program;
    utils::CLKernel kernelSegmentColoring;
    utils::CLKernel kernelPointAssignment;

    //OpenCL buffer
    utils::CLBuffer segmentColorImageRBuffer;
    utils::CLBuffer segmentColorImageGBuffer;
    utils::CLBuffer segmentColorImageBBuffer;
    utils::CLBuffer assignmentBuffer;

    utils::CLBuffer neighboursBuffer;
    utils::CLBuffer xyzBuffer;
    utils::CLBuffer maskBuffer;
    utils::CLBuffer assignmentOutBuffer;
  #endif
};


SegmenterUtils::SegmenterUtils(Mode mode) :
  m_data(new Data(mode)) {

  if(m_data->useCL==true){
    #ifdef ICL_HAVE_OPENCL
      try
      {
        m_data->program = utils::CLProgram("gpu", utilsKernel);
        m_data->kernelSegmentColoring = m_data->program.createKernel("segmentColoring");
        m_data->kernelPointAssignment = m_data->program.createKernel("calculatePointAssignment");
        m_data->clReady = true;
      } catch (utils::CLException &err) { //catch openCL errors
        ERROR_LOG(err.what());
        m_data->clReady = false;

      }

    #else
      std::cout << "no openCL parallelization available" << std::endl;
      m_data->clReady = false;
    #endif
  }
}


SegmenterUtils::~SegmenterUtils() {
  delete m_data;
}


Mat SegmenterUtils::createColorImage(Mat &labelImage){
  //core::Img8u colorImage;
  Mat colorImage;
  if(m_data->useCL==true && m_data->clReady==true){
    createColorImageCL(labelImage, colorImage);
  }else{
    createColorImageCPU(labelImage, colorImage);
  }
  return colorImage;
}


/*core::Img8u*/Mat SegmenterUtils::createROIMask(/*core::DataSegment<float,4>*/Mat &xyzh, /*core::Img32f*/Mat &depthImage,
            float xMin, float xMax, float yMin, float yMax, float zMin, float zMax){
  Size size = depthImage.size();
  /*core::Img8u*/ Mat maskImage(size, CV_8UC1);
  //core::Channel8u maskImageC = maskImage[0];
  //core::Channel32f depthImageC = depthImage[0];
  for(int y=0;y<size.height;++y){
    for(int x=0;x<size.width;++x){
      int i = x+size.width*y;
      if(xyzh.at<Point4f>(y, x)[0]<xMin || xyzh.at<Point4f>(y, x)[0]>xMax || xyzh.at<Point4f>(y, x)[1]<yMin || xyzh.at<Point4f>(y, x)[1]>yMax || xyzh.at<Point4f>(y, x)[2]<zMin || xyzh.at<Point4f>(y, x)[2]>zMax){
        maskImage.at<int>(x,y)=1;
      }else{
        maskImage.at<int>(x,y)=0;
      }
      if(depthImage.at<int>(x,y)==2047){
        maskImage.at<int>(x,y)=1;
      }
    }
  }
  return maskImage;
}


/*core::Img8u*/Mat SegmenterUtils::createMask(/*core::Img32f*/Mat &depthImage){
  Size size = depthImage.size();
  /*core::Img8u*/Mat maskImage(size, CV_8UC1);
  /*core::Channel8uMat maskImageC = maskImage[0];
  core::Channel32fMat depthImageC = depthImage[0];*/
  for(int y=0;y<size.height;++y){
    for(int x=0;x<size.width;++x){
      maskImage.at<int>(x,y)=0;
      if(depthImage.at<float>(x,y)==2047){
        maskImage.at<int>(x,y)=1;
      }
    }
  }
  return maskImage;
}


/*core::Img32s*/Mat SegmenterUtils::stabelizeSegmentation(/*core::Img32s*/Mat &labelImage){
  /*core::Img32s stableLabelImage(labelImage.getSize(),1,core::formatMatrix);
  core::Channel32s labelImageC = labelImage[0];
  core::Channel32s stableLabelImageC = stableLabelImage[0];
  */
 Mat stableLabelImage(labelImage.size(), CV_8UC1);
  Size size = labelImage.size();
  if(m_data->stabelizeCounter==0){//first image
    m_data->lastLabelImage = labelImage.clone();
  }else{
    //core::Channel32s lastLabelImageC = m_data->lastLabelImage[0];

    //count number of segments of previous and current label image
    int countCur=0;
    int countLast=0;
    for(int y=0; y<size.height; y++){
      for(int x=0; x<size.width; x++){
        if(labelImage.at<int>(x,y)>countCur){
          countCur=labelImage.at<int>(x,y);
        }
        if(m_data->lastLabelImage.at<int>(x,y)>countLast){
          countLast=m_data->lastLabelImage.at<int>(x,y);
        }
      }
    }

    if(countCur==0 || countLast==0){//no relabeling possible
        m_data->lastLabelImage = labelImage.clone();
        return labelImage;
    }

    std::vector<int> curAss = calculateLabelReassignment(countCur, countLast, labelImage, m_data->lastLabelImage, size);

    for(int y=0; y<size.height; y++){//reassign label
      for(int x=0; x<size.width; x++){
        if(labelImage.at<int>(x,y)>0){
          stableLabelImage.at<int>(x,y)=curAss[labelImage.at<int>(x,y)-1];
        }else{
          stableLabelImage.at<int>(x,y)=0;
        }
      }
    }

    m_data->lastLabelImage = stableLabelImage.clone();//copy image for next iteration

  }
  m_data->stabelizeCounter=1;

  return stableLabelImage;
}


Mat SegmenterUtils::calculateAdjacencyMatrix(Mat &xyzh, Mat &labelImage,
                          Mat &maskImage, int radius, float euclideanDistance, int numSurfaces){
  //math::DynMatrix<bool> adjacencyMatrix;
  Mat adjacencyMatrix;
  if(m_data->useCL==true && m_data->clReady==true){
    adjacencyMatrix=edgePointAssignmentAndAdjacencyMatrixCL(xyzh, labelImage, maskImage, radius, euclideanDistance, numSurfaces, false);
  }else{
    adjacencyMatrix=edgePointAssignmentAndAdjacencyMatrixCPU(xyzh, labelImage, maskImage, radius, euclideanDistance, numSurfaces, false);
  }
  return adjacencyMatrix;
}


void SegmenterUtils::edgePointAssignment(Mat &xyzh, Mat &labelImage,
                          Mat &maskImage, int radius, float euclideanDistance, int numSurfaces){
  //math::DynMatrix<bool> adjacencyMatrix;
  Mat adjacencyMatrix;
  if(m_data->useCL==true && m_data->clReady==true){
    adjacencyMatrix=edgePointAssignmentAndAdjacencyMatrixCL(xyzh, labelImage, maskImage, radius, euclideanDistance, numSurfaces, true);
  }else{
    adjacencyMatrix=edgePointAssignmentAndAdjacencyMatrixCPU(xyzh, labelImage, maskImage, radius, euclideanDistance, numSurfaces, true);
  }
}


Mat SegmenterUtils::edgePointAssignmentAndAdjacencyMatrix(Mat &xyzh, Mat &labelImage,
                          Mat &maskImage, int radius, float euclideanDistance, int numSurfaces){
  Mat adjacencyMatrix;
  if(m_data->useCL==true && m_data->clReady==true){
    adjacencyMatrix=edgePointAssignmentAndAdjacencyMatrixCL(xyzh, labelImage, maskImage, radius, euclideanDistance, numSurfaces, true);
  }else{
    adjacencyMatrix=edgePointAssignmentAndAdjacencyMatrixCPU(xyzh, labelImage, maskImage, radius, euclideanDistance, numSurfaces, true);
  }
  return adjacencyMatrix;
}


std::vector<std::vector<int> > SegmenterUtils::extractSegments(Mat &labelImage){
  int h=labelImage.size().height;
  int w=labelImage.size().width;
  //core::Channel32s labelImageC = labelImage[0];
  std::vector<std::vector<int> > segments;
  for(int y=0; y<h; y++){
    for(int x=0; x<w; x++){
      if(labelImage.at<int>(x,y)>(int)segments.size()){
        segments.resize(labelImage.at<int>(x,y));
      }
      if(labelImage.at<int>(x,y)>0){
        segments.at(labelImage.at<int>(x,y)-1).push_back(x+y*w);
      }
    }
  }
  return segments;
}


void SegmenterUtils::relabel(Mat &labelImage, std::vector<std::vector<int> > &assignment, int maxOldLabel){
  std::vector<int> mapping;
  if(maxOldLabel>0){
    mapping.resize(maxOldLabel,0);
  }else{
    int maxLabel=0;
    for(unsigned int i=0; i<assignment.size(); i++){
      for(unsigned int j=0; j<assignment[i].size(); j++){
        if(assignment[i][j]+1>maxLabel){
          maxLabel=assignment[i][j]+1;//assignment [0..n-1], label [1..n]
        }
      }
    }
    mapping.resize(maxLabel,0);
  }
  for(unsigned int i=0; i<assignment.size(); i++){//calculate mapping
    for(unsigned int j=0; j<assignment[i].size(); j++){
      mapping[assignment[i][j]]=i;
    }
  }
  int w = labelImage.size().width;
  int h = labelImage.size().height;
  //core::Channel32s labelImageC = labelImage[0];
  for(int y=0; y<h; y++){//map
    for(int x=0; x<w; x++){
      if(labelImage.at<int>(x,y)>0){
        labelImage.at<int>(x,y)=mapping[labelImage.at<int>(x,y)-1]+1;
      }
    }
  }
}


bool SegmenterUtils::occlusionCheck(Mat &depthImage, Point p1, Point p2, float distanceTolerance, float outlierTolerance){
  //core::Channel32f depthImageC = depthImage[0];
  bool sampleX=false;//over x or y
  int step=0;//positive or negative
  float startValue=depthImage.at<float>(p1.x,p1.y);
  float endValue=depthImage.at<float>(p2.x,p2.y);
  float depthGradient, gradient;

  if(abs(p2.x-p1.x)>abs(p2.y-p1.y)){//sample x init
    sampleX=true;
    gradient = (float)(p2.y-p1.y)/(float)(p2.x-p1.x);
    depthGradient=(endValue-startValue)/(p2.x-p1.x);
    if(p2.x-p1.x>0){
      step=1;
    }else{
      step=-1;
    }
  }else{//sample y init
    sampleX=false;
    gradient = (float)(p2.x-p1.x)/(float)(p2.y-p1.y);
    depthGradient=(endValue-startValue)/(p2.y-p1.y);
    if(p2.y-p1.y>0){
      step=1;
    }else{
      step=-1;
    }
  }

  int numReject=0;
  if(sampleX){//sample x process
    for(int i=p1.x; (i-p2.x)*step<=0; i+=step){
      int newY=(int)round(p1.y+(i-p1.x)*gradient);
      float realValue = depthImage.at<float>(i,newY);
      float augmentedValue = startValue+(i-p1.x)*depthGradient;
      float s1 = realValue-augmentedValue;//minus -> real closer than augmented
      if(s1-distanceTolerance>0 && depthImage.at<float>(i,newY)!=2047){//not occluding
        numReject++;
      }
    }
    if((float)numReject/(float)(abs(p2.x-p1.x)+1)>outlierTolerance/100.){
      return false;
    }
  }else{//sample y process
    for(int i=p1.y; (i-p2.y)*step<=0; i+=step){
      int newX=(int)round(p1.x+(i-p1.y)*gradient);
      float realValue = depthImage.at<float>(newX,i);
      float augmentedValue = startValue+(i-p1.y)*depthGradient;
      float s1 = realValue-augmentedValue;
      if(s1-distanceTolerance>0 && depthImage.at<float>(newX,i)!=2047){
        numReject++;
      }
    }
    if((float)numReject/(float)(abs(p2.y-p1.y)+1)>outlierTolerance/100.){
      return false;
    }
  }
  return true;
}


std::vector<std::vector<int> > SegmenterUtils::createLabelVectors(Mat &labelImage){
  Size s = labelImage.size();
  //core::Channel32s labelImageC = labelImage[0];
  std::vector<std::vector<int> > labelVector;
  for(int y=0; y<s.height; y++){
    for( int x=0; x<s.width; x++){
      int id = x+y*s.width;
      if(labelImage.at<int>(x,y)>0){
        if(labelImage.at<int>(x,y)>(int)labelVector.size()){
          labelVector.resize(labelImage.at<int>(x,y));
        }
        labelVector[labelImage.at<int>(x,y)-1].push_back(id);
      }
    }
  }
  return labelVector;
}


void SegmenterUtils::createColorImageCL(Mat &labelImage, Mat &colorImage){
  #ifdef ICL_HAVE_OPENCL
    Size s = labelImage.size();
    if(s!=m_data->size || m_data->kernelSegmentColoringInitialized==false){//reinit
      m_data->size = s;
      int w = s.width;
      int h = s.height;

      m_data->segmentColorImageRArray.resize(w*h);
      m_data->segmentColorImageGArray.resize(w*h);
      m_data->segmentColorImageBArray.resize(w*h);
      m_data->segmentColorImageRBuffer = m_data->program.createBuffer("rw", w*h * sizeof(unsigned char));
      m_data->segmentColorImageGBuffer = m_data->program.createBuffer("rw", w*h * sizeof(unsigned char));
      m_data->segmentColorImageBBuffer = m_data->program.createBuffer("rw", w*h * sizeof(unsigned char));
      m_data->assignmentBuffer = m_data->program.createBuffer("r", w*h * sizeof(int));

      m_data->kernelSegmentColoringInitialized=true;
    }

    try {
      int w = m_data->size.width;
      int h = m_data->size.height;
      m_data->assignmentBuffer.write(labelImage.begin(0),w*h*sizeof(int));
      m_data->kernelSegmentColoring.setArgs(m_data->assignmentBuffer,
          m_data->segmentColorImageRBuffer,
          m_data->segmentColorImageGBuffer,
          m_data->segmentColorImageBBuffer);

      m_data->kernelSegmentColoring.apply(w*h);
      m_data->segmentColorImageRBuffer.read(&m_data->segmentColorImageRArray[0], w*h * sizeof(unsigned char));
      m_data->segmentColorImageGBuffer.read(&m_data->segmentColorImageGArray[0], w*h * sizeof(unsigned char));
      m_data->segmentColorImageBBuffer.read(&m_data->segmentColorImageBArray[0], w*h * sizeof(unsigned char));

      std::vector<icl8u*> data(3);
      data[0] = m_data->segmentColorImageRArray.data();
      data[1] = m_data->segmentColorImageGArray.data();
      data[2] = m_data->segmentColorImageBArray.data();
      colorImage = core::Img8u(utils::Size(w,h),3,data,false);

    } catch (utils::CLException &err) { //catch openCL errors
      ERROR_LOG(err.what() );
    }
  #endif
}


void SegmenterUtils::createColorImageCPU(Mat &labelImage, Mat &colorImage){
  Size s = labelImage.size();
  //colorImage.size(s);
  //colorImage.setChannels(3);
  Mat ColorImage(s, CV_8UC3);
  for (int y = 0; y < s.height; y++) {
    for (int x = 0; x < s.width; x++) {
      if (labelImage.at<int>(x,y,0) == 0) {
        colorImage.at<int>(x, y, 0) = 128;
        colorImage.at<int>(x, y, 1) = 128;
        colorImage.at<int>(x, y, 2) = 128;
      } else {
        int H = (int) (labelImage.at<int>(x,y,0) * 35.) % 360;
        float S = 1.0 - labelImage.at<int>(x,y,0) * 0.01;
        float hi = floor((float) H / 60.);
        float f = ((float) H / 60.) - hi;
        float pp = 1.0 - S;
        float qq = 1.0 - S * f;
        float tt = 1.0 - S * (1. - f);
        float newR = 0;
        float newG = 0;
        float newB = 0;
        if ((int) hi == 0 || (int) hi == 6) {
          newR = 1.0;
          newG = tt;
          newB = pp;
        } else if ((int) hi == 1) {
          newR = qq;
          newG = 1.0;
          newB = pp;
        } else if ((int) hi == 2) {
          newR = pp;
          newG = 1.0;
          newB = tt;
        } else if ((int) hi == 3) {
          newR = pp;
          newG = qq;
          newB = 1.0;
        } else if ((int) hi == 4) {
          newR = tt;
          newG = pp;
          newB = 1.0;
        } else if ((int) hi == 5) {
          newR = 1.0;
          newG = pp;
          newB = qq;
        }
        colorImage.at<int>(x, y, 0) = (unsigned char) (newR * 255.);
        colorImage.at<int>(x, y, 1) = (unsigned char) (newG * 255.);
        colorImage.at<int>(x, y, 2) = (unsigned char) (newB * 255.);
      }
    }
  }

}


std::vector<int> SegmenterUtils::calculateLabelReassignment(int countCur, int countLast, Mat &labelImageC, Mat &lastLabelImageC, Size size){
  //math::DynMatrix<int> assignmentMatrix(countCur,countLast,0);
  Mat assignmentMatrix(countCur,countLast,0);
  std::vector<int> lastNum(countLast,0);
  std::vector<int> curNum(countCur,0);
  std::vector<int> curAss(countCur,0);
  std::vector<float> curVal(countCur,0);

  for(int y=0; y<size.height; y++){//count overlap points (cross-correlated)
    for(int x=0; x<size.width; x++){
      if(labelImageC.at<int>(x,y)>0 && lastLabelImageC.at<int>(x,y)>0){
        assignmentMatrix.at<int>(labelImageC.at<int>(x,y)-1, lastLabelImageC.at<int>(x,y)-1)++;//num match points
        lastNum[lastLabelImageC.at<int>(x,y)-1]++;//num segment points
        curNum[labelImageC.at<int>(x,y)-1]++;//num segment points
      }
    }
  }

  for(int i=0; i<countCur; i++){//calculate assignment score
    for(int j=0; j<countLast; j++){
      float curScore;
      float lastScore;
      float compScore;
      if(assignmentMatrix.at<int>(i,j)>0){
        curScore=(float)assignmentMatrix.at<int>(i,j)/(float)curNum[i];
        lastScore=(float)assignmentMatrix.at<int>(i,j)/(float)lastNum[j];
        compScore=(curScore+lastScore)/2.;
      }
      else{
        curScore=0;
        lastScore=0;
        compScore=0;
      }

      if(curVal[i]<compScore){//assign highest score
        curVal[i]=compScore;
        curAss[i]=j+1;
      }
    }
  }

  std::vector<bool> empties(countCur,true);//find unassigned ids
  for(int i=0; i<countCur; i++){
    if(curAss[i]!=0){
      //empties[curAss[i]-1]=false;
      empties[i]=false;
    }
    for(int j=i+1; j<countCur; j++){//multiple use of id (use highest score)
      if(curAss[i]==curAss[j]){
        if(curVal[i]<curVal[j]){
          curAss[i]=0;
        }else{
          curAss[j]=0;
        }
      }
    }
  }

  for(int i=0; i<countCur; i++){//assign unassigned ids to first empty id
    for(int j=0; j<countCur; j++){
      if(curAss[i]==0 && empties[j]==true){
        empties[j]=false;
        curAss[i]=j+1;
        break;
      }
    }
  }

  return curAss;
}


Mat SegmenterUtils::edgePointAssignmentAndAdjacencyMatrixCL(Mat &xyzh, Mat &labelImage,
                          Mat &maskImage, int radius, float euclideanDistance, int numSurfaces, bool pointAssignment){
  #ifdef ICL_HAVE_OPENCL
    utils::Size s = labelImage.getSize();
    math::DynMatrix<bool> neighbours(numSurfaces,numSurfaces,false);
    math::DynMatrix<unsigned char> neighboursC(numSurfaces,numSurfaces,(unsigned char)0);
    if(s!=m_data->size || m_data->kernelPointAssignmentInitialized==false){//reinit
      m_data->size = s;
      int w = s.width;
      int h = s.height;

      m_data->maskArray.resize(w*h);
      m_data->assignmentArray.resize(w*h);

      m_data->assignmentBuffer = m_data->program.createBuffer("r", w*h * sizeof(int));
      m_data->assignmentOutBuffer = m_data->program.createBuffer("rw", w*h * sizeof(int));
      m_data->xyzBuffer = m_data->program.createBuffer("r", w*h * sizeof(Vec));
      m_data->maskBuffer = m_data->program.createBuffer("rw", w*h * sizeof(unsigned char));

      m_data->kernelPointAssignmentInitialized=true;
    }

    try {
      int w = s.width;
      int h = s.height;

      core::Img32s labelImageOut(labelImage.getSize(),1,core::formatMatrix);

      m_data->assignmentBuffer.write(labelImage.begin(0),w*h*sizeof(int));
      m_data->maskBuffer.write(maskImage.begin(0),w*h*sizeof(unsigned char));
      m_data->xyzBuffer.write(&xyzh[0][0],w*h*sizeof(Vec));//FixedColVector<float, 4>));

      m_data->neighboursBuffer = m_data->program.createBuffer("rw", numSurfaces*numSurfaces * sizeof(unsigned char), &neighboursC[0]);

      m_data->kernelPointAssignment.setArgs(m_data->xyzBuffer,
          m_data->maskBuffer,
          m_data->assignmentBuffer,
          radius,
          numSurfaces,
          m_data->neighboursBuffer,
          m_data->assignmentOutBuffer,
          w,
          h,
          euclideanDistance);
      m_data->kernelPointAssignment.apply(w,h);
      m_data->neighboursBuffer.read(neighboursC.data(),
          numSurfaces*numSurfaces * sizeof(unsigned char));
      for(unsigned int i=0; i<neighboursC.rows(); i++){
        for(unsigned int j=0; j<neighboursC.cols(); j++){
          neighbours(i,j)=(bool)neighboursC(i,j);
        }
      }
      if(pointAssignment){
        m_data->assignmentOutBuffer.read(m_data->assignmentArray.data(), w*h * sizeof(int));
        labelImage = core::Img32s(utils::Size(w,h),1,std::vector<int*>(1,m_data->assignmentArray.data()),false);
        m_data->maskBuffer.read(m_data->maskArray.data(), w*h * sizeof(unsigned char));
        maskImage = core::Img8u(utils::Size(w,h),1,std::vector<unsigned char*>(1,m_data->maskArray.data()),false);
      }
      for(int i=0; i<numSurfaces; i++) {
        neighbours(i,i)=true;
      }
    } catch (utils::CLException &err) { //catch openCL errors
      ERROR_LOG(err.what());
    }

    return neighbours;
  #else
    return Mat();
  #endif
}


Mat SegmenterUtils::edgePointAssignmentAndAdjacencyMatrixCPU(Mat &xyzh, Mat &labelImage,
                          Mat &maskImage, int radius, float euclideanDistance, int numSurfaces, bool pointAssignment){
  Size s = labelImage.size();
  int w = s.width;
  int h = s.height;
  Mat neighbours(numSurfaces, numSurfaces, false);
  //core::Img32s labelImageOut(labelImage.getSize(),1,core::formatMatrix);
  Mat labelImageOut(s, CV_32SC1);

  //core::Channel32s labelImageC = labelImage[0];
  //core::Channel32s labelImageOutC = labelImageOut[0];
  //core::Channel8u maskImageC = maskImage[0];

  for (int x=0; x<w; x++) {
    for(int y=0; y<h; y++) {
      int i=x+w*y;
      float dist=100000;
      int ass=0;
      bool assigned=false;
      if(maskImage.at<int>(x,y)==0 && labelImage.at<int>(x,y)==0) {
        std::vector<bool> adj(numSurfaces,false);
        for(int xx=-radius; xx<=radius; xx++) {
          for(int yy=-radius; yy<=radius; yy++) {
            if(x+xx>=0 && x+xx<w && y+yy>=0 && y+yy<h && labelImage.at<int>(x+xx,y+yy)!=0) {
              Point3f p1=xyzh.at<Point3f>(y,x);
              Point3f p2=xyzh.at<Point3f>(y+yy,x+xx);
              float distance=dist3(p1, p2);
              if(distance<euclideanDistance) {
                adj[labelImage.at<int>(x+xx,y+yy)-1]=true;
              }
              if(distance<dist && distance<euclideanDistance) {
                dist=distance;
                ass=labelImage.at<int>(x+xx,y+yy);
                assigned=true;
              }
            }
          }
        }
        for(int a=0; a<numSurfaces-1; a++) {
          for (int b=a+1; b<numSurfaces; b++) {
            if(adj[a]==1 && adj[b]==1) {
              neighbours.at<int>(a,b)=1;
              neighbours.at<int>(b,a)=1;
            }
          }
        }
        if(pointAssignment){
          if(assigned==1) {
            maskImage.at<int>(x,y)=1;
            labelImageOut.at<int>(x,y)=ass;
          }
          else {
            labelImageOut.at<int>(x,y)=labelImage.at<int>(x,y);
          }
        }
      }
      else {
        labelImageOut.at<int>(x,y)=labelImage.at<int>(x,y);
      }
    }
  }
  for (int i = 0; i < numSurfaces; i++) {
    neighbours.at<int>(i, i) = 1;
  }
  if(pointAssignment==true) {
    labelImage = labelImageOut.clone();
  }
  return neighbours;
}
