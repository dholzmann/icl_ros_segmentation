/***********************************************************************
 **                Image Component Library (ICL)                       **
 **                                                                    **
 ** Copyright (C) 2006-2013 CITEC, University of Bielefeld             **
 **                         Neuroinformatics Group                     **
 ** Website: www.iclcv.org and                                         **
 **          http://opensource.cit-ec.de/projects/icl                  **
 **                                                                    **
 ** File   : ICLGeom/src/ICLGeom/ConfigurableDepthImageSegmenter.cpp   **
 ** Module : ICLGeom                                                   **
 ** Authors: Andre Ueckermann                                          **
 **                                                                    **
 **                                                                    **
 ** GNU LESSER GENERAL PUBLIC LICENSE                                  **
 ** This file may be used under the terms of the GNU Lesser General    **
 ** Public License version 3.0 as published by the                     **
 **                                                                    **
 ** Free Software Foundation and appearing in the file LICENSE.LGPL    **
 ** included in the packaging of this file.  Please review the         **
 ** following information to ensure the license requirements will      **
 ** be met: http://www.gnu.org/licenses/lgpl-3.0.txt                   **
 **                                                                    **
 ** The development of this software was supported by the              **
 ** Excellence Cluster EXC 277 Cognitive Interaction Technology.       **
 ** The Excellence Cluster EXC 277 is a grant of the Deutsche          **
 ** Forschungsgemeinschaft (DFG) in the context of the German          **
 ** Excellence Initiative.                                             **
 **                                                                    **
 ***********************************************************************/

#include <ConfigurableDepthImageSegmenter.h>
#include <FeatureGraphSegmenter.h>
//#include <ICLGeom/PointCloudCreator.h>
#include <ICLGeom/ObjectEdgeDetector.h>
//#include <ICLFilter/MotionSensitiveTemporalSmoothing.h>

namespace icl{
  namespace geom{  
    
    struct ConfigurableDepthImageSegmenter::Data {
				Data(Mode mode, Camera depthCam, Camera colorCam,
						 PointCloudCreator::DepthImageMode depth_mode){
            depthCamera=depthCam;
						//creator = new PointCloudCreator(depthCam, colorCam, depth_mode);
						use_extern_edge_image = false;
            init(mode);
	    }

				Data(Mode mode, Camera depthCam,
						 PointCloudCreator::DepthImageMode depth_mode){
            depthCamera=depthCam;
						//creator = new PointCloudCreator(depthCam, depth_mode);
						use_extern_edge_image = false;
            init(mode);
        }

	    ~Data() {
	    }
      
      void init(Mode mode) {
          //temporalSmoothing = new filter::MotionSensitiveTemporalSmoothing(2047, 15);

          if(mode==CPU){
            //temporalSmoothing->setUseCL(false);
            objectEdgeDetector = new ObjectEdgeDetector(ObjectEdgeDetector::CPU);
            segmentation = new FeatureGraphSegmenter(FeatureGraphSegmenter::CPU);
          }else{
            //temporalSmoothing->setUseCL(true);
            objectEdgeDetector = new ObjectEdgeDetector(ObjectEdgeDetector::BEST);
            segmentation = new FeatureGraphSegmenter(FeatureGraphSegmenter::BEST);
          }
      }

      //PointCloudCreator *creator;
      ObjectEdgeDetector *objectEdgeDetector;
      //filter::MotionSensitiveTemporalSmoothing *temporalSmoothing;
      FeatureGraphSegmenter *segmentation;
      
      Camera depthCamera;
      
      core::Img8u edgeImage;
      core::Img8u normalImage;

			bool use_extern_edge_image;
			
			std::vector<std::pair<utils::Point,utils::Point> > boundingBoxes2D;
    };
    
    
		ConfigurableDepthImageSegmenter::ConfigurableDepthImageSegmenter(Mode mode, Camera depthCam,
																																		 PointCloudCreator::DepthImageMode depth_mode) :
				m_data(new Data(mode, depthCam, depth_mode)){
        initProperties();
    }

		ConfigurableDepthImageSegmenter::ConfigurableDepthImageSegmenter(Mode mode, Camera depthCam, Camera colorCam,
																																		 icl::geom::PointCloudCreator::DepthImageMode depth_mode) :
					m_data(new Data(mode, depthCam, colorCam, depth_mode)){
        initProperties();
    }
  	
    void ConfigurableDepthImageSegmenter::initProperties() {
        addProperty("general.enable segmentation","flag","",true);
        addProperty("general.stabelize segmentation","flag","",true);
        addProperty("general.depth scaling","range","[0.9,1.1]",1.05);
        addProperty("general.use ROI","flag","",false);
        addProperty("general.ROI min x","range","[-1500,500]",-1200);
        addProperty("general.ROI max x","range","[-500,1500]",1200);
        addProperty("general.ROI min y","range","[-1500,800]",-100);
        addProperty("general.ROI max y","range","[-500,1500]",1050);
        addProperty("general.ROI min z","range","[-500,1500]",0);
        addProperty("general.ROI max z","range","[0,2000]",1050);
        
        addProperty("general.img corner ll","range","[0,200]",0);
        addProperty("general.img corner lr","range","[0,200]",0);
        addProperty("general.img corner ul","range","[0,200]",0);
        addProperty("general.img corner ur","range","[0,200]",0);

        //addProperty("pre.enable temporal smoothing","flag","",true);
        //addProperty("pre.temporal smoothing size","range","[1,15]:1",6);
        //addProperty("pre.temporal smoothing diff","range","[1,22]:1",10);
        addProperty("pre.filter","menu","unfiltered,median3x3,median5x5","median3x3");
        addProperty("pre.normal range","range","[1,15]:1",1);
        addProperty("pre.averaging","flag","",true);
        addProperty("pre.averaging range","range","[1,15]:1",2);
        addProperty("pre.smoothing","menu","linear,gaussian","linear");
        addProperty("pre.edge threshold","range","[0.7,1]",0.89);
        addProperty("pre.edge angle method","menu","max,mean","mean");
        addProperty("pre.edge neighborhood","range","[1,15]",1);

        addProperty("surfaces.min surface size","range","[5,75]:1",50);
        addProperty("surfaces.assignment radius","range","[2,15]:1",7);
        addProperty("surfaces.assignment distance","range","[3.0,25.0]",15.0);

        addProperty("cutfree.enable cutfree adjacency feature","flag","",true);
        addProperty("cutfree.ransac euclidean distance","range","[2.0,20.0]",8.0);
        addProperty("cutfree.ransac passes","range","[5,50]:1",20);
        addProperty("cutfree.ransac tolerance","range","[5,50]:1",30);
        addProperty("cutfree.min angle","range","[0.0,70.0]",30.0);

        addProperty("coplanarity.enable coplanarity feature","flag","",true);
        addProperty("coplanarity.max angle","range","[0.0, 60.0]",30.0);
        addProperty("coplanarity.distance tolerance","range","[1.0,10.0]",3.0);
        addProperty("coplanarity.outlier tolerance","range","[1.0,10.0]",5.0);
        addProperty("coplanarity.num triangles","range","[10,50]:1",20);
        addProperty("coplanarity.num scanlines","range","[1,20]:1",9);

        addProperty("curvature.enable curvature feature","flag","",true);
        addProperty("curvature.histogram similarity","range","[0.1,1.0]",0.5);
        addProperty("curvature.enable open objects","flag","",true);
        addProperty("curvature.max distance","range","[1,20]:1",10);
        addProperty("curvature.enable occluded objects","flag","",true);
        addProperty("curvature.max error","range","[1.0,20.0]",10.0);
        addProperty("curvature.ransac passes","range","[5,50]:1",20);
        addProperty("curvature.distance tolerance","range","[1.0,10.0]",3.0);
        addProperty("curvature.outlier tolerance","range","[1.0,10.0]",5.0);

        addProperty("remaining.enable remaining points feature","flag","",true);
        addProperty("remaining.min size","range","[5,50]:1",10);
        addProperty("remaining.euclidean distance","range","[2.0,20.0]",10.0);
        addProperty("remaining.radius","range","[0,10]:1",0);
        addProperty("remaining.assign euclidean distance","range","[2.0,20.0]",10.0);
        addProperty("remaining.support tolerance","range","[0,30]:1",9);

        addProperty("graphcut.threshold","range","[0.0, 1.1]",0.5);

        setConfigurableID("segmentation");
    }

  	
    ConfigurableDepthImageSegmenter::~ConfigurableDepthImageSegmenter(){
      delete m_data;
    }

    		
    void ConfigurableDepthImageSegmenter::apply(const core::Img32f &depthImage, PointCloudObject &obj){// const core::Img8u &colorImage){

    /*  //pre segmentation
      m_data->temporalSmoothing->setFilterSize(getPropertyValue("pre.temporal smoothing size"));
      m_data->temporalSmoothing->setDifference(getPropertyValue("pre.temporal smoothing diff"));
      static core::ImgBase *filteredImage = 0;
      bool useTempSmoothing = getPropertyValue("pre.enable temporal smoothing");
      if(useTempSmoothing==true){//temporal smoothing
        m_data->temporalSmoothing->apply(&depthImage,&filteredImage);    
      }
		*/
      std::string usedFilter=getPropertyValue("pre.filter");
      bool useAveraging=getPropertyValue("pre.averaging");
      std::string usedAngle=getPropertyValue("pre.edge angle method");
      std::string usedSmoothing=getPropertyValue("pre.smoothing");
      int normalrange = getPropertyValue("pre.normal range");
      int neighbrange = getPropertyValue("pre.edge neighborhood");
      float threshold = getPropertyValue("pre.edge threshold");
      int avgrange = getPropertyValue("pre.averaging range");
      
      bool usedFilterFlag=true;
      if(usedFilter.compare("median3x3")==0){ //median 3x3
        m_data->objectEdgeDetector->setMedianFilterSize(3);
      }
      else if(usedFilter.compare("median5x5")==0){ //median 5x5
        m_data->objectEdgeDetector->setMedianFilterSize(5);
      }else{
        usedFilterFlag=false;
      }
      
      m_data->objectEdgeDetector->setNormalCalculationRange(normalrange);	
      m_data->objectEdgeDetector->setNormalAveragingRange(avgrange);	
		 
      if(usedAngle.compare("max")==0){//max
        m_data->objectEdgeDetector->setAngleNeighborhoodMode(0);
      }
      else if(usedAngle.compare("mean")==0){//mean
        m_data->objectEdgeDetector->setAngleNeighborhoodMode(1);
      }
      
      bool usedSmoothingFlag = true;
      if(usedSmoothing.compare("linear")==0){//linear
        usedSmoothingFlag = false;
      }
      else if(usedSmoothing.compare("gaussian")==0){//gauss
        usedSmoothingFlag = true;
      }

      m_data->objectEdgeDetector->setAngleNeighborhoodRange(neighbrange);
      m_data->objectEdgeDetector->setBinarizationThreshold(threshold);
		 
			if (!m_data->use_extern_edge_image) {
				/*if(useTempSmoothing==true){
					m_data->edgeImage=m_data->objectEdgeDetector->calculate(*filteredImage->as32f(), usedFilterFlag,
																								 useAveraging, usedSmoothingFlag);
				}else{*/
					m_data->edgeImage=m_data->objectEdgeDetector->calculate(*depthImage.as32f(), usedFilterFlag,
																								 useAveraging, usedSmoothingFlag);
				//}
				m_data->objectEdgeDetector->applyWorldNormalCalculation(m_data->depthCamera);
				m_data->normalImage=m_data->objectEdgeDetector->getRGBNormalImage();
			}
		 

      obj.lock();
      
      //create pointcloud
      //float depthScaling=getPropertyValue("general.depth scaling");
   //   GeomColor c(1.,0.,0.,1.);
	 //   obj.selectRGBA32f().fill(c);
     /* if(useTempSmoothing==true){
        m_data->creator->create(*filteredImage->as32f(), obj, 0, depthScaling);
      }else{
        m_data->creator->create(*depthImage.as32f(), obj, 0, depthScaling);
      }  
     */ 
      //high level segmenation
      bool enableSegmentation = getPropertyValue("general.enable segmentation");
   
      if(enableSegmentation){
        bool useROI = getPropertyValue("general.use ROI");      
        bool stabelizeSegmentation = getPropertyValue("general.stabelize segmentation");
                    
        int surfaceMinSize = getPropertyValue("surfaces.min surface size");
        int surfaceRadius = getPropertyValue("surfaces.assignment radius");
        float surfaceDistance = getPropertyValue("surfaces.assignment distance");
        
        bool cutfreeEnable = getPropertyValue("cutfree.enable cutfree adjacency feature");
        float cutfreeEuclDist = getPropertyValue("cutfree.ransac euclidean distance");
        int cutfreePasses = getPropertyValue("cutfree.ransac passes");
        int cutfreeTolerance = getPropertyValue("cutfree.ransac tolerance");
        float cutfreeMinAngle = getPropertyValue("cutfree.min angle");
        
        bool coplanEnable = getPropertyValue("coplanarity.enable coplanarity feature");
        float coplanMaxAngle = getPropertyValue("coplanarity.max angle");
        float coplanDistance = getPropertyValue("coplanarity.distance tolerance");
        float coplanOutlier = getPropertyValue("coplanarity.outlier tolerance");
        int coplanNumTriangles = getPropertyValue("coplanarity.num triangles");
        int coplanNumScanlines = getPropertyValue("coplanarity.num scanlines");
        
        bool curveEnable = getPropertyValue("curvature.enable curvature feature");
        float curveHistogram = getPropertyValue("curvature.histogram similarity");
        bool curveUseOpen = getPropertyValue("curvature.enable open objects");
        int curveMaxDist = getPropertyValue("curvature.max distance");
        bool curveUseOccluded = getPropertyValue("curvature.enable occluded objects");
        float curveMaxError = getPropertyValue("curvature.max error");
        int curvePasses = getPropertyValue("curvature.ransac passes");
        float curveDistance = getPropertyValue("curvature.distance tolerance");
        float curveOutlier = getPropertyValue("curvature.outlier tolerance");
        
        bool remainingEnable = getPropertyValue("remaining.enable remaining points feature");
        int remainingMinSize = getPropertyValue("remaining.min size");
        float remainingEuclDist = getPropertyValue("remaining.euclidean distance");
        int remainingRadius = getPropertyValue("remaining.radius");
        float remainingAssignEuclDist = getPropertyValue("remaining.assign euclidean distance");
        int remainingSupportTolerance = getPropertyValue("remaining.support tolerance");
        
        float graphcutThreshold = getPropertyValue("graphcut.threshold");     
      
        if(useROI){
          m_data->segmentation->setROI(getPropertyValue("general.ROI min x"),
                                      getPropertyValue("general.ROI max x"),
                                      getPropertyValue("general.ROI min y"),
                                      getPropertyValue("general.ROI max y"),
                                      getPropertyValue("general.ROI min z"),
                                      getPropertyValue("general.ROI max z"));
                             
        }
        
        m_data->segmentation->setCornerRemoval(getPropertyValue("general.img corner ll"),
                                               getPropertyValue("general.img corner lr"),
                                               getPropertyValue("general.img corner ul"),
                                               getPropertyValue("general.img corner ur"));
        
        m_data->segmentation->setMinSurfaceSize(surfaceMinSize);
        m_data->segmentation->setAssignmentParams(surfaceDistance, surfaceRadius);
        m_data->segmentation->setCutfreeParams(cutfreeEuclDist, cutfreePasses, cutfreeTolerance, cutfreeMinAngle);
        m_data->segmentation->setCoplanarityParams(coplanMaxAngle, coplanDistance, coplanOutlier, coplanNumTriangles, coplanNumScanlines);
        m_data->segmentation->setCurvatureParams(curveHistogram, curveUseOpen, curveMaxDist, curveUseOccluded, curveMaxError, 
                                curvePasses, curveDistance, curveOutlier);
        m_data->segmentation->setRemainingPointsParams(remainingMinSize, remainingEuclDist, remainingRadius, remainingAssignEuclDist, remainingSupportTolerance);
        m_data->segmentation->setGraphCutThreshold(graphcutThreshold);
        
			/*	if(useTempSmoothing){
					core::Img8u lI=m_data->segmentation->apply(obj.selectXYZH(),m_data->edgeImage,*filteredImage->as32f(), m_data->objectEdgeDetector->getNormals(),
																										 stabelizeSegmentation, useROI, cutfreeEnable, coplanEnable, curveEnable, remainingEnable);
//					obj.setColorsFromImage(lI);
        }else{*/
					core::Img8u lI=m_data->segmentation->apply(obj.selectXYZH(), m_data->edgeImage, *depthImage.as32f(), m_data->objectEdgeDetector->getNormals(),
																										 stabelizeSegmentation, useROI, cutfreeEnable, coplanEnable, curveEnable, remainingEnable);
//					obj.setColorsFromImage(lI);
        //}
      } 
      
    obj.unlock();  
    }

	std::vector<geom::SurfaceFeatureExtractor::SurfaceFeature>
	ConfigurableDepthImageSegmenter::getSurfaceFeatures() {
		return m_data->segmentation->getSurfaceFeatures();
	}

    const core::DataSegment<float,4> ConfigurableDepthImageSegmenter::getNormalSegment() {
		return m_data->objectEdgeDetector->getNormals();
    }

	const core::Img32f ConfigurableDepthImageSegmenter::getAngleImage() {
		return m_data->objectEdgeDetector->getAngleImage();
	}

    const core::Img8u ConfigurableDepthImageSegmenter::getNormalImage(){
      return m_data->normalImage;
    }
    
	  	
    const core::Img8u ConfigurableDepthImageSegmenter::getEdgeImage(){
      return m_data->edgeImage;
    }
    
	  
	  core::Img32s ConfigurableDepthImageSegmenter::getLabelImage(){
	    return m_data->segmentation->getLabelImage(getPropertyValue("general.stabelize segmentation"));
	  }
	  	
	  	
    core::Img8u ConfigurableDepthImageSegmenter::getColoredLabelImage(){
      return m_data->segmentation->getColoredLabelImage(getPropertyValue("general.stabelize segmentation"));
    }

    core::Img8u ConfigurableDepthImageSegmenter::getMappedColorImage(const core::Img8u &image) {
        core::Img8u mapped(image.getParams());
     //   if (!m_data->creator->hasColorCamera())
     //       return mapped;
     //   m_data->creator->mapImage(&image,bpp(mapped));
        return mapped;
    }

	void ConfigurableDepthImageSegmenter::mapImageToDepth(const core::ImgBase *src, core::ImgBase **dst) {
		//m_data->creator->mapImage(src,dst);
	}

	void ConfigurableDepthImageSegmenter::setNormals(core::DataSegment<float,4> &normals) {
		m_data->objectEdgeDetector->setNormals(normals);
	}

	void ConfigurableDepthImageSegmenter::setEdgeSegData(core::Img8u &edges, core::Img8u &normal_img) {
		m_data->edgeImage = edges;
		m_data->normalImage = normal_img;
	}

	void ConfigurableDepthImageSegmenter::setUseExternalEdges(bool use_external_edges) {
		m_data->use_extern_edge_image = use_external_edges;
	}
    
    std::vector<std::vector<int> > ConfigurableDepthImageSegmenter::getSurfaces(){
      return m_data->segmentation->getSurfaces();
    }
    
    
    std::vector<std::vector<int> > ConfigurableDepthImageSegmenter::getSegments(){
      return m_data->segmentation->getSegments();
    }
    
    std::vector<PointCloudSegmentPtr> ConfigurableDepthImageSegmenter::getClusters(PointCloudObject &obj){
      //create point cloud segments (incl. leafs with data)
      std::vector<std::vector<int> > segments = m_data->segmentation->getSegments();
      std::vector<std::vector<int> > surfaces = m_data->segmentation->getSurfaces();
      core::DataSegment<float,4> xyz = obj.selectXYZH();
      core::DataSegment<float,4> rgb = obj.selectRGBA32f();
      std::vector<PointCloudSegmentPtr> pointCloudSegments(segments.size());
      
      m_data->boundingBoxes2D.resize(segments.size());
      int w = obj.getSize().width;
      for(unsigned int i=0; i<segments.size(); i++){//create segments
        //if(cutNodes[i].children.size()==0){//leaf (no children)
        int segSize=0;
        std::pair<utils::Point,utils::Point> boundingBox2D;
        boundingBox2D.first = utils::Point(1000000,1000000);
        boundingBox2D.second = utils::Point(-1000000, -1000000);
        for(unsigned int j=0; j<segments[i].size(); j++){
          segSize += surfaces[segments[i][j]].size();
        }
        pointCloudSegments[i]=new PointCloudSegment(segSize, true);
        core::DataSegment<float,4> dst_xyzh=pointCloudSegments[i]->selectXYZH();
        core::DataSegment<float,4> dst_rgba=pointCloudSegments[i]->selectRGBA32f();
        int c=0;
        for(unsigned int j=0; j<segments[i].size(); j++){
          for(unsigned int k=0; k<surfaces[segments[i][j]].size(); k++){
          	dst_xyzh[c]=xyz[surfaces[segments[i][j]][k]];
          	dst_rgba[c]=rgb[surfaces[segments[i][j]][k]];
          	int id = surfaces[segments[i][j]][k];
          	int y = (int)floor((float)id/(float)w);
				    int x = id-y*w;
				    if(x<boundingBox2D.first.x) boundingBox2D.first.x=x;
				    if(x>boundingBox2D.second.x) boundingBox2D.second.x=x;
				    if(y<boundingBox2D.first.y) boundingBox2D.first.y=y;
				    if(y>boundingBox2D.second.y) boundingBox2D.second.y=y;
          	c++;
          }
        }
        pointCloudSegments[i]->cutCost=0;//cost 0
        m_data->boundingBoxes2D[i]=boundingBox2D;
      //  return pointCloudSegments;
      }

		    //add root nodes to result
		    for(unsigned int i=0; i<segments.size(); i++){//create segments
		      //results.push_back(pointCloudSegments[i]);
		      //results.back()->updateFeatures();//new          
		      pointCloudSegments[i]->updateFeatures();
		    }
		    return pointCloudSegments;
		  }
		  
		  std::vector<std::pair<utils::Point,utils::Point> > ConfigurableDepthImageSegmenter::getBoundingBoxes2D(){
		    return m_data->boundingBoxes2D;
		  }
		  
		  int ConfigurableDepthImageSegmenter::getTableID(core::DataSegment<float,4> &xyz){
		    core::DataSegment<float,4> worldNormals = m_data->objectEdgeDetector->getWorldNormals();
		    //DataSegment<float,4> xyz = obj->selectXYZH();
		    std::vector<std::vector<int> > segments = m_data->segmentation->getSegments();
        std::vector<std::vector<int> > surfaces = m_data->segmentation->getSurfaces();
		    //std::vector<std::vector<int> > cluster = segmentation->getCluster();
		    //std::vector<std::vector<int> > blobs = segmentation->getBlobs();//
		    bool found=false;
		    int foundID=0;
		    int foundZPos=100000;
		    for(unsigned int i=0; i<segments.size(); i++){
		      float nz=0;
		      int numPoints=0;
		      float zPos=0;
		      int nInZ=0;
		      float sumZ=0;
		      for(unsigned int j=0; j<segments.at(i).size(); j++){
		        for(unsigned int k=0; k<surfaces.at(segments.at(i).at(j)).size(); k++){
		          int p = surfaces.at(segments.at(i).at(j)).at(k);
		          zPos+= xyz[p][2];
		          nz+= worldNormals[p][2];
		          numPoints++;
		          if(worldNormals[p][2]>0.9){
		            nInZ++;
		          }
		        }
		      }
		      zPos/= numPoints; //for height (minimal)
		      nz/= numPoints;//meanNormal for horizontal
		      sumZ=(float)nInZ/(float)numPoints;//percent of horizontal points (for planar)
		     
		      if(nz>0.8 && sumZ>0.7 && numPoints>10000 && zPos>-30 && zPos<30){
		     	 //std::cout<<i<<": zPos: "<<zPos<<" , normZ: "<<nz<<" , numPoints: "<<numPoints<<" , percentInZDir: "<<sumZ<<std::endl<<std::endl;
			 		
		        if(found==false){
		          found=true;
		          foundID=i;
		          foundZPos=zPos;
		        }
		        else{
		          if(zPos<foundZPos){
		            foundID=i;
		            foundZPos=zPos;
		          }
		        }	
		      }		 		
		    }
			 				
		    if(found==true){
		      //std::cout<<"found table id "<<foundID<<" at "<<foundZPos<<std::endl<<std::endl;
		      return foundID;
		    }else{
		      //std::cout<<"no table id"<<std::endl<<std::endl;
		      return -1;
		    }
		    //return -1;
		  }
		  
		  std::vector<int> ConfigurableDepthImageSegmenter::getROI(){
        std::vector<int> result;
        result.push_back(getPropertyValue("general.ROI min x"));
        result.push_back(getPropertyValue("general.ROI max x"));
        result.push_back(getPropertyValue("general.ROI min y"));
        result.push_back(getPropertyValue("general.ROI max y"));
        result.push_back(getPropertyValue("general.ROI min z"));
        result.push_back(getPropertyValue("general.ROI max z"));
        return result;
    }
     
  } // namespace geom
}
