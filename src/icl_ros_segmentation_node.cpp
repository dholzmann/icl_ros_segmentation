/********************************************************************
 **                Image Component Library (ICL)                    **
 **                                                                 **
 ** Copyright (C) 2006-2013 CITEC, University of Bielefeld          **
 **                         Neuroinformatics Group                  **
 ** Website: www.iclcv.org and                                      **
 **          http://opensource.cit-ec.de/projects/icl               **
 **                                                                 **
 ** File   : ICLGeom/demos/kinect-depth-image-segmentation/         **
 **          kinect-depth-image-segmentation.cpp                    **
 ** Module : ICLGeom                                                **
 ** Authors: Andre Ueckermann
 ** updated by Qiang Li for REBA project
 ** modified by Guillaume Walck for CLF 
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

#include "ConfigurableDepthImageSegmenter.h"
#include <ICLIO/GenericGrabber.h>
#include <ICLQt/Qt.h>
#include <ICLQt/Application.h>
#include <ICLGeom/Scene.h>
//#include <ICLQt/Common.h>
#include <Kinect.h>
#include <ICLGeom/SurfaceFeatureExtractor.h>
#include <ICLGeom/Primitive3DFilter.h>
#include <ICLGeom/PCLPointCloudObject.h>
#include <ICLFilter/MotionSensitiveTemporalSmoothing.h>
#include <ICLCore/CCFunctions.h>
#include <ICLCore/OpenCV.h>


// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
//#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <segmentation_msgs/WorldBelief.h>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

//#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

// common
#include <iostream>
#include <map>
#include <string>

using icl::utils::pa;
using icl::utils::pa_init;

using namespace icl;
using namespace icl::qt;
using namespace icl::core;
using namespace icl::geom;
using namespace icl::utils;
using namespace icl::math;
using namespace icl::filter;
using namespace icl::io;


HSplit gui;
int KINECT_CAM=0,VIEW_CAM=1;

ros::Publisher pub_pc;
ros::Publisher pub_pc2;

ros::Publisher pub_seg;

// pointers to individual modules 
SmartPtr<Kinect> kinect;
SmartPtr<ConfigurableDepthImageSegmenter> segmenter;
SmartPtr<PointCloudCreator> pointCloudCreator;

filter::MotionSensitiveTemporalSmoothing* temporalSmoothing;

// ros
ros::NodeHandle *nh;

// for primitive filter
SceneObject *primitive_holder;
std::vector<Primitive3DFilter::Primitive3D> primitives;
SmartPtr<Primitive3DFilter> primitiveFilter;
//ListenerPtr primitivesetListener;
icl::utils::Mutex primitivesMutex;

PointCloudObject *pc_obj;
PCLPointCloudObject<pcl::PointXYZRGB> *pcl_pc_obj;
Scene scene;

bool primitive_filtering = false;


struct BBoxes : public geom::SceneObject{
      BBoxes(){};
      void update(std::vector<PointCloudSegmentPtr > &clusters);
    };

BBoxes *bboxes;

struct AdaptedSceneMouseHandler : public MouseHandler{
  icl::utils::Mutex mutex;
  MouseHandler *h;

  AdaptedSceneMouseHandler(MouseHandler *h):h(h){
  }

  void process(const MouseEvent &e){
    icl::utils::Mutex::Locker l(mutex);
    h->process(e);
  }

} *mouse = 0;


  
void BBoxes::update(std::vector<PointCloudSegmentPtr > &clusters){
  SceneObject::lock();
  removeAllChildren();
  
  for(size_t i=0;i<clusters.size();++i){
    SceneObject *so = 0, *so2=0;
    const PointCloudSegment::AABB &aabb = clusters[i]->aabb;
    so = addCuboid((aabb.min[0]+aabb.max[0])*0.5,
                  (aabb.min[1]+aabb.max[1])*0.5,
                  (aabb.min[2]+aabb.max[2])*0.5,
                  (aabb.max[0]-aabb.min[0]),
                  (aabb.max[1]-aabb.min[1]),
                  (aabb.max[2]-aabb.min[2]));

    so->setVisible(Primitive::quad,false);
    so->setColor(Primitive::quad,geom_red(30));
    so->setColor(Primitive::line,geom_red());
    
    for(int j=0;j<clusters[i]->getNumSubSegments();++j){
      const PointCloudSegment::AABB &aabb = clusters[i]->getSubSegment(j)->aabb;
      so2 = so->addCuboid((aabb.min[0]+aabb.max[0])*0.5,
                          (aabb.min[1]+aabb.max[1])*0.5,
                          (aabb.min[2]+aabb.max[2])*0.5,
                          (aabb.max[0]-aabb.min[0]),
                          (aabb.max[1]-aabb.min[1]),
                          (aabb.max[2]-aabb.min[2]));

      so2->setVisible(Primitive::quad,true);
      so2->setColor(Primitive::quad,geom_green(30));
      so2->setColor(Primitive::line,geom_green());
    }
  }
  SceneObject::unlock();
}

//ros callback function
void getPrimitivesFromROS(const visualization_msgs::MarkerArray::ConstPtr& markerarray){
	ROS_DEBUG("get primitives from ros");
	std::vector<Primitive3DFilter::Primitive3D> rosPrimitives;
	for(int i = 0; i < markerarray->markers.size(); ++i) {
		visualization_msgs::Marker marker = markerarray->markers[i];
		Primitive3DFilter::PrimitiveType primitiveType = Primitive3DFilter::CUBE;
		switch(markerarray->markers[i].type){
			case visualization_msgs::Marker::CYLINDER :
				primitiveType = Primitive3DFilter::CYLINDER;
				break;
			case visualization_msgs::Marker::SPHERE :
				primitiveType = Primitive3DFilter::SPHERE;
				break;
			case visualization_msgs::Marker::CUBE :
				primitiveType = Primitive3DFilter::CUBE;
				break;
			default:
				//how to handle different types?
				break;
		}
		icl::geom::Vec primitivePosition(markerarray->markers[i].pose.position.x*1000.0, markerarray->markers[i].pose.position.y*1000.0, markerarray->markers[i].pose.position.z*1000.0, 1);
		Primitive3DFilter::Quaternion primitiveOrientation(Vec3(markerarray->markers[i].pose.orientation.x, markerarray->markers[i].pose.orientation.y, markerarray->markers[i].pose.orientation.z), markerarray->markers[i].pose.orientation.w);
		icl::geom::Vec primitiveScale(markerarray->markers[i].scale.x*1000.0, markerarray->markers[i].scale.y*1000.0, markerarray->markers[i].scale.z*1000.0, 1);
		Primitive3DFilter::Primitive3D primitive(primitiveType, primitivePosition, primitiveOrientation, primitiveScale, 0, markerarray->markers[i].text);
		rosPrimitives.push_back(primitive);
	}
	primitivesMutex.lock();
	primitives = rosPrimitives;
	primitivesMutex.unlock();
}

void init(){
  
  // initialize the input streams via a kinect class
  // TODO:Guillaume: read all this from rosparam
  std::string desired_depth_unit = "";
  bool isKinect=true;
  if(pa("-no-kinect")){
    isKinect=false;
  }
  if (pa("-du"))
    desired_depth_unit = *pa("-du",0);
  if (pa("-pc"))
    primitive_filtering = true;
  if(pa("-file")){
    Kinect::SourceSpec spec;
    if(*pa("-file",0) == "sm") spec=Kinect::simFromSharedMem;
    else if(*pa("-file",0) == "raw") spec=Kinect::simFromFileRaw;
    else if(*pa("-file",0) == "depth") spec=Kinect::simFromFileDepth;
    else throw ICLException("expecting source type from [sm|raw|depth]");
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"),
                        desired_depth_unit, spec,
                        *pa("-file",1), *pa("-file",2), "","","","","","","",isKinect);
  }else if (pa("-dir")){
    std::string sDepthFiles=*pa("-dir") + "/depth*";
    std::string sColorFiles=*pa("-dir") + "/color*";
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"),
                        desired_depth_unit,
                        Kinect::simFromFileDepth,
                        sDepthFiles, sColorFiles, "","","","","","","",isKinect);
  }else if(pa("-ue")){
    std::string props;
    if(pa("-dc-camera-properties")){
      props = pa("-dc-camera-properties").as<string>();
    }
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"),
                        desired_depth_unit,
                        Kinect::realButExternalColor,
                        "","", pa("-ue",0), pa("-ue",1), props, "","","","", isKinect);
  } else if (pa("-di") && pa("-ci")) {
    std::string dc_props;
    if(pa("-dc-camera-properties")){
      dc_props = pa("-dc-camera-properties").as<string>();
    }
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"),
                        desired_depth_unit,
                        Kinect::iclSource, "","","","",dc_props,
                        pa("-di",0),pa("-di",1),pa("-ci",0),pa("-ci",1), isKinect);
  }else{
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"), desired_depth_unit, Kinect::realCamera,"","","","","","","","","", isKinect);
  }

  if(pa("-no-kinect")){
    temporalSmoothing = new filter::MotionSensitiveTemporalSmoothing(0, 15);
  }else{
    temporalSmoothing = new filter::MotionSensitiveTemporalSmoothing(2047, 15);
  }
  temporalSmoothing->setUseCL(true);

  // initialize the pointcloud
  pc_obj = new PointCloudObject(kinect->size.width, kinect->size.height,true,false,true); //was  true,true,true at Qiang's code
  pcl_pc_obj = new PCLPointCloudObject<pcl::PointXYZRGB>(kinect->size.width, kinect->size.height);
       
                                       
  if(pa("-no-kinect")){                                     
    pointCloudCreator = new PointCloudCreator(kinect->depthCam, kinect->colorCam,icl::geom::PointCloudCreator::DistanceToCamPlane);                                  
  }else{
    pointCloudCreator = new PointCloudCreator(kinect->depthCam, kinect->colorCam,icl::geom::PointCloudCreator::KinectRAW11Bit);                                  
  }  
  pointCloudCreator->setUseCL(false);//texture data extraction only supported in cpp version

  // initialize the segmenter on CPU, GPU or AUTOMATIC
  // TODO:Guillaume: read all this from rosparam
  if(pa("-fcpu")){
    segmenter = new ConfigurableDepthImageSegmenter(ConfigurableDepthImageSegmenter::CPU, kinect->depthCam, kinect->colorCam);
  }else if(pa("-fgpu")){
    segmenter = new ConfigurableDepthImageSegmenter(ConfigurableDepthImageSegmenter::GPU, kinect->depthCam, kinect->colorCam);
  }else{
    segmenter = new ConfigurableDepthImageSegmenter(ConfigurableDepthImageSegmenter::BEST, kinect->depthCam, kinect->colorCam);
  }
  std::cout<<"segmenter constructed "<<std::endl;
  // load properties from file
  // TODO:Guillaume: read all this from rosparam
  if(pa("-so")){
    segmenter->loadProperties(pa("-so"));
  }
  std::cout<<"segmenter configured"<<std::endl;

  // initialize GUI
  GUI controls = HBox().minSize(12,12);
  controls << ( VBox()
                << Button("reset view").handle("resetView")
                << CheckBox("Show primitives").handle("showPrimitivesHandle") //for primitives filtering
                << Prop("segmentation").minSize(10,8)
                << (VBox().label("temporal smoothing")
                   << CheckBox("use temporal smoothing",true).handle("useTemporalSmoothing")
        				   << Slider(1,15,6).out("smoothingSize").label("smoothing size").handle("smoothingSizeHandle")
        				   << Slider(1,22,10).out("smoothingDiff").label("smoothinh diff").handle("smoothingDiffHandle")
        				   << CheckBox("color pointcloud",false).handle("colorPointcloudHandle")
                   )
                );
  
  
  gui << ( VBox() 
           << Image().handle("hdepth").minSize(8,5).label("depth image")
           << Image().handle("hcolor").minSize(8,5).label("color image")
           << Image().handle("hedgefil").minSize(8,5).label("depth filtered")
           << Image().handle("hedge").minSize(8,5).label("surface edges")
         ) << 
         ( HSplit()
           << Draw3D().handle("draw3D").minSize(40,30)
           << controls
           )<< Show();

  // kinect camera
  scene.addCamera(kinect->depthCam);

  //  view camera
  scene.addCamera(kinect->depthCam);

  scene.setBounds(1000);  // ??
    
  // initialize primitive visualization
  if (primitive_filtering)
  {
    primitive_holder = new SceneObject();
    primitive_holder->setLockingEnabled(true);
    scene.addObject(primitive_holder,true);
  }

  scene.setDrawCoordinateFrameEnabled(true);
  scene.setDrawCamerasEnabled(true);

  scene.addObject(pc_obj);
  scene.setBounds(1000);
  
  // initialize a bounding box handler object
  bboxes = new BBoxes();
  scene.addObject((SceneObject*)bboxes);

  // move out of the camera center 
  //Vec pold = scene.getCamera(0).getPosition();
  //Vec p = Vec(0,pold[1]+10,pold[2]+10);
  //scene.getCamera(1).setPosition(p);

  DrawHandle3D draw3D = gui["draw3D"];

  // initialize a mouse handler
  mouse = new AdaptedSceneMouseHandler(scene.getMouseHandler(VIEW_CAM));
  draw3D->install(mouse);

  scene.setLightingEnabled(false);
  pc_obj->setPointSize(3);

  if (primitive_filtering)
  {

    // initialize primitive filter and read config from file
    // TODO:Guillaume: read config from rosparam
    primitiveFilter = new Primitive3DFilter(Primitive3DFilter::FilterConfig(*pa("-pc")));
    std::cout<<"primitive filter ready "<<std::endl;
  }
  ros::spinOnce();

}

void run(){


  static ButtonHandle resetView = gui["resetView"];
  if(resetView.wasTriggered()){
    scene.lock();
    scene.getCamera(1) = scene.getCamera(0);
    //Vec pold = scene.getCamera(0).getPosition();
    //Vec p = Vec(0,pold[1]+10,pold[2]+10);
    //scene.getCamera(1).setPosition(p);
    scene.unlock();
  }
  // grab images
  Kinect::Frame f = kinect->grab();
  if (f.isValid())
  {
    
    gui["hdepth"] = f.d();
    gui["hcolor"] = f.c();
      //invalid_frame = false;

    // compute the pointcloud
    ROS_DEBUG_STREAM("captured an image ");
    pc_obj->lock();
//    segmenter->computePointCloudFirst(f.d(), *pc_obj);

    int smoothingSize = gui["smoothingSize"];
    int smoothingDiff = gui["smoothingDiff"];
    temporalSmoothing->setFilterSize(smoothingSize);
    temporalSmoothing->setDifference(smoothingDiff);
    static core::ImgBase *filteredImage = 0;
    bool useTempSmoothing = gui["useTemporalSmoothing"];
    if(useTempSmoothing==true){//temporal smoothing
      temporalSmoothing->apply(&f.d(),&filteredImage);    
    }
    
    if(useTempSmoothing){
    	pointCloudCreator->create(*filteredImage->as32f(), *pc_obj,&f.c(), segmenter->getPropertyValue("general.depth scaling"));//1.0);//, 1.0);//, depthScaling);
    }else{
      pointCloudCreator->create(f.d(), *pc_obj,&f.c(), segmenter->getPropertyValue("general.depth scaling"));//1.0);//, 1.0);//, depthScaling);
    }



    ROS_DEBUG_STREAM("pointcloud computed ");
    // create a copy of the depthImage
    Img32f depthImageFiltered = f.d();

    // cleanup previouly displayed primitives
    if (primitive_filtering)
    {
      primitive_holder->lock();
      primitive_holder->removeAllChildren();
      // primitive filter
      primitivesMutex.lock();
      primitiveFilter->apply(primitives, *pc_obj, &depthImageFiltered);
      // display primitives
      static CheckBoxHandle showPrimitives = gui["showPrimitivesHandle"];
      if(showPrimitives.isChecked()) {
        for(uint i = 0; i < primitives.size(); ++i) {
          primitives[i].toSceneObject(primitive_holder);
        }
      }
      primitivesMutex.unlock();
      primitive_holder->unlock();
      ROS_DEBUG_STREAM("depth image filtered");
    }
    ROS_DEBUG_STREAM("starting segmentation");
    // segment with depthImage cleared from primitives
    if (primitive_filtering)
    {
      segmenter->apply(depthImageFiltered, *pc_obj);
      ROS_DEBUG_STREAM("segmentation after filter done");
    }
    else
    {
      //segmenter->applySecond(f.d(), *pc_obj);
      if(useTempSmoothing==true){
      	segmenter->apply(*filteredImage->as32f(),*pc_obj);
			}else{
      	segmenter->apply(f.d(),*pc_obj);			
			}

      ROS_DEBUG_STREAM("segmentation done");
    }
    
    
    bool colorPointcloud = gui["colorPointcloudHandle"];
    if(colorPointcloud){
      pc_obj->setColorsFromImage(segmenter->getColoredLabelImage());
    }
    
    
    pc_obj->unlock();
    // extract data
    std::vector<std::vector<int> > seg;
    std::vector<std::vector<int> > surf;
    std::vector<geom::SurfaceFeatureExtractor::SurfaceFeature>  vec_sf;
    ROS_DEBUG_STREAM("extracting segments ");
    seg = segmenter->getSegments();
    ROS_DEBUG_STREAM("extracting surfaces ");
    surf = segmenter->getSurfaces();
    ROS_DEBUG_STREAM("extracting surface features ");
    vec_sf = segmenter->getSurfaceFeatures();
    ROS_DEBUG_STREAM("extracting clusters ");
//    std::vector<PointCloudSegmentPtr> clusters  = segmenter->getClusters(depthImageFiltered, *pc_obj);
      std::vector<PointCloudSegmentPtr> clusters = segmenter->getClusters(*pc_obj);
    ROS_DEBUG_STREAM("data extracted");
    
    // update bounding box views (compute and add objects to scene)
    scene.lock();
    bboxes->update(clusters);
    scene.unlock();
   
    std::vector<std::pair<icl::utils::Point,icl::utils::Point> > bboxes2DColor(clusters.size());
    for(int i=0; i<clusters.size(); i++){
      PointCloudSegment::AABB &aabb = clusters[i]->aabb;         
      std::vector<icl::geom::Vec> wp(8);
      wp[0] = icl::geom::Vec(aabb.min[0],aabb.min[1],aabb.min[2],1);
      wp[1] = icl::geom::Vec(aabb.min[0],aabb.min[1],aabb.max[2],1);
      wp[2] = icl::geom::Vec(aabb.min[0],aabb.max[1],aabb.min[2],1);
      wp[3] = icl::geom::Vec(aabb.min[0],aabb.max[1],aabb.max[2],1);
      wp[4] = icl::geom::Vec(aabb.max[0],aabb.min[1],aabb.min[2],1);
      wp[5] = icl::geom::Vec(aabb.max[0],aabb.min[1],aabb.max[2],1);
      wp[6] = icl::geom::Vec(aabb.max[0],aabb.max[1],aabb.min[2],1);
      wp[7] = icl::geom::Vec(aabb.max[0],aabb.max[1],aabb.max[2],1);
      std::vector<icl::utils::Point32f> cp = kinect->colorCam.project(wp);
      
      std::pair<icl::utils::Point,icl::utils::Point> colorBBox;
      colorBBox.first=icl::utils::Point(1000000,1000000);
      colorBBox.second=icl::utils::Point(-1000000,-1000000);
      for(unsigned int j=0; j<cp.size(); j++){
        if(cp[j].x>=0 && cp[j].y>=0){//point is in visible space
          if(cp[j].x<colorBBox.first.x) colorBBox.first.x=cp[j].x;
          if(cp[j].y<colorBBox.first.y) colorBBox.first.y=cp[j].y;
          if(cp[j].x>colorBBox.second.x) colorBBox.second.x=cp[j].x;
          if(cp[j].y>colorBBox.second.y) colorBBox.second.y=cp[j].y;
        }          
      }
      bboxes2DColor[i]=colorBBox;
    }

	  /*core::DataSegment<float,4> xyz = pc_obj->selectXYZH(); 
    int w = depthImageFiltered.getSize().width;
    //get 2D(image space) bounding boxes of segments; depthImage/pointcloud coordinates
               
    std::vector<std::pair<utils::Point,utils::Point> > bboxes2D = segmenter->getBoundingBoxes2D();
    
    //for all depthImage/PointCloud coordinates the corresponding colorImage coordinates
    core::DataSegment<float,2> colorTex = pointCloudCreator->getColorTexturePoints();
    
    std::vector<std::pair<utils::Point,utils::Point> > bboxes2DColor(bboxes2D.size());
    
    //direct mapping from depth bbox to color bbox would destroy the axis alignment
    for(unsigned int i=0; i<bboxes2D.size(); i++){
    
      //the 4 edges of the bbox for depth/pointcloud
      std::vector<utils::Point> p(4);
      p[0] = utils::Point(bboxes2D[i].first.x,bboxes2D[i].first.y);
      p[1] = utils::Point(bboxes2D[i].second.x,bboxes2D[i].first.y);
      p[2] = utils::Point(bboxes2D[i].second.x,bboxes2D[i].second.y);
      p[3] = utils::Point(bboxes2D[i].first.x,bboxes2D[i].second.y);
      
      //the point ids
      std::vector<int> id(p.size());
      for(unsigned int j=0; j<p.size(); j++){
        id[j] = p[j].x+p[j].y*w;
      }
      
      //the corresponding colorImage points
      std::vector<utils::Point> cp(p.size());
      for(unsigned int j=0; j<p.size(); j++){
        cp[j] = utils::Point(colorTex[id[j]][0],colorTex[id[j]][1]);
      }
      
      //recalculate the axis aligned bbox for color
      std::pair<utils::Point,utils::Point> colorBBox;
      colorBBox.first=utils::Point(1000000,1000000);
      colorBBox.second=utils::Point(-1000000,-1000000);
      for(unsigned int j=0; j<p.size(); j++){
        if(cp[j].x>=0 && cp[j].y>=0){//point is in visible space
          if(cp[j].x<colorBBox.first.x) colorBBox.first.x=cp[j].x;
          if(cp[j].y<colorBBox.first.y) colorBBox.first.y=cp[j].y;
          if(cp[j].x>colorBBox.second.x) colorBBox.second.x=cp[j].x;
          if(cp[j].y>colorBBox.second.y) colorBBox.second.y=cp[j].y;
        }          
      }
      bboxes2DColor[i]=colorBBox;
    }
    */
      

    // render images
    gui["hedge"] = segmenter->getEdgeImage();
    gui["hedgefil"] = &depthImageFiltered;

//PointCloud with Transform to PointCloud2Msg    
/*    pc_obj->setColorsFromImage(segmenter->getColoredLabelImage());
    pcl_pc_obj->selectXYZH() = pc_obj->selectXYZH();    
    pcl_pc_obj->selectRGBA32f() = pc_obj->selectRGBA32f();
    sensor_msgs::PointCloud2::Ptr pc_msg (new sensor_msgs::PointCloud2);
    pcl::PCLPointCloud2 pcl_pc_obj2;
    pcl::toPCLPointCloud2(pcl_pc_obj->pcl(),pcl_pc_obj2);
    pcl_conversions::fromPCL (pcl_pc_obj2, *pc_msg);
    pub_pc.publish(pc_msg);    
*/

//PointCloud directly
    pcl_pc_obj->pcl().header.frame_id = "camera_calibration_frame";
    //Choose RGB vs ColoredLabel from App
    //pcl_pc_obj->setColorsFromImage(segmenter->getColoredLabelImage());
    DataSegment<float,4> xyzh = pc_obj->selectXYZH();
    DataSegment<float,4> rgba = pc_obj->selectRGBA32f();    
    DataSegment<float,3> pcl_xyz = pcl_pc_obj->selectXYZ();
    DataSegment<icl8u,3> pcl_bgr = pcl_pc_obj->selectBGR();
    for(int i=0; i<pc_obj->getDim(); i++){
      pcl_xyz[i][0]=xyzh[i][0]*0.001;
      pcl_xyz[i][1]=xyzh[i][1]*0.001;
      pcl_xyz[i][2]=xyzh[i][2]*0.001;
      pcl_bgr[i][0]=(icl8u)(rgba[i][2]*255.);//
      pcl_bgr[i][1]=(icl8u)(rgba[i][1]*255.);//
      pcl_bgr[i][2]=(icl8u)(rgba[i][0]*255.);//
    }
    pub_pc2.publish(pcl_pc_obj->pcl());



    //world belief msg
    segmentation_msgs::WorldBelief wb;
    if(clusters.size()>0){
      wb.object_beliefs.resize(clusters.size());
      for(int i = 0; i < clusters.size(); i++){
        PointCloudSegmentPtr cloud = clusters[i]->flatten();
        //PCLPointCloudObject<pcl::PointXYZRGB> *pcl_pc = new PCLPointCloudObject<pcl::PointXYZRGB>(1,cloud->getDim());
        PCLPointCloudObject<pcl::PointXYZRGB> pcl_pc(cloud->getDim(),1);
        pcl_pc.pcl().header.frame_id = "camera_calibration_frame";
        DataSegment<float,4> xyzh = cloud->selectXYZH();
        DataSegment<float,4> rgba = cloud->selectRGBA32f();    
        DataSegment<float,3> pcl_xyz = pcl_pc.selectXYZ();
        DataSegment<icl8u,3> pcl_bgr = pcl_pc.selectBGR();
        for(int i=0; i<cloud->getDim(); i++){
          pcl_xyz[i][0]=xyzh[i][0]*0.001;
          pcl_xyz[i][1]=xyzh[i][1]*0.001;
          pcl_xyz[i][2]=xyzh[i][2]*0.001;
          pcl_bgr[i][0]=(icl8u)(rgba[i][2]*255.);//
          pcl_bgr[i][1]=(icl8u)(rgba[i][1]*255.);//
          pcl_bgr[i][2]=(icl8u)(rgba[i][0]*255.);//
        }
        sensor_msgs::PointCloud2 pc_msg;//::Ptr pc_msg (new sensor_msgs::PointCloud2);
        pcl::PCLPointCloud2 pcl_pc2;
        pcl::toPCLPointCloud2(pcl_pc.pcl(),pcl_pc2);
        pcl_conversions::fromPCL (pcl_pc2, pc_msg);
        wb.object_beliefs[i].pointcloud = pc_msg;

        PointCloudSegment::AABB aabb = clusters[i]->aabb;
        wb.object_beliefs[i].axis_aligned_box.pose.position.x=aabb.min.x+(aabb.max.x-aabb.min.x)/2.;
        wb.object_beliefs[i].axis_aligned_box.pose.position.y=aabb.min.y+(aabb.max.y-aabb.min.y)/2.;
        wb.object_beliefs[i].axis_aligned_box.pose.position.z=aabb.min.z+(aabb.max.z-aabb.min.z)/2.;
        wb.object_beliefs[i].axis_aligned_box.pose.orientation.w=1.0;
        wb.object_beliefs[i].axis_aligned_box.dimensions.x=aabb.max.x-aabb.min.x;
        wb.object_beliefs[i].axis_aligned_box.dimensions.y=aabb.max.y-aabb.min.y;
        wb.object_beliefs[i].axis_aligned_box.dimensions.z=aabb.max.z-aabb.min.z;
        
        segmentation_msgs::Hypothesis idHypo;
        idHypo.type="id";
        idHypo.values.push_back(str(i));
        idHypo.reliabilities.push_back(1);
        wb.object_beliefs[i].hypotheses.push_back(idHypo);    
        
      }
      
      cv::Mat *ros_img = icl::core::img_to_mat(&f.c());
      sensor_msgs::ImagePtr imgMsg = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::RGB8, *ros_img).toImageMsg();
      delete ros_img;
      wb.segmented_image.image=*imgMsg;
      
      wb.segmented_image.regions.resize(bboxes2DColor.size());
      for(int i=0; i<bboxes2DColor.size(); i++){
        if((bboxes2DColor[i].second.x-bboxes2DColor[i].first.x)>0){
          wb.segmented_image.regions[i].x_offset=bboxes2DColor[i].first.x;
          wb.segmented_image.regions[i].y_offset=bboxes2DColor[i].first.y;
          wb.segmented_image.regions[i].width=bboxes2DColor[i].second.x-bboxes2DColor[i].first.x;
          wb.segmented_image.regions[i].height=bboxes2DColor[i].second.y-bboxes2DColor[i].first.y;
        }else{
          wb.segmented_image.regions[i].x_offset=0;
          wb.segmented_image.regions[i].y_offset=0;
          wb.segmented_image.regions[i].width=0;
          wb.segmented_image.regions[i].height=0;
        }
      }
      
      pub_seg.publish(wb);
    }
    


//No RGB mapping for both color channels in rviz (only rainbow (HLS?))
//PointCloudMsg
/*
    sensor_msgs::PointCloud::Ptr pc_msg (new sensor_msgs::PointCloud);
    core::Img8u labelImg = segmenter->getColoredLabelImage();
    //DataSegment<float,4> xyz = pc_obj->selectXYZH();
    DataSegment<float,4> rgba = pc_obj->selectRGBA32f();
    pc_msg->points.clear();
    pc_msg->channels.clear();
    pc_msg->header.frame_id = "camera_link";//"frame_seg";
    //pcl_conversions::toPCL(ros::Time::now(), pc_msg->header.stamp);
    sensor_msgs::ChannelFloat32 colorChannel;
    colorChannel.name="rgb-label";
    sensor_msgs::ChannelFloat32 colorChannel2;
    colorChannel2.name="rgb-color";
    for(int y=0; y<labelImg.getSize().height; y++){
      for(int x=0; x<labelImg.getSize().width; x++){
        int i = x+y*labelImg.getSize().width;
        //pcl::PointXYZRGB p;
        geometry_msgs::Point32 p;
        p.x = xyz[i][0]*0.001;
        p.y = xyz[i][1]*0.001;
        p.z = xyz[i][2]*0.001;
        uint32_t rgb = ((uint32_t)labelImg(x,y,0) << 16 | (uint32_t)labelImg(x,y,1) << 8 | (uint32_t)labelImg(x,y,2));
        float frgb = (float)rgb;//*reinterpret_cast<float*>(&rgb);
        
        uint32_t rgb2 = ((uint32_t)(rgba[i][0]*255) << 16 | (uint32_t)(rgba[i][1]*255) << 8 | (uint32_t)(rgba[i][2]*255));
        float frgb2 = *reinterpret_cast<float*>(&rgb2);
        //pc_msg->points.push_back(pcl::PointXYZRGB(p));
        pc_msg->points.push_back(geometry_msgs::Point32(p));
        colorChannel.values.push_back(frgb);
        colorChannel2.values.push_back(frgb2);
      }
    }
    pc_msg->channels.push_back(colorChannel);
    pc_msg->channels.push_back(colorChannel2);
    pub_pc.publish(pc_msg);
*/


    /* not verified // example code to access data
    DataSegment<float,4> pcs = obj->selectXYZH();
    if((seg.size()<25)&&(seg.size()>0)){

      //publish all point clouds
      
      if(vec_sf.size() != 0){
        ColorPointCloud::Ptr msg (new ColorPointCloud);
        msg->points.clear();
        msg->header.frame_id = "frame";
        pcl_conversions::toPCL(ros::Time::now(), msg->header.stamp);

        uint8_t r,g,b;
        for(unsigned int i = 0; i < seg.size(); i++){
          std::cout<<i<<" segment has "<<seg.at(i).size()<<" surfaces"<<std::endl;
          for (unsigned int j = 0; j < seg.at(i).size();j++){
            if(j+i*seg.size() == 0){
              r = 255, g = 0, b = 0;
            }
            if(j+i*seg.size() == 1){
              r = 0, g = 255, b = 0;
            }
            if(j+i*seg.size() == 2){
              r = 0, g = 0, b = 255;
            }
            //for point clouds
            for(int k = 0; k<surf[seg[i][j]].size(); k++){
              pcl::PointXYZRGB p;
              p.x = pcs[surf[seg[i][j]][k]][0]*0.001;
              p.y = pcs[surf[seg[i][j]][k]][1]*0.001;
              p.z = pcs[surf[seg[i][j]][k]][2]*0.001;
              // pack r/g/b into rgb
              uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
              p.rgb = *reinterpret_cast<float*>(&rgb);

              msg->points.push_back (pcl::PointXYZRGB(p));
            }
          }
          pub_pc.publish (msg);
        }
        
        //publish the surface center feature
        std::cout<<" there are "<<seg.size()<<"segments and "<<vec_sf.size()<<"surface features"<<std::endl;
        if ((marker_surf_array_pub.getNumSubscribers() >= 1)){
          visualization_msgs::MarkerArray act_marker;
          act_marker.markers.resize(vec_sf.size());
          for(int i = 0; i < vec_sf.size(); i++){

            act_marker.markers[i].header.frame_id = "frame";
            act_marker.markers[i].header.stamp = ros::Time::now();
            // Set the namespace and id for this marker.  This serves to create a unique ID
            // Any marker sent with the same namespace and id will overwrite the old one
            act_marker.markers[i].ns = "iclgeopub";
            act_marker.markers[i].id = i;
            // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
            act_marker.markers[i].type = visualization_msgs::Marker::CUBE;
            // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
            act_marker.markers[i].action = visualization_msgs::Marker::ADD;
            act_marker.markers[i].pose.position.x = vec_sf.at(i).meanPosition[0]*0.001;
            act_marker.markers[i].pose.position.y = vec_sf.at(i).meanPosition[1]*0.001;
            act_marker.markers[i].pose.position.z = vec_sf.at(i).meanPosition[2]*0.001;
            act_marker.markers[i].pose.orientation.x = 0.0;
            act_marker.markers[i].pose.orientation.y = 0.0;
            act_marker.markers[i].pose.orientation.z = 0.0;
            act_marker.markers[i].pose.orientation.w = 1.0;

            // Set the scale of the marker -- 1x1x1 here means 1m on a side
            act_marker.markers[i].scale.x = .01;
            act_marker.markers[i].scale.y = .01;
            act_marker.markers[i].scale.z = .01;

            // Set the color -- be sure to set alpha to something non-zero!
            act_marker.markers[i].color.r = 0.0f;
            act_marker.markers[i].color.g = 0.0f;
            act_marker.markers[i].color.b = 0.1f;
            act_marker.markers[i].color.a = 1.0;
            act_marker.markers[i].lifetime = ros::Duration();
          }
          marker_surf_array_pub.publish(act_marker);
        }
      } 
    }
    */

    gui["draw3D"].link(scene.getGLCallback(VIEW_CAM));
    gui["draw3D"].render();
  }
  ros::spinOnce();
}

int main(int argc, char* argv[]){
  ros::init(argc, argv, "icl_ros_segmentation");
  nh = new ros::NodeHandle();
  
  //subscribe to ros topic
  ros::Subscriber sub = nh->subscribe<visualization_msgs::MarkerArray>("robot_collision_shape",1,getPrimitivesFromROS);
  pub_pc = nh->advertise<sensor_msgs::PointCloud> ("segmentation/segmented_tool_pc", 1);
  pub_pc2 = nh->advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("segmentation/segmented_tool_pc2", 1);
  
  pub_seg = nh->advertise<segmentation_msgs::WorldBelief> ("segmentation/world_belief", 1);

  return ICLApp(argc,argv,"-size|-s(Size=VGA) -fcpu|force-cpu "
                          "-fgpu|force-gpu "
                          "-primitive-config|-pc(filename) "
                          "-depth-cam|-d(file) -color-cam|-c(file) "
                          "-file(type=raw|depth|sm,depth-source,color-source) -dir(directory) "
                          "-dc-camera-properties|-dcp(filename) -use-external-color-camera|-ue(2) "
                          "-initial-segmenter-options|-so(xml-file-name) "
                          "-depth-in|-di(type=kinectd,source=0) "
                          "-color-in|-ci(type=kinectc,source=0) "
                          "-depth-unit|-du(unit=raw|mm) "
                          "-no-kinect"
                          ,init,run).exec();
  
  
  ros::shutdown();
  delete nh;

}
