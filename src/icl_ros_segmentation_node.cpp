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
#include <ICLQt/Application.h>
#include <ICLGeom/Scene.h>
#include <ICLQt/Common.h>
#include <Kinect.h>
#include <ICLGeom/SurfaceFeatureExtractor.h>
#include <ICLGeom/Primitive3DFilter.h>


// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

// common
#include <iostream>
#include <map>
#include <string>

using icl::utils::pa;
using icl::utils::pa_init;

HSplit gui;
int KINECT_CAM=0,VIEW_CAM=1;

// pointers to individual modules 
SmartPtr<Kinect> kinect;
SmartPtr<ConfigurableDepthImageSegmenter> segmenter;

// ros
ros::NodeHandle *nh;

// for primitive filter
SceneObject *primitive_holder;
std::vector<Primitive3DFilter::Primitive3D> primitives;
SmartPtr<Primitive3DFilter> primitiveFilter;
//ListenerPtr primitivesetListener;
icl::utils::Mutex primitivesMutex;

PointCloudObject *pc_obj;
Scene scene;

bool primitive_filtering = false;


struct BBoxes : public geom::SceneObject{
      BBoxes(){};
      void update(std::vector<PointCloudSegmentPtr > &clusters);
    };

BBoxes *bboxes;

struct AdaptedSceneMouseHandler : public MouseHandler{
  Mutex mutex;
  MouseHandler *h;

  AdaptedSceneMouseHandler(MouseHandler *h):h(h){
  }

  void process(const MouseEvent &e){
    Mutex::Locker l(mutex);
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
				//TODO:sarah: how to handle different types?
				break;
		}
		Vec primitivePosition(markerarray->markers[i].pose.position.x*1000.0, markerarray->markers[i].pose.position.y*1000.0, markerarray->markers[i].pose.position.z*1000.0, 1);
		Primitive3DFilter::Quaternion primitiveOrientation(Vec3(markerarray->markers[i].pose.orientation.x, markerarray->markers[i].pose.orientation.y, markerarray->markers[i].pose.orientation.z), markerarray->markers[i].pose.orientation.w);
		Vec primitiveScale(markerarray->markers[i].scale.x*1000.0, markerarray->markers[i].scale.y*1000.0, markerarray->markers[i].scale.z*1000.0, 1);
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
                        *pa("-file",1), *pa("-file",2));
  }else if (pa("-dir")){
    std::string sDepthFiles=*pa("-dir") + "/depth*";
    std::string sColorFiles=*pa("-dir") + "/color*";
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"),
                        desired_depth_unit,
                        Kinect::simFromFileDepth,
                        sDepthFiles, sColorFiles);
  }else if(pa("-ue")){
    std::string props;
    if(pa("-dc-camera-properties")){
      props = pa("-dc-camera-properties").as<string>();
    }
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"),
                        desired_depth_unit,
                        Kinect::realButExternalColor,
                        "","", pa("-ue",0), pa("-ue",1), props);
  } else if (pa("-di") && pa("-ci")) {
    std::string dc_props;
    if(pa("-dc-camera-properties")){
      dc_props = pa("-dc-camera-properties").as<string>();
    }
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"),
                        desired_depth_unit,
                        Kinect::iclSource, "","","","",dc_props,
                        pa("-di",0),pa("-di",1),pa("-ci",0),pa("-ci",1));
  }else{
    kinect = new Kinect(pa("-size"), pa("-d"), pa("-c"), desired_depth_unit);
  }

  // initialize the pointcloud
  pc_obj = new PointCloudObject(kinect->size.width, kinect->size.height,true,false,true); //was  true,true,true at Qiang's code

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
  Vec pold = scene.getCamera(0).getPosition();
  Vec p = Vec(0,pold[1]+10,pold[2]+10);
  scene.getCamera(1).setPosition(p);

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
    Vec pold = scene.getCamera(0).getPosition();
    Vec p = Vec(0,pold[1]+10,pold[2]+10);
    scene.getCamera(1).setPosition(p);
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
    segmenter->computePointCloudFirst(f.d(), *pc_obj);
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
      segmenter->applySecond(depthImageFiltered, *pc_obj);
      ROS_DEBUG_STREAM("segmentation after filter done");
    }
    else
    {
      segmenter->applySecond(f.d(), *pc_obj);
      ROS_DEBUG_STREAM("segmentation done");
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
    std::vector<PointCloudSegmentPtr> clusters  = segmenter->getClusters(depthImageFiltered, *pc_obj);
    ROS_DEBUG_STREAM("data extracted");
    
    // update bounding box views (compute and add objects to scene)
    scene.lock();
    bboxes->update(clusters);
    scene.unlock();
   

    // render images
    gui["hedge"] = segmenter->getEdgeImage();
    gui["hedgefil"] = &depthImageFiltered;

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
    ros::spinOnce();
  }
}

int main(int argc, char* argv[]){
  ros::init(argc, argv, "icl_ros_segmentation");
  nh = new ros::NodeHandle();
  
  //subscribe to ros topic
  ros::Subscriber sub = nh->subscribe<visualization_msgs::MarkerArray>("/robot_collision_shape",1,getPrimitivesFromROS);
  //pub_pc = nh->advertise<sensor_msgs::PointCloud2> ("segmented_tool_pc", 1);

  return ICLApp(argc,argv,"-size|-s(Size=VGA) -fcpu|force-cpu "
                          "-fgpu|force-gpu "
                          "-primitive-config|-pc(filename) "
                          "-depth-cam|-d(file) -color-cam|-c(file) "
                          "-file(type=raw|depth|sm,depth-source,color-source) -dir(directory) "
                          "-dc-camera-properties|-dcp(filename) -use-external-color-camera|-ue(2) "
                          "-initial-segmenter-options|-so(xml-file-name) "
                          "-depth-in|-di(type=kinectd,source=0) "
                          "-color-in|-ci(type=kinectc,source=0) "
                          "-depth-unit|-du(unit=raw|mm)"
                          ,init,run).exec();
  
  
  ros::shutdown();
  delete nh;

}
