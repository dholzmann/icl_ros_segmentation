#include <Kinect.h>
#include <ICLUtils/Exception.h>


namespace icl{

  using namespace utils;
  using namespace core;
  using namespace geom;

  Kinect::Kinect(const Size &size,
                 const std::string &cfgDepth,
                 const std::string &cfgColor,
                 const std::string &desired_depth_unit,
                 Kinect::SourceSpec spec,
                 const std::string &simDepthSource,
                 const std::string &simColorSource,
                 const std::string &externalColorDevice,
                 const std::string &externalColorID,
                 const std::string &propertyFilename,
                 const std::string &icl_device_d,
                 const std::string &icl_device_args_d,
                 const std::string &icl_device_c,
                 const std::string &icl_device_args_c):
    srcSpec(spec),size(size),desiredDepthUnit(desired_depth_unit){

    switch(spec){
    case iclSource:
      depthGrabber.init(icl_device_d,icl_device_d+"="+icl_device_args_d);
      setGrabberDepthUnit(icl_device_d, icl_device_args_d);
      colorGrabber.init(icl_device_c,icl_device_c+"="+icl_device_args_c);
      if(propertyFilename.length()){
        colorGrabber.loadProperties(propertyFilename);
      }
      break;
      case realCamera:
        depthGrabber.init("kinectd","kinectd=0");
        setGrabberDepthUnit("kinectd", icl_device_args_d);
        colorGrabber.init("kinectc","kinectc=0");
        break;
      case realButExternalColor:
        depthGrabber.init("kinectd","kinectd=0");
        setGrabberDepthUnit("kinectd", icl_device_args_d);
        colorGrabber.init(externalColorDevice,externalColorDevice+"="+externalColorID);
        if(propertyFilename.length()){
          colorGrabber.loadProperties(propertyFilename);
        }
        break;
      case simFromSharedMem:
        depthGrabber.init("sm",str("sm=")+simDepthSource);
        setGrabberDepthUnit(icl_device_d, icl_device_args_d);
        colorGrabber.init("sm",str("sm=")+simColorSource);
        break;
      case simFromFileRaw:
      case simFromFileDepth:
        // for backward compatibility, empty option for file input is mm
        if (desiredDepthUnit.empty())
          desiredDepthUnit = "mm";
        depthGrabber.init("file",str("file=")+simDepthSource);
        setGrabberDepthUnit(icl_device_d, icl_device_args_d);
        colorGrabber.init("file",str("file=")+simColorSource);
        break;
      default:
        throw ICLException(str(__FUNCTION__) + ": invalid image source spec.");
        break;
    }
    
    depthGrabber.useDesired(depth32f, size, formatMatrix);
    colorGrabber.useDesired(depth8u, size,formatRGB);
  
    Camera *cs[] = {&depthCam, &colorCam};
    const std::string *files[] = { &cfgDepth, &cfgColor };
    for(int i=0;i<2;++i){
      *cs[i] = Camera(*files[i]);
      cs[i]->setName(str("colordepth").substr(i*5,5) + " camera");
      if(cs[i]->getRenderParams().chipSize != size){
        throw ICLException(cs[i]->getName() +" file resolution differs from video resolution");
      }
    }
  }
  
  void Kinect::setGrabberDepthUnit(const std::string &device_d, const std::string &device_args){

    // find if camera parameters are set from camera args
    bool camera_depth_unit_set = false;
    if (device_args.find("depth-image-unit") != std::string::npos)
      camera_depth_unit_set = true;

    // camera parameters are set from camera args, check if it conflicts with non-empty depth_is_mm option
    if (camera_depth_unit_set && !desiredDepthUnit.empty())
    {
      std::string curDepthUnit = depthGrabber.getPropertyValue("depth-image-unit");
      // verify both options are not conflicting
      if (desiredDepthUnit!=curDepthUnit)
        throw ICLException(str(__FUNCTION__) + ": conflicting depth-image-unit and desired_depth_unit option.");
    }
    else
    {
      // no camera param set, use option if set or default to raw (backward compatibility)
      if (desiredDepthUnit.empty() || desiredDepthUnit=="raw")
      {
        if (device_d == "kinectd")
          depthGrabber.setPropertyValue("depth-image-unit","raw");
        desiredDepthUnit = "raw"; // store this information if was empty
      }
      else
      {
        if (desiredDepthUnit=="mm")
        {
          if (device_d == "kinectd")
          {
            depthGrabber.setPropertyValue("depth-image-unit","mm");
          }
        }
        else
          throw ICLException(str(__FUNCTION__) + ": unknown desired depth unit :" + desiredDepthUnit);
      }
    }
  }

  Kinect::Frame Kinect::grab(){
    try{
      const core::ImgBase *imgDepth = depthGrabber.grab();
      const core::ImgBase *imgColor = colorGrabber.grab();

      if (!imgDepth || !imgColor)
      {
        return Frame (NULL, NULL);
      }

      // convert to raw if mm provided
        if (ICL_UNLIKELY(desiredDepthUnit == "mm")) {
          Img32f *d = const_cast<Img32f*>(imgDepth->as32f());
          for(int x=size.width-1; x >= 0; --x){
            for(int y=size.height-1; y >= 0; --y){
              
              float val=d->operator()(x,y,0);
              if(val<1){
                d->operator()(x,y,0)=2047.;
              }else{
                d->operator()(x,y,0)=((1000./(val/1.046))-3.3309495161)/(-0.0030711016);  
              }
            }
          }
        }
      
      return Frame (imgDepth, imgColor);
    }
    catch (icl::utils::ICLException &e) {
      ERROR_LOG("failed to grab: " << e.what());
      return Frame (NULL, NULL);
    }
    catch ( const std::exception& e )
    {
        ERROR_LOG("failed to grab other errors: " << e.what());
        return Frame (NULL, NULL);
    }
    catch ( ... )
    {
        ERROR_LOG("failed to grab the rest of the other errors: ");
        return Frame (NULL, NULL);
    }
  }

  Kinect::Frame::Frame(const core::ImgBase *d, const core::ImgBase *c){
    this->depthImage = d;
    this->colorImage = c;
  }


}
