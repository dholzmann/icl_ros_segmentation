#pragma once

#include <ICLIO/GenericGrabber.h>
#include <ICLGeom/Camera.h>

namespace icl{
  struct Kinect{
    enum SourceSpec{
      realCamera,
      simFromSharedMem,
      simFromFileRaw,
      simFromFileDepth,
      realButExternalColor,
      iclSource
    } srcSpec;
    
    io::GenericGrabber depthGrabber, colorGrabber;
    geom::Camera depthCam, colorCam;
    utils::Size size;
    std::string desiredDepthUnit;
    bool isKinect;
    
    Kinect(const utils::Size &size,
           const std::string &cfgCamDepth,  const std::string &cfgCamColor,
           const std::string &desired_depth_unit,
           SourceSpec spec = realCamera,
           const std::string &simDepthSource="",
           const std::string &simColorSource="",
           const std::string &externalColorDevice="",
           const std::string &externalColorID="",
           const std::string &propertyFileName="",
           const std::string &icl_device_d = "",
           const std::string &icl_device_args_d = "",
           const std::string &icl_device_c = "",
           const std::string &icl_device_args_c = "",
           bool isKinect=true);

    class Frame{
      const core::ImgBase *depthImage;
      const core::ImgBase *colorImage;
      public:
      Frame(const core::ImgBase *d, const core::ImgBase *c);
      inline const bool isValid() const {return (depthImage != NULL && colorImage != NULL); }
      
      inline const core::Img32f &d() const { return *depthImage->as32f(); }
      inline const core::Img8u &c() const { return *colorImage->as8u(); }
    };
    
    void setGrabberDepthUnit(const std::string &device_d, const std::string &device_args);
    Frame grab();
  };

}
