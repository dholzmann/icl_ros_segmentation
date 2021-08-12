#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "SurfaceFeature.h"
#include "SegmenterHelper.h"
/// class for coplanarity feature.
/** This class implements the coplanarity feature for feature-graph based segmentation.*/
using namespace cv;

class CoPlanarityFeature{

  public:

    /// Calculates the coplanarity feature matrix.
  /** @param initialMatrix the initial boolean test matrix (0 test if both planar, 1 dont test, preferably an adjacency matrix)
      @param features the surface feature for the surfaces
      @param depthImage the input depthImage
      @param surfaces the vector of surface id vectors
      @param maxAngle the maximum angle for surface orientation similarity test
      @param distanceTolerance the maximum distance in mm for occlusion check
      @param outlierTolerance the maximum number of outlier points in percent for occlusion check
      @param triangles number of combined surface tests
      @param scanlines number of occlusion tests
      @return the boolean coplanarity matrix */
    static Mat apply(Mat &initialMatrix, std::vector<SurfaceFeatureExtract::SurfaceRegionFeature> features,
                      const Mat &depthImage, std::vector<std::vector<int> > &surfaces, float maxAngle=30,
                      float distanceTolerance=3, float outlierTolerance=5, int triangles=50, int scanlines=9);

  private:

    static float getAngle(Vec4f n1, Vec4f n2);
    static Point getRandomPoint(std::vector<int> surface, int imgWidth);
    static Vec4f getNormal(Vec4f p1, Vec4f p2, Vec4f p3);
    static bool criterion1(Vec4f n1, Vec4f n2, float maxAngle);
    static bool criterion2(const Mat &depthImage, std::vector<int> &surface1, std::vector<int> &surface2,
                          Vec4f n1, Vec4f n2, float maxAngle, int triangles);
    static bool criterion3(const Mat &depthImage, std::vector<int> &surface1, std::vector<int> &surface2,
                          float distanceTolerance, float outlierTolerance, int scanlines);
};
