#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>
#include "SurfaceFeature.h"
#include "SegmenterHelper.h"

/// class for curvature feature (open and occluded objects).
/** This class implements the curvature feature for feature-graph based segmentation.*/
using namespace cv;
class CurvatureFeature{

  public:

    /// Calculates the curvature feature matrix.
  /** @param depthImg the input depth image
      @param xyz the input xyz pointcloud data segment
      @param initialMatrix the initial boolean test matrix (0 test if both curved, 1 dont test, preferably an adjacency matrix)
      @param features the surface feature for the surfaces
      @param surfaces the vector of surface id vectors
      @param normals the input normals
      @param useOpenObjects true for computation of open objects
      @param useOccludedObjects true for computation of occluded objects
      @param histogramSimilarity the minimum histogram similarity
      @param distance the maximum distance between two curved objects in pixel for open objects (e.g. a cup)
      @param maxError maximum RANSAC error for object alignment detection
      @param ransacPasses number of RANSAC passes for object alignment detection
      @param distanceTolerance distance tolerance for occlusion check
      @param outlierTolerance outlier tolerance for occlusion check
      @return the boolean curvature matrix */
    static Mat apply(Mat &depthImg, Mat &xyz, Mat &initialMatrix,
                      std::vector<SurfaceFeatureExtract::SurfaceRegionFeature> features,
                      std::vector<std::vector<int> > &surfaces, Mat &normals, bool useOpenObjects=true, bool useOccludedObjects=true,
                      float histogramSimilarity=0.5, int distance=10, float maxError=10., int ransacPasses=20, float distanceTolerance=3., float outlierTolerance=5.);

  private:

    static bool computeOpenObject(Mat &normals, SurfaceFeatureExtract::SurfaceRegionFeature feature1, SurfaceFeatureExtract::SurfaceRegionFeature feature2,
                                std::vector<int> &surface1, std::vector<int> &surface2, int distance, int w);

    static bool computeOccludedObject(Mat &depthImg, Mat &xyz, Mat &normals,
                                SurfaceFeatureExtract::SurfaceRegionFeature feature1, SurfaceFeatureExtract::SurfaceRegionFeature feature2,
                                std::vector<int> &surface1, std::vector<int> &surface2, int w, float maxError, int ransacPasses, float distanceTolerance, float outlierTolerance);

    static float computeConvexity(Mat &normals, SurfaceFeatureExtract::SurfaceRegionFeature feature, std::vector<int> &surface, int w);

    static std::pair<Point,Point> computeExtremalBins(SurfaceFeatureExtract::SurfaceRegionFeature feature);

    static std::pair<Point,Point> backproject(Mat &normals,
                    std::pair<Point,Point> &histo1ExtremalBins, std::vector<int> &surface1, int w);

    static std::vector<int> backprojectPointIDs(Mat &normals, Point bin, std::vector<int> &surface);

    static Point computeMean(std::vector<int> &imgIDs, int w);

    static std::vector<Point> createPointsFromIDs(std::vector<int> &imgIDs, int w);

    static std::vector<Vec3f> createPointsFromIDs(Mat &xyz, std::vector<int> &imgIDs);

    static float computeConvexity(std::pair<Point,Point> histoExtremalBins, std::pair<Point,Point> imgBackproject);

    static float linePointDistance(std::pair<Vec3f,Vec3f> line, Vec3f point);

    static Point idToPoint(int id, int w);

};