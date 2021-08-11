#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "PlanarRansac.h"
#include "RegionGrowing.h"
/// class for remaining points feature.
/** This class implements the remaining points feature for feature-graph based segmentation.*/
using namespace cv;
class RemainingPointsFeature{

  public:

    /// Calculates and assigns the remaining points segments (including local minima separation).
  /** @param xyz the input xyz pointcloud data segment
      @param depthImage the input depth image
      @param labelImage the label image (modified by the method)
      @param maskImage the mask image (modified by the method)
      @param surfaces the surfaces (modified by the method)
      @param segments the segments (modified by the method)
      @param minSize the minimum size of remaining points segments
      @param euclideanDistance the euclidean distance for clustering
      @param radius the radius for local minima separation
      @param assignEuclideanDistance the euclidean distance for assigning to neighbouring surfaces
      @param supportTolerance the max tolerance for detecting a blob as supported and therefore separate object */
    static void apply(Mat &xyz, const Mat &depthImage, Mat &labelImage, Mat &maskImage,
                      std::vector<std::vector<int> > &surfaces, std::vector<std::vector<int> > &segments, int minSize=10, float euclideanDistance=5., int radius=6, float assignEuclideanDistance=5., int supportTolerance=9);

    /// Calculates and assigns the remaining points segments (excluding local minima separation).
  /** @param xyz the input xyz pointcloud data segment
      @param labelImage the label image (modified by the method)
      @param maskImage the mask image (modified by the method)
      @param surfaces the surfaces (modified by the method)
      @param segments the segments (modified by the method)
      @param minSize the minimum size of remaining points segments
      @param euclideanDistance the euclidean distance for clustering
      @param assignEuclideanDistance the euclidean distance for assigning to neighbouring surfaces
      @param supportTolerance the max tolerance for detecting a blob as supported and therefore separate object */
    static void apply(Mat &xyz, Mat &labelImage, Mat &maskImage,
                      std::vector<std::vector<int> > &surfaces, std::vector<std::vector<int> > &segments, int minSize=10, float euclideanDistance=5., float assignEuclideanDistance=5., int supportTolerance=9);

    /// Calculates the remaining points segments and creates a connectivity matrix (including local minima separation).
  /** @param xyz the input xyz pointcloud data segment
      @param depthImage the input depth image
      @param labelImage the label image (modified by the method)
      @param maskImage the mask image (modified by the method)
      @param surfaces the surfaces (modified by the method)
      @param minSize the minimum size of remaining points segments
      @param euclideanDistance the euclidean distance for clustering
      @param radius the radius for local minima separation
      @param assignEuclideanDistance the euclidean distance for assigning to neighbouring surfaces
      @return the connectivity matrix*/
    static Mat_<bool> apply(Mat &xyz, const Mat &depthImage, Mat &labelImage, Mat &maskImage,
                      std::vector<std::vector<int> > &surfaces, int minSize=10, float euclideanDistance=5., int radius=6, float assignEuclideanDistance=5.);

    /// Calculates the remaining points segments and creates a connectivity matrix (excluding local minima separation).
  /** @param xyz the input xyz pointcloud data segment
      @param labelImage the label image (modified by the method)
      @param maskImage the mask image (modified by the method)
      @param surfaces the surfaces (modified by the method)
      @param minSize the minimum size of remaining points segments
      @param euclideanDistance the euclidean distance for clustering
      @param assignEuclideanDistance the euclidean distance for assigning to neighbouring surfaces
      @return the connectivity matrix */
    static Mat_<bool> apply(Mat &xyz, Mat &labelImage, Mat &maskImage,
                      std::vector<std::vector<int> > &surfaces, int minSize=10, float euclideanDistance=5., float assignEuclideanDistance=5.);

  private:

    static void calculateLocalMinima(const Mat &depthImage, Mat &maskImage, int radius);

    static void clusterRemainingPoints(Mat &xyz, std::vector<std::vector<int> > &surfaces, Mat &labelImage, Mat &maskImage,
                                        int minSize, float euclideanDistance, int numCluster);

    static void detectNeighbours(Mat &xyz, std::vector<std::vector<int> > &surfaces, Mat &labelImage, std::vector<std::vector<int> > &neighbours,
                                  std::vector<std::vector<int> > &neighboursPoints, int numCluster, float assignEuclideanDistance);

    static bool checkNotExist(int zw, std::vector<int> &nb, std::vector<int> &nbPoints);

    static void ruleBasedAssignment(Mat &xyz, Mat &labelImage, std::vector<std::vector<int> > &surfaces, std::vector<std::vector<int> > &segments,
                                    std::vector<std::vector<int> > &neighbours, std::vector<std::vector<int> > &neighboursPoints, int numCluster, int supportTolerance);

    static std::vector<int> segmentMapping(std::vector<std::vector<int> > &segments, int numSurfaces);

    static int ransacAssignment(Mat &xyz, std::vector<std::vector<int> > &surfaces, std::vector<int> &nb, int x);

    static bool checkSupport(Mat &labelImage, std::vector<int> &surface, int neighbourID, int supportTolerance);
};
