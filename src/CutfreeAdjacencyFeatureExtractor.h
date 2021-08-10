#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>
#include "PlanarRansacEstimator.h"
#include "SurfaceFeatureExtractor.h"
 
 using namespace cv;
/**
   This class implements the cutfree adjacency feature for feature-graph based segmentation.*/
class CutfreeAdjacencyFeatureExtractor{

  public:

  enum Mode {BEST, GPU, CPU};

  /// Constructor
  /** Constructs an object of this class.
      @param mode GPU, CPU and BEST (default) */
  CutfreeAdjacencyFeatureExtractor(Mode mode=BEST);

  /// Destructor
  ~CutfreeAdjacencyFeatureExtractor();

  /// Calculates the cutfree adjacency feature matrix.
  /** @param xyzh the xyzh DataSegment from the PointCloudObject class
      @param surfaces the vector of surface id vectors
      @param testMatrix the initial boolean test matrix (1 test, 0 dont test, preferably an adjacency matrix)
      @param euclideanDistance the maximum euclidean distance for RANSAC in mm
      @param passes the RANSAC passes
      @param tolerance the RANSAC tolerance in number of points (outlier)
      @param labelImage the label image
      @return the boolean cutfree adjacency matrix */
    Mat apply(Mat &xyzh,
            std::vector<std::vector<int> > &surfaces, Mat &testMatrix, float euclideanDistance,
            int passes, int tolerance, Mat labelImage);

    /// Calculates the cutfree adjacency feature matrix with minimum angle constraint.
  /** @param xyzh the xyzh DataSegment from the PointCloudObject class
      @param surfaces the vector of surface id vectors
      @param testMatrix the initial boolean test matrix (1 test, 0 dont test, preferably an adjacency matrix)
      @param euclideanDistance the maximum euclidean distance for RANSAC in mm
      @param passes the RANSAC passes
      @param tolerance the RANSAC tolerance in number of points (outlier)
      @param labelImage the label image
      @param feature the surface feature for the surfaces
      @param minAngle the minimum angle for combination
      @return the boolean cutfree adjacency matrix */
    Mat apply(Mat &xyzh,
            std::vector<std::vector<int> > &surfaces, Mat &testMatrix, float euclideanDistance,
            int passes, int tolerance, Mat labelImage,
            std::vector<SurfaceFeatureExtractor::SurfaceFeature> feature, float minAngle);

  private:

  struct Data;  //!< internal data type
  Data *m_data; //!< internal data pointer

};
