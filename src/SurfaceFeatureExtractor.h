#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <math.h>

/// class for extraction of surface features.
/** The SurfaceFeatureExtractor class computes normalHistograms, meanNormals, meanPosition, and curvatureFactor(planar, 1D curved, 2D curved) for sets of points. */
using namespace cv;

class SurfaceFeatureExtractor{

  public:
    enum Mode {
      NORMAL_HISTOGRAM = 1,
      CURVATURE_FACTOR = 2,
      MEAN_NORMAL = 4,
      MEAN_POSITION = 8,
      BOUNDING_BOX_3D = 16,
      BOUNDING_BOX_2D = 32,
      ALL = 63
    };

    typedef struct{
        int numPoints;//number of surface points
        //core::Img32f normalHistogram;//normal histogram (11x11 bins representing x and y component (each from -1 to 1 in 0.2 steps)
        Mat normalHistogram;
        //core::Channel32f normalHistogramChannel;
        //Vec meanNormal;//mean normal
        Vec4f meanNormal;
        //Vec meanPosition;//mean position
        Vec4f meanPosition;
        int curvatureFactor;//curvature Factor from enum CurvatureFactor
        std::pair<Vec4f,Vec4f> boundingBox3D;
        std::pair<Point2f,Point2f> boundingBox2D;
        float volume;//volume of the 3D bounding box
    }SurfaceFeature;


    enum CurvatureFactor {
      UNDEFINED=0,
      PLANAR=1,
      CURVED_1D=2,
      CURVED_2D=3
    };


    /// Applies the surface feature calculation for one single surface (please note: no BoundingBox2D)
    /** @param points the points xyz
        @param normals the corresponding point normals
        @param mode the mode from Mode enum (e.g. A | B)
        @return the SurfaceFeature struct.
    */
    static SurfaceFeature apply(std::vector<Vec4f> &points, std::vector<Vec4f> &normals, int mode=ALL);

    /// Applies the surface feature calculation for all segments in the label image
    /** @param labelImage the label image of the segmentation
        @param xyzh the xyz pointcloud
        @param normals the corresponding point normals
        @param mode the mode from Mode enum (e.g. A | B)
        @return a vector of SurfaceFeature struct for each segment.
    */
    static std::vector<SurfaceFeature> apply(Mat labelImage, Mat &xyzh, Mat &normals, int mode=ALL);

    /// Calculates the matching score between 0 and 1 for two normal histograms.
    /** @param a the first normal histogram
        @param b the second normal histogram
        @return the matching score between 0 (no matching) and 1 (perfect matching).
    */
    static float matchNormalHistograms(Mat &a, Mat &b);

  private:

    static SurfaceFeature getInitializedStruct();
    static void update(Vec4f &normal, Vec4f &point, SurfaceFeature &feature, int mode, int x=0, int y=0);
    static void finish(SurfaceFeature &feature, int mode);
};
