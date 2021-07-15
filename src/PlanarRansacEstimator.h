#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>

/// class for planar RANSAC estimation on poincloud data (xyzh).
/** The PlanarRansacEstimator class does not minimize the error but simply counts the points with a distance smaller a given threshold.
    The smaller the threshold the preciser the model. Additionally, it is possible to assign all points on the plane to the initial surface.*/
using namespace cv;
class PlanarRansacEstimator{

  public:
    enum Mode {BEST, GPU, CPU};

    /// Constructor
    /** Constructs an object of this class.
        @param mode the selected mode: CPU, GPU or BEST (uses GPU if available)*/
    PlanarRansacEstimator(Mode mode=BEST);


    /// Destructor
    ~PlanarRansacEstimator();


    typedef struct{
      int numPoints;//number of destination points
      int countOn;//number of points on the model
      int countAbove;//number of points above the model
      int countBelow;//number of points below the model
      float euclideanThreshold;//selected threshold
      Vec4f n0;//best model (normal)
      float dist;//best model (distance)
      int tolerance;//tolerance for ON_ONE_SIDE
      int acc;//number of accepted passes for ON_ONE_SIDE (result smaller tolerance)
      int nacc;//number of rejected passes for ON_ONE_SIDE (result bigger tolerance)
      //int maxID;//for assignment of points (all with this id + on plane)
    }Result;


    enum OptimizationCriterion {
      MAX_ON=1,
      ON_ONE_SIDE=2
    };


    /// Applies the planar RANSAC estimation on a destination region with a model from the src region (e.g. for best fitting plane src=dst).
    /** @param xyzh the input xyzh from the pointcloud
        @param srcIDs vector of IDs (pointcloud position -- for structured x+y*w) of the source surface (for model determination)
        @param dstIDs vector of IDs of the destination surface (for matching)
        @param threshold the maximal euclidean distance in mm
        @param passes number of ransac passes
        @param subset the subset of points for matching (2 means every second point)
        @param tolerance number of points allowed not to be on one single side of the object for ON_ONE_SIDE
        @param optimization the optimization criterion (ON_ONE_SIDE e.g. for cutfree adjacency test or MAX_ON e.g. for best fitting plane)
        @return the Result struct.
    */
    Result apply(Mat &xyzh, std::vector<int> &srcIDs, std::vector<int> &dstIDs,
                float threshold, int passes, int subset, int tolerance, int optimization);


    /// Applies the planar RANSAC estimation on a destination region with a model from the src region (e.g. for best fitting plane src=dst).
    /** @param xyzh the input xyzh from the pointcloud
        @param srcPoints vector of pointcloud points of the source surface (for model determination)
        @param dstIDs vector of pointcloud points of the destination surface (for matching)
        @param threshold the maximal euclidean distance in mm
        @param passes number of ransac passes
        @param subset the subset of points for matching (2 means every second point)
        @param tolerance number of points allowed not to be on one single side of the object for ON_ONE_SIDE
        @param optimization the optimization criterion (ON_ONE_SIDE e.g. for cutfree adjacency test or MAX_ON e.g. for best fitting plane)
        @return the Result struct.
    */
    Result apply(std::vector<Vec4f> &srcPoints, std::vector<Vec4f> &dstPoints, float threshold,
                int passes, int subset, int tolerance, int optimization);


    /// Applies the planar RANSAC estimation on multiple pairs of surfaces (given by a boolean test matrix).
    /** @param xyzh the input xyzh from the pointcloud
        @param pointsIDs vector (surfaces) of vector IDs (pointcloud position -- for structured x+y*w) of the surfaces
        @param testMatrix boolean matrix for surface pair testing (1 = test, 0 = dont test)
        @param threshold the maximal euclidean distance in mm
        @param passes number of ransac passes
        @param tolerance number of points allowed not to be on one single side of the object for ON_ONE_SIDE
        @param optimization the optimization criterion (ON_ONE_SIDE e.g. for cutfree adjacency test or MAX_ON e.g. for best fitting plane)
        @param labelImage the labelImage with the point label
        @return matrix of Result structs.
    */
    Mat_<Result> apply(Mat &xyzh, std::vector<std::vector<int> > &pointIDs,
                Mat &testMatrix, float threshold, int passes, int tolerance, int optimization, Mat labelImage);


    /// Create a label image of all points on the planar model (incl. the original surface)
    /** @param xyzh the input xyzh from the pointcloud
        @param newMask the mask (e.g. ROI) for the relabeling
        @param oldLabel the original label image
        @param newLabel the resulting label image with all points on the model
        @param desiredID the id in the resulting label image
        @param srcID the ID of the label image from the fitted surface
        @param threshold the maximum euclidean distance for relabeling
        @param result the result struct returned by apply (contains the model and the original surface ID)
    */
    void relabel(Mat &xyzh, Mat &newMask, Mat &oldLabel, Mat &newLabel,
                  int desiredID, int srcID, float threshold, Result &result);


    /// Creates random models (n and distance) for RANSAC
    /** @param srcPoints the input points
        @param n0 the empty input n0 vector
        @param dist the empty input distance vector
        @param passes the number of passes
    */
    static void calculateRandomModels(std::vector<Vec4f> &srcPoints, std::vector<Vec4f> &n0, std::vector<float> &dist, int passes);


    /// Creates random models (n and distance) for RANSAC
    /** @param xyzh the input xyz pointcloud data
        @param srcPoints the sourcePoint Ids
        @param n0 the empty input n0 vector
        @param dist the empty input distance vector
        @param passes the number of passes
    */
    static void calculateRandomModels(Mat &xyzh, std::vector<int> &srcPoints, std::vector<Vec4f> &n0, std::vector<float> &dist, int passes);

  private:

    struct Data;  //!< internal data type
    Data *m_data; //!< internal data pointer

    void calculateMultiCL(Mat &xyzh, Mat labelImage, Mat &testMatrix, float threshold, int passes,
                std::vector<Vec4f> &n0, std::vector<float> &dist, std::vector<int> &cAbove, std::vector<int> &cBelow, std::vector<int> &cOn,
                std::vector<int> &adjs, std::vector<int> &start, std::vector<int> &end);

    void calculateMultiCPU(Mat &xyzh, std::vector<std::vector<int> > &pointIDs, Mat &testMatrix,
                float threshold, int passes, std::vector<std::vector<Vec4f> > &n0Pre, std::vector<std::vector<float> > &distPre, std::vector<int> &cAbove,
                std::vector<int> &cBelow, std::vector<int> &cOn, std::vector<int> &adjs, std::vector<int> &start, std::vector<int> &end);

    void calculateSingleCL(std::vector<Vec4f> &dstPoints, float threshold, int passes, int subset,
                std::vector<Vec4f> &n0, std::vector<float> &dist, std::vector<int> &cAbove, std::vector<int> &cBelow, std::vector<int> &cOn);

    void calculateSingleCPU(std::vector<Vec4f> &dstPoints, float threshold, int passes, int subset,
                std::vector<Vec4f> &n0, std::vector<float> &dist, std::vector<int> &cAbove, std::vector<int> &cBelow, std::vector<int> &cOn);

    void initOpenCL();

    Result createResult(std::vector<Vec4f> &n0, std::vector<float> &dist, std::vector<int> &cAbove, std::vector<int> &cBelow, std::vector<int> &cOn,
            float threshold, int passes, int tolerance, int optimization, int numPoints);

    Mat_<Result> createResultMatrix(Mat &testMatrix, std::vector<int> &start, std::vector<int> &end, std::vector<int> &adjs,
                std::vector<int> &cAbove, std::vector<int> &cBelow, std::vector<int> &cOn, std::vector<std::vector<int> > &pointIDs,
                std::vector<std::vector<Vec4f> > &n0Pre, std::vector<std::vector<float> > &distPre, float threshold, int passes, int tolerance, int optimization);

    static void calculateModel(Vec4f &fa, Vec4f &fb, Vec4f &rPoint, Vec4f &n0, float &dist);

    void relabelCL(Mat &xyzh, Mat &newMask, Mat &oldLabel, Mat &newLabel,
                  int desiredID, int srcID, float threshold, Result &result, int w, int h);

    void relabelCPU(Mat &xyzh, Mat &newMask, Mat &oldLabel, Mat &newLabel,
                  int desiredID, int srcID, float threshold, Result &result, int w, int h);

};

