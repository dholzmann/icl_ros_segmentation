#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#include <string>
#include <sstream>
#include <iostream>
#include <math.h>

#include "SegmenterHelper.h"
#include "CutfreeAdjacencyFeature.h"
#include "CurvatureFeature.h"
#include "CoPlanarityFeature.h"
#include "RemainingPointsFeature.h"
#include "GraphCut.h"

using namespace cv;

namespace ObjectSegmenter{

    struct Data {
        Data(int height, int width){
            maskImage = cv::Mat::zeros(height, width, CV_8UC1);
            labelImage = cv::Mat::zeros(height, width, CV_16UC1);
            xMinROI = 0, xMaxROI = 0;
            yMinROI = 0, yMaxROI = 0;
            zMinROI = 0, zMaxROI = 0;
            
            llCorner = 0;
            lrCorner = 0;
            ulCorner = 0;
            urCorner = 0;
            
            minSurfaceSize = 25;
            
            assignmentRadius=5;
            assignmentDistance=10;

            cutfreeRansacEuclideanDistance=5.0;
            cutfreeRansacPasses=20;
            cutfreeRansacTolerance=30;
            cutfreeMinAngle=30.;
            
            coplanarityMaxAngle=30.0;
            coplanarityDistanceTolerance=3.0;
            coplanarityOutlierTolerance=5.0;
            coplanarityNumTriangles=20;
            coplanarityNumScanlines=9;
            
            curvatureHistogramSimilarity=0.5;
            curvatureUseOpenObjects=true;
            curvatureMaxDistance=10;
            curvatureUseOccludedObjects=true;
            curvatureMaxError=10.0;
            curvatureRansacPasses=20;
            curvatureDistanceTolerance=3.0;
            curvatureOutlierTolerance=5.0;                        
        
            remainingMinSize=10;
            remainingEuclideanDistance=5.0;
            remainingRadius=6;
            remainingAssignEuclideanDistance=5.0;
            remainingSupportTolerance=9;    
            
            graphCutThreshold=0.5;
        }

        ~Data() {
        }

        CutfreeAdjacencyFeature* cutfree;
        SegmenterHelper* segUtils;
        //RegionDetector* region;
            
        std::vector<std::vector<int>> surfaces;
        std::vector<std::vector<int>> segments;

        cv::Mat maskImage= cv::Mat::zeros(480, 640, CV_8UC1);
        cv::Mat labelImage= cv::Mat::zeros(480, 640, CV_16UC1);

        std::vector<SurfaceFeatureExtract::SurfaceRegionFeature> features;

        float xMinROI, xMaxROI, yMinROI, yMaxROI, zMinROI, zMaxROI;

        int llCorner, lrCorner, ulCorner, urCorner;

        unsigned int minSurfaceSize;

        int surfaceSize;

        int assignmentRadius;
        float assignmentDistance;

        float cutfreeRansacEuclideanDistance;
        int cutfreeRansacPasses;
        int cutfreeRansacTolerance;
        float cutfreeMinAngle;
            
        float coplanarityMaxAngle;
        float coplanarityDistanceTolerance;
        float coplanarityOutlierTolerance;
        int coplanarityNumTriangles;
        int coplanarityNumScanlines;

        float curvatureHistogramSimilarity;
        bool curvatureUseOpenObjects;
        int curvatureMaxDistance;
        bool curvatureUseOccludedObjects;
        float curvatureMaxError;
        int curvatureRansacPasses;
        float curvatureDistanceTolerance;
        float curvatureOutlierTolerance;                        

        int remainingMinSize;
        float remainingEuclideanDistance;
        int remainingRadius;
        float remainingAssignEuclideanDistance;
        int remainingSupportTolerance;
        
        float graphCutThreshold;
        };  

        cv::Mat apply(Data &data, cv::Mat xyz, const cv::Mat &edgeImage, cv::Mat &depthImg, cv::Mat normals, bool stabilize, bool useROI, bool useCutfreeAdjacency, 
            bool useCoplanarity, bool useCurvature, bool useRemainingPoints);

        cv::Mat applyHierarchy(Data &data, cv::Mat xyz, const Mat &edgeImage, Mat &depthImg, Mat normals, bool stabilize, bool useROI, bool useCutfreeAdjacency, 
            bool useCoplanarity, bool useCurvature, bool useRemainingPoints,
            float weightCutfreeAdjacency, float weightCoplanarity, float weightCurvature, float weightRemainingPoints);
    
        void surfaceSegmentation(Data &data, cv::Mat &xyz, const cv::Mat &edgeImg, cv::Mat &depthImg, int minSurfaceSize, bool useROI);

        Mat getMaskImage(Data &data);

        void setROI(Data &data, float xMin, float xMax, float yMin, float yMax, float zMin, float zMax);

        Mat getColoredLabelImage(Data &data, bool stabelize);

        Mat getLabelImage(Data &data, bool stabelize);

        void setCornerRemoval (Data &data, int ll, int lr, int ul, int ur);

        void setMinSurfaceSize(Data &data, unsigned int size);

        void setAssignmentParams(Data &data, float distance, int radius);

        void setCutfreeParams(Data &data, float euclideanDistance, int passes, int tolerance, float minAngle);

        void setCoplanarityParams(Data &data, float maxAngle, float distanceTolerance, float outlierTolerance, int numTriangles, int numScanlines);

        void setCurvatureParams(Data &data, float histogramSimilarity, bool useOpenObjects, int maxDistance, bool useOccludedObjects, float maxError, 
                        int ransacPasses, float distanceTolerance, float outlierTolerance);

        void setRemainingPointsParams(Data &data, int minSize, float euclideanDistance, int radius, float assignEuclideanDistance, int supportTolerance);

        void setGraphCutThreshold(Data &data, float threshold);

        std::vector<SurfaceFeatureExtract::SurfaceRegionFeature> getSurfaceFeatures(Data &data);

        std::vector<std::vector<int> > getSegments(Data &data);

        std::vector<std::vector<int> > getSurfaces(Data &data);
        //struct Data;  //!< internal data type
        //Data *m_data; //!< internal data pointer

} // namespace Objectsegmenter