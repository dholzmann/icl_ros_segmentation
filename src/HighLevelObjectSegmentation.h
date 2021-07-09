#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>

#include <string>
#include <sstream>
#include <iostream>
#include <math.h>

#include "Point4f.h"
#include "SurfaceFeatureExtractor.h"
#include "SegmenterUtils.h"

namespace ObjectSegmenter{
    using namespace cv;

    struct Data {

            Data(int height, int width){
                maskImage = Mat(height, width, CV_8U);
                labelImage = Mat(height, width, CV_32S);
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
        
        CutfreeAdjacencyFeatureExtractor* cutfree;
        SegmenterUtils* segUtils;
        //RegionDetector* region;
                
        std::vector<std::vector<int> > surfaces;
        std::vector<std::vector<int> > segments;

        Mat maskImage;
        Mat labelImage;

        std::vector<SurfaceFeatureExtractor::SurfaceFeature> features;
        
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
}