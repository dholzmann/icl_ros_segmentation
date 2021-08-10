#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/matx.hpp>

/// class for region growing on images and DataSegments (e.g. poincloud xyzh)
/** The RegionGrower class is designed as template applying a growing criterion to given input data.
    A mask defines the points for processing (e.g. a region of interest).
*/
using namespace cv;

class RegionGrower{

  public:

    /// Applies the region growing on an input image with a growing criterion
    /** @param image the input image for region growing
        @param crit the region growing criterion
        @param initialMask the initial mask (e.g. ROI)
        @param minSize the minimum size of regions (smaller regions are removed)
        @param startID the start id for the result label image
        @return the result label image
    */
    template<class Criterion>
    const Mat &apply(const Mat_<int> &image, Criterion crit, Mat *initialMask = 0,
                        const unsigned int minSize=0, const unsigned int startID=1){
      this->result=Mat(image);
      //this->mask=Img8u(image.getParams());
      Mat &useMask = initialMask ? *initialMask : this->mask;
      region_grow<Mat,int,1, Criterion>(image, useMask, this->result, crit, minSize, startID);
      return this->result;
    }

    /// Applies the region growing on an input image with a growing criterion
    /** @param image the input image for region growing
        @param crit the region growing criterion
        @param initialMask the initial mask (e.g. ROI)
        @param minSize the minimum size of regions (smaller regions are removed)
        @param startID the start id for the result label image
        @return the result label image
    */
    const Mat &applyColorImageGrowing(const Mat_<int> &image, float const th, Mat *initialMask = 0,
                        const unsigned int minSize=0, const unsigned int startID=1){
      this->result=Mat(image.size(),1);
      Mat &useMask = initialMask ? *initialMask : this->mask;
      region_grow<Mat,int,3, U8EuclideanDistance>(image, useMask, this->result, U8EuclideanDistance(th), minSize, startID);
      return this->result;
    }

    /// Applies the region growing on an input data segment with a growing criterion
    /** @param dataseg the input data segment for region growing
        @param crit the region growing criterion
        @param initialMask the initial mask (e.g. ROI)
        @param minSize the minimum size of regions (smaller regions are removed)
        @param startID the start id for the result label image
        @return the result label image
    */
    template<class Criterion>
    const Mat &apply(const Mat_<float> &dataseg, Criterion crit, Mat *initialMask = 0,
                        const unsigned int minSize=0, const unsigned int startID=1){
      Mat &useMask = initialMask ? *initialMask : this->mask;
      this->result = cv::Mat(dataseg.size(), CV_32FC1);
      region_grow<Mat,float,4, Criterion>(dataseg, useMask, this->result, crit, minSize, startID);
      return this->result;
    }


    /// Applies the region growing on an input data segment with euclidean distance criterion
    /** @param dataseg the input data segment for region growing
        @param mask the initial mask (e.g. ROI)
        @param threshold the maximum euclidean distance
        @param minSize the minimum size of regions (smaller regions are removed)
        @param startID the start id for the result label image
        @return the result label image
    */
    const Mat &applyFloat4EuclideanDistance(const Mat_<float> &dataseg, Mat mask,
                        const int threshold, const unsigned int minSize=0, const unsigned int startID=1){
      return apply(dataseg, Float4EuclideanDistance(threshold), &mask, minSize, startID);
    }


    /// Applies the region growing on an input image with value-equals-threshold criterion
    /** @param image the input image for region growing
        @param mask the initial mask (e.g. ROI)
        @param threshold the equals-to-value (growing criterion)
        @param minSize the minimum size of regions (smaller regions are removed)
        @param startID the start id for the result label image
        @return the result label image
    */
    const Mat &applyEqualThreshold(const Mat_<int> &image, Mat mask, const int threshold,
                        const unsigned int minSize=0, const unsigned int startID=1){
      return apply(image, EqualThreshold(threshold), &mask, minSize, startID);
    }


    /// Returns a vector of regions containing the image IDs. This is an additional representation of the result.
    /** @return the vector of regions with the image IDs.
    */
    std::vector<std::vector<int> > getRegions(){
      return regions;
    }


  private:

    Mat mask;
    Mat result;
    std::vector<std::vector<int> > regions;

    template<class T, class DataT, int DIM>
    struct RegionGrowerDataAccessor{
      RegionGrowerDataAccessor(const T &t){};
      int w() const { return 0; }
      int h() const { return 0; }
      Vec<DataT, DIM> operator()(int x, int y) const { return Vec<DataT,DIM>(); }
    };

    static float dist3u8(const Vec3i &a, const Vec3i &b) {
        Vec3i c = b-a;
        return sqrt( c[0]*c[0] + c[1]*c[1] + c[2]*c[2] );
    }

    struct U8EuclideanDistance{
        float t;
        U8EuclideanDistance(float t):t(t){}
        bool operator() (const Vec3i &a, const Vec3i &b) const {
            return dist3u8(a,b) < t;
        }
    };

    struct EqualThreshold{
      int t;
      EqualThreshold(int t):t(t){}
      bool operator()(int a, int b) const{
        return (int)b == t;
      }
    };

    struct Float4EuclideanDistance{
      float t;
      Float4EuclideanDistance(float t):t(t){}
      bool operator()(const Vec4f &a, const Vec4f &b) const{
        return norm(a,b, NORM_L2) < t;
      }
      bool operator()(const Vec2i &a, const Vec2i &b) const{
        return norm(a,b, NORM_L2) < t;
      }
      bool operator()(const Vec3f &a, const Vec3f &b) const{
        return norm(a,b, NORM_L2) < t;
      }
    };


    template<class T, class DataT, int DIM, class Criterion>
    static void flood_fill(const RegionGrowerDataAccessor<T,DataT,DIM> &a, int xStart, int yStart,
                            Mat &processed, Criterion crit, std::vector<int> &result,  Mat &result2, int id);


    template<class T, class DataT, int DIM, class Criterion>
    void region_grow(const T &data, Mat &mask, Mat &result, Criterion crit, const unsigned int minSize, const unsigned int startID=1){
      RegionGrowerDataAccessor<T,DataT,DIM> a(data);

      Mat processed = mask;
      Mat p = processed;
      std::vector<int> r;
      Mat res = result;
      result = 0;

      int nextID = startID;
      regions.clear();

      std::vector<std::vector<int> > clear;

      for(int y=0;y<a.h();++y){
        for(int x=0;x<a.w();++x){
          if(!p.at<bool>(x,y) && crit(a(x,y),a(x,y))){
            r.clear();
            flood_fill<T,DataT,DIM,Criterion>(a ,x ,y ,p, crit, r, res, nextID++);
            if(r.size()<minSize){
              nextID--;
              clear.push_back(r);//delete later
            }else{
              regions.push_back(r);//add region
            }
          }
        }
      }

      //clear regions smaller minSize
      for(unsigned int i=0; i<clear.size(); i++){
        for(unsigned int j=0; j<clear.at(i).size(); j++){
          p.at<bool>(clear.at(i).at(j))=false;
          res.at<int>(clear.at(i).at(j))=0;
        }
      }
    }

  };


  template<>
  struct RegionGrower::RegionGrowerDataAccessor<Mat, int, 1>{
    const Mat c;
    RegionGrowerDataAccessor(const Mat &image):c(image){}
    int w() const { return c.size().width; }
    int h() const { return c.size().height; }
    int operator()(int x, int y) const { return c.at<int>(x,y); }
  };

  template<>
  struct RegionGrower::RegionGrowerDataAccessor<Mat, int, 3>{
    Mat c;
      RegionGrowerDataAccessor(const Mat &image){
        c = Mat(image);
      }
    int w() const { return c.size().width; }
    int h() const { return c.size().height; }
    Vec3i operator()(int x, int y) const {
      //return Vec3i(c.at<int>(x,y,0), c.at<int>(x,y,1), c.at<int>(x,y,2));
      return c.at<Vec3i>(x,y);
    }
  };

  template<>
  struct RegionGrower::RegionGrowerDataAccessor<Mat, float, 4>{
    Mat data;
    int ww,hh;
    RegionGrowerDataAccessor(const Mat &data):data(data){
      ww = data.size().width;
      hh = data.size().height;
    }
    int w() const { return ww; }
    int h() const { return hh; }
    Vec4f operator()(int x, int y) const { return data.at<Vec4f>(x,y); }
  };

  template<class T, class DataT, int DIM, class Criterion>
  void RegionGrower::flood_fill(const RegionGrowerDataAccessor<T,DataT,DIM> &a, int xStart, int yStart,
                            Mat &processed, Criterion crit, std::vector<int> &result,  Mat &result2, int id){
    std::vector<Point> stack(1,Point(xStart,yStart));
    processed.at<bool>(xStart,yStart) = true;//update mask
    result2.at<int>(xStart,yStart) = id;//update result image
    result.push_back(xStart+yStart*a.w());//add to region vector
    unsigned int next = 0;
    while(next < stack.size()){
      const Point p = stack[next];
      next++;
      for(int dy=-1;dy<=1;++dy){
        const int y = p.y+dy;
        if(y < 0 || y >=a.h()) continue;
        for(int dx=-1;dx<=1;++dx){
          const int x = p.x+dx;
          if(x < 0 || x >=a.w()) continue;
          if(dx==0 && dy==0) continue;

          if(crit(a(p.x,p.y),a(x,y)) && processed.at<bool>(x,y)==false){
            stack.push_back(Point(x,y));
            processed.at<bool>(x,y) = true;
            result2.at<int>(x,y) = id;
            result.push_back(x+y*a.w());
          }
        }
      }
    }
  }
