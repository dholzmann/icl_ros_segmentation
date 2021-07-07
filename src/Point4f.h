#include <string>
#include <sstream>
#include <iostream>

struct Point4f{ 
        float x, y, z, w; 
        Point4f(){
            x=0; y=0; z=0; w=0;
        }
        Point4f(float x, float y, float z, float w){
            x=x; y=y; z=z; w=w;
        }

        Point4f& operator+=(Point4f& other){
            x += other.x;
            y += other.y;
            z += other.z;
            w += other.w;
        }

        Point4f& operator/=(float other){
            x /= other;
            y /= other;
            z /= other;
            w /= other;
        }

        float Point4f::operator[](unsigned int idx){
            switch(idx){
                case 0: return x;
                case 1: return y;
                case 2: return z;
                case 3: return w;
                default: throw std::invalid_argument( "received invalid index for Point4f type." );
            }
        }
};