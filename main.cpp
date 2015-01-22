//
//  main.cpp
//  MDS
//
//  Created by miaoever on 3/3/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

#include <iostream>
#include <string>
#include "MDS.h"


int main(int argc, const char * argv[])
{
    cv::Mat a = cv::Mat::ones(3, 3, CV_64F);
    a.at<double>(0,0) = 0;
    a.at<double>(0,2) = 0;

     //cv::invert(a*3,a);
    //std::cout<<a<<std::endl;

    //Normalization(a);
    
    //PSDProjection(a);
    //std::cout<<PSDProjection(a)<<std::endl;

    //MDS mds;
    //mds.Test(10,true);
    //mds.Train(10,false);
    //mds.RANK(11);
    
    return 0;
}








