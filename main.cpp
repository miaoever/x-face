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
    //cv::Mat a = cv::Mat::eye(3, 3, CV_64F);
    //cv::invert(a*3,a);
    //std::cout<<a<<std::endl;
    std::cout<<">>Go."<<std::endl;
    MDS mds;
    //mds.Test(10,true);
    mds.Train(10,false);
    //mds.RANK(11);
    
    return 0;
}








