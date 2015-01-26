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
#include "X.h"

cv::Mat PSDProjection(const cv::Mat& W) {
    cv::Mat eigenVector, eigenValue;
    cv::Mat S = cv::Mat::zeros(W.rows, W.cols, W.type());
    if (cv::eigen(W , eigenValue, eigenVector)) {
        int len = eigenVector.rows;
        std::cout<<eigenVector<<std::endl;
        std::cout<<eigenValue<<std::endl;
        for (int i = 0; i < len; i++) {
            ////cv::Mat t = eigenVector.row(i).t() * eigenVector.row(i);
            //cv::Mat t = cv::max(eigenValue.row(i).at<double>(0), 0.0) * eigenVector.row(i).t() * eigenVector.row(i);
            S = S + cv::max(eigenValue.row(i).at<double>(0), 0.0) * eigenVector.row(i).t() * eigenVector.row(i);
        }
    }
    
    return S;
}

int main(int argc, const char * argv[])
{
    cv::Mat a = cv::Mat::eye(3, 3, CV_64F);
   
    a.at<double>(0,2) = 1;
    a.at<double>(2,0) = 1;
    
//    cv::Mat sum1;
//    cv::Mat sum2;
//
//    cv::reduce(a, sum1, 0, CV_REDUCE_SUM);
//    cv::reduce(sum1.t(), sum2, 0, CV_REDUCE_SUM);
//
//    std::cout<<sum2;
    //a = b;
    //std::cout<<a<<std::endl;
    
    //std::cout<<PSDProjection(a)<<std::endl;
    
    

    
    //std::cout<<cv::exp(-600)<<std::endl;
    X x;
    x.Train(1, 70, 0.1, 10, 10, 10);
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








