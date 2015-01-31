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
#include "util.hpp"

std::vector<cv::Mat> Pca(const std::vector<cv::Mat> &samples, int dim) {
    cv::Mat featureMap;
    std::vector<cv::Mat> res;
    PCA pca;

    size_t len = samples.size();
    for (int i = 0; i < len; i++) {
        featureMap.push_back(samples[i]);
    }
    
    featureMap.convertTo(featureMap, CV_32FC1);
    pca(featureMap, Mat(), CV_PCA_DATA_AS_ROW, dim);
    
    for (int i = 0; i < len; i++) {
        res.push_back(pca.project(samples[i]));
    }
    
    return res;
}

int main(int argc, const char * argv[])
{
    
    X x;
    //void X::Train(double lambda, int dim_unified, double step, double gamma1, double gamma2, int Max_iteration) {
    
    std::vector<cv::Mat> train_h = Pca(util::LoadImages("/Users/miaoever/Project/Matlab/Feret/High/", true), 125);
    std::vector<cv::Mat> train_l = Pca(util::LoadImages("/Users/miaoever/Project/Matlab/Feret/Low/", true), 70);
    
   
    //x.Train(1, 70, 0.1, 6, 5, 20);
    //MDS mds;
    //mds.Test(10,true);
    //mds.Train(10,false);
    //mds.RANK(11);
    
    return 0;
}








