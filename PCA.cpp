//
//  PCA.cpp
//  MDS
//
//
//
//  Created by miaoever on 3/3/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

#include "PCA.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>



cv::PCA PCA(cv::Mat& matrix, int maxComponents)
{
    //cv::Mat t = matrix.t();
    //cv::Scalar mean_t = cv::mean(t);
    cv::PCA pca(matrix, cv::Mat(), CV_PCA_DATA_AS_COL, maxComponents);
    return pca;
}
