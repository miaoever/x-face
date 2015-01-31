//
//  util.h
//  LGBP
//
//  Created by miaoever on 4/28/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

//#ifndef __LGBP__util__
//#define __LGBP__util__

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>


#define SIFT_POINTS_NUM 10

using namespace std;
using namespace cv;

namespace util {
    
    vector<cv::Mat> LoadImages(string dir_path, bool TransferToRow);
    
    void getLGBPFeature(
                     const string file,
                     Mat& hist,
                     int kerelSize = 15,
                     int neighbor = 8,
                     int radius = 1,
                     bool uniform = true);
    void getLGBPFeature(
                         const Mat& src,
                         Mat& hist,
                         int kerelSize = 11,
                         int neighbor = 8,
                         int radius = 1,
                         bool uniform = true);
    
    Mat getSiftFeature(Mat src, float facialPoints[], int pointsNum);
    void getRegionLGBPFeature(
                             const Mat src,
                             Mat& hist,
                             float facialPoints[],
                             int pointsNum,
                             int kerelSize = 10,
                             int neighbor = 8,
                             int radius = 1,
                             bool uniform = true);

    void getRegionSIFTFeature(
                               const Mat src,
                               Mat& feature,
                               float facialPoints[],
                               int pointsNum);
    
    cv::Mat rotateImage(const cv::Mat& src, float faicialPoints[], int pointsNum);
    double angleToRadian(double angle);
};

//#endif /* defined(__LGBP__util__) */
