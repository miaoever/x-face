//
//  util.cpp
//  LGBP
//
//  Created by miaoever on 4/28/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

#include "util.hpp"
#include "LGBP.hpp"
#include "lbp.hpp"
#include "Gabor.hpp"
#include "histogram.hpp"

vector<cv::Mat> util::LoadImages(string dir_path, bool TransferToRow) {
    cv::Directory dir;
    vector<string> filenames = dir.GetListFiles(dir_path);
    
    vector<cv::Mat> faces;
    for (int i=0;i<filenames.size();i++)
    {
        string filename = filenames[i];
        
        if (filename.substr(filename.length() - 4, 4) != ".jpg")
            continue;
        
        string fullPath = dir_path + filename;
        cv::Mat face = cv::imread(fullPath, -1);
        
        if (TransferToRow) {
            //cv::Mat tmp;
            resize(face, face, cv::Size(face.rows * face.cols, 1));
        }
        faces.push_back(face);
    }
    
    return faces;
}

void util::getLGBPFeature(
                     const string file,
                     Mat& hist,
                     int kernelSize,
                     int neighbor,
                     int radius,
                     bool uniform) {
    
	Mat src=imread(file);
    resize(src, src, Size(100,100));

    cv::cvtColor(src, src, CV_BGR2GRAY);
	normalize(src, src, 1, 0, CV_MINMAX, CV_32FC1);
    lgbp::getLGBP(src, kernelSize, radius, neighbor, uniform, hist);
    //src.convertTo(src, CV_32FC1);
}

void util::getLGBPFeature(const Mat& src,
                           Mat& hist,
                           int kernelSize,
                           int neighbor,
                           int radius,
                           bool uniform) {
    Mat image = src;
    resize(image, image, Size(100,100));
    cv::cvtColor(image, image, CV_BGR2GRAY);
	normalize(image, image, 1, 0, CV_MINMAX, CV_32FC1);
    lgbp::getLGBP(image, kernelSize, radius, neighbor, uniform, hist);
    //src.convertTo(src, CV_32FC1);
}

Mat util::getSiftFeature(Mat src, float facialPoints[], int pointsNum) {
    vector<KeyPoint> keyPoints;
    /*
    for (int j = 0; j < pointsNum; j = j + 2) {
        keyPoints.push_back(KeyPoint(facialPoints[j], facialPoints[j + 1], 500));
    }
     */
    cv::cvtColor(src, src, CV_BGR2GRAY);
    src.convertTo(src, CV_8U);
    cv::SIFT sift;
    cv::Mat output;
    //sift(src, Mat(), keyPoints, output,true);
    sift(src, Mat(), keyPoints, output);
    
    return output.reshape(0, 1);
}


void util::getRegionLGBPFeature(Mat src,
                                Mat& hist,
                                float facialPoints[],
                                int pointsNum,
                                int kernelSize,
                                int neighbor,
                                int radius,
                                bool uniform) {
    int block = 50;
    int nose_block = 50;
    Mat tmp;
    
    cv::Rect myROI(facialPoints[0] - block/2, facialPoints[1] - block/2, block, block);
    myROI = myROI & cv::Rect(0 ,0 , src.cols, src.rows);
    cv::Mat croppedImage = src(myROI).clone();
    resize(croppedImage, croppedImage, Size(block,block));
    cv::cvtColor(croppedImage, croppedImage, CV_BGR2GRAY);
    normalize(croppedImage, croppedImage, 1, 0, CV_MINMAX, CV_32FC1);
    lgbp::getLGBP(croppedImage, kernelSize, radius, neighbor, uniform, hist);

    for (int i = 2; i < pointsNum; i = i + 2) {
        tmp.release();
        cv::Rect myROI;
        if (i != 4 ) {
            //for the region except nose.
            Mat img = src.clone();
            circle(img, Point(facialPoints[i], facialPoints[i + 1]), block / 2, Scalar(0, 0, 0),1,8,0);
            
            cv::Rect tmp(facialPoints[i] - block/2, facialPoints[i + 1] - block/2, block, block);
            myROI = tmp& cv::Rect(0 ,0 , src.cols, src.rows);
            
        } else {
            cv::Rect tmp(facialPoints[i] - nose_block/2, facialPoints[i + 1] - block/2, nose_block, block);
            myROI = tmp & cv::Rect(0 ,0 , src.cols, src.rows);
            
        }
        cv::Mat croppedImage = src(myROI).clone();
        resize(croppedImage, croppedImage, Size(block,block));
        cv::cvtColor(croppedImage, croppedImage, CV_BGR2GRAY);
        normalize(croppedImage, croppedImage, 1, 0, CV_MINMAX, CV_32FC1);
        lgbp::getLGBP(croppedImage, kernelSize, radius, neighbor, uniform, tmp);
        hconcat(hist, tmp, hist);
    }
}


void util::getRegionSIFTFeature(Mat src,
                                Mat& feature,
                                float facialPoints[],
                                int pointsNum) {
    int block = 50;
    int nose_block = 50;
    Mat tmpMat;
    
    cv::Rect myROI(facialPoints[0] - block/2, facialPoints[1] - block/2, block, block);
    myROI = myROI & cv::Rect(0 ,0 , src.cols, src.rows);
    cv::Mat croppedImage = src(myROI).clone();
    resize(croppedImage, croppedImage, Size(block,block));
    //cv::cvtColor(croppedImage, croppedImage, CV_BGR2GRAY);
    //normalize(croppedImage, croppedImage, 1, 0, CV_MINMAX, CV_32FC1);
    feature = getSiftFeature(croppedImage, facialPoints, 10);
    
    for (int i = 2; i < pointsNum; i = i + 2) {
        tmpMat.release();
        cv::Rect myROI;

        cv::Rect tmp(facialPoints[i] - nose_block/2, facialPoints[i + 1] - block/2, nose_block, block);
        myROI = tmp & cv::Rect(0 ,0 , src.cols, src.rows);
            
        
        cv::Mat croppedImage = src(myROI).clone();
        resize(croppedImage, croppedImage, Size(block,block));
        //cv::cvtColor(croppedImage, croppedImage, CV_BGR2GRAY);
        //normalize(croppedImage, croppedImage, 1, 0, CV_MINMAX, CV_32FC1);
        tmpMat = getSiftFeature(croppedImage, facialPoints, 10);
        hconcat(feature, tmpMat, feature);
    }
}


double util::angleToRadian(double angle) {
    return angle * CV_PI / 180;
}

Mat util::rotateImage(const cv::Mat& src, float facialPoints[], int pointsNum) {
    Mat dst;
    int len = std::max(src.cols, src.rows);
    double angle = atan( (facialPoints[1] - facialPoints[3]) / (facialPoints[0] - facialPoints[2] )) * 360 / (2 * CV_PI);
    double theta = abs(angle);
    //angle = abs(angle);
    double alpha = atan((len/2) / (len/2)) * 360 / (2 * CV_PI);
    double beta = (180 - angle) / 2;
    double gamma = 180 - alpha - beta;
    double center_y = 2 * sin(angleToRadian(gamma)) * sin(angleToRadian(theta/2)) * sqrt((len/2) * (len/2) + (len/2) * (len/2));
    double center_x = 2 * cos(angleToRadian(gamma)) * sin(angleToRadian(theta/2)) * sqrt((len/2) * (len/2) + (len/2) * (len/2));
    
    if (facialPoints[3] < facialPoints[1]) {
        swap(center_x, center_y);
        center_y = -center_y;
    } else {
        center_x = -center_x;
    }
    
    cv::Point2f pt(len/2., len/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
    
    cv::warpAffine(src, dst, r, cv::Size(len, len));
    vector<Point2f> points;
    for (int i = 0; i < pointsNum; i = i + 2) {
        double rotatedX = facialPoints[i] * cos(angleToRadian(angle)) + facialPoints[i + 1] * sin(angleToRadian(angle)) + center_x;
        double rotatedY = -facialPoints[i] * sin(angleToRadian(angle)) + facialPoints[i + 1] * cos(angleToRadian(angle)) + center_y;
        facialPoints[i] = rotatedX;
        facialPoints[i + 1] = rotatedY;
        points.push_back(Point2f(facialPoints[i], facialPoints[i + 1]));
    }
    return dst;
}
