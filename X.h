//
//  X.h
//  MDS
//
//  Created by miaoever on 1/19/15.
//  Copyright (c) 2015 miaoever. All rights reserved.
//
#ifndef MDS_X_h
#define MDS_X_h

#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mat.h"


typedef struct Point{
    int x;
    int y;
} Point;

class X {
public:
    X();
    void RANK(int x);
    void Train();
    void Test(int rank , bool Load_Projection_Matrix);
private:
    void LoadTrainLabel(const char* file);
    void LoadTestLabel(const char* file);
    void LoadTrainData(const char* file);
    void LoadTestData(const char* file);
    
    std::vector<std::vector<bool>> Knn(int);
    void Gradient_descent();
    cv::Mat BuildGraph(const cv::Mat&);
    void Normalization(cv::Mat&);
    cv::Mat Calc_diff(const cv::Mat&);
    cv::Mat PSDProjection(const cv::Mat&);
    double Calc_value_obj(const cv::Mat&);
    
private:
    double* train_label;
    double* test_l_label;
    double* test_h_label;
    double *val_l;
    double *val_h;
    size_t dim_train_label;
    
    double lambda;
    double gamma1;
    double gamma2;
    
    cv::Mat W;
    cv::Mat P;
    cv::Mat theta_l;
    cv::Mat theta_h;
    
    cv::Mat train_l;
    cv::Mat train_h;
    int dim_unified;
    int Max_iteration;
    int step;
    
    std::vector<std::vector<bool>> adj;
    std::vector<Point> NonZeroAdj;
    //cv::Mat weight;
};



#endif
