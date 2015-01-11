//
//  MDS.h
//  MDS
//
//  Created by miaoever on 3/3/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

#ifndef __MDS__MDS__
#define __MDS__MDS__

#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "mat.h"

class MDS {
public:
    MDS();
    void RANK(int x);
    void Train(int, bool);
    void Test(int rank , bool Load_Projection_Matrix);
private:
    void LoadTrainLabel(const char* file);
    void LoadTestLabel(const char* file);
    void LoadTrainData(const char* file);
    void LoadTestData(const char* file);
    double GetValue(double* array, size_t nRow, int idx_i, int idx_j);
    void LoadPreproceData(cv::Mat& A, double& B);
    void LoadProjectionMatrix(const char* file);
    void SavePreproceData(cv::Mat& A, double& B);
    void SaveProjectionMatrix();
public :
    int dim_unified;
    float lambda;
    int rank;
    double rate[20];
private:
    double* train_label;
    double* test_l_label;
    double* test_h_label;
    double *val_l;
    double *val_h;
    size_t dim_train_label;
    cv::Mat train_h;
    cv::Mat train_l;
    cv::Mat test_h;
    cv::Mat test_l;
    cv::Mat W;
};


    struct dist {
        double distance = 0;
        int idx = 0;
    };
bool cmp(struct dist a, struct dist b);

#endif /* defined(__MDS__MDS__) */
