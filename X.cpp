//
//  X.cpp
//  MDS
//
//  Created by miaoever on 1/19/15.
//  Copyright (c) 2015 miaoever. All rights reserved.
//

#include "X.h"
#include <iostream>
#include <limits>

#define __DEBUG__

typedef struct Pos {
    double value;
    int idx;
} Pos;

bool cmp (Pos a, Pos b) {
    return (a.value < b.value );
}

double GetValue(double* array, size_t nRow, int idx_i, int idx_j){
    return array[nRow * idx_j + idx_i];
}

double lowest_double = 0.00000000000001;


X::X() {
    std::cout<<"Loading data ..."<<std::endl;
//    LoadTrainLabel("/Users/miaoever/Project/MDS/MDS/mat/cls_label_feret.mat");
//    LoadTrainData("/Users/miaoever/Project/MDS/MDS/mat/MDS_train.mat");
    LoadTrainData("/Users/miaoever/Desktop/train.mat");
    LoadTrainLabel("/Users/miaoever/Desktop/train_Label.mat");
    
}

void X::Train(double lambda, int dim_unified, int step, double gamma1, double gamma2, int Max_iteration) {
    // 4 - the scale of neighborhood
    //cv::Mat adj = Knn(4);
    
    this->lambda = lambda;
    this->dim_unified = dim_unified;
    this->step = step;
    this->gamma1 = gamma1;
    this->gamma2 = gamma2;
    this->Max_iteration = Max_iteration;

    int type = train_l.type();
    int num_vector = train_l.cols;
    int dim_l = train_l.rows;
    int dim_h = train_h.rows;
  
//    cv::Mat w1(dim_unified, dim_l, type);
//    cv::Mat w2(dim_unified, dim_h, type);
    
//    cv::randu(w1, cv::Scalar::all(-1), cv::Scalar::all(1));
//    cv::randu(w2, cv::Scalar::all(-1), cv::Scalar::all(1));
    
    cv::Mat w1 = cv::Mat::eye(dim_unified, dim_l, type);
    cv::Mat w2 = cv::Mat::eye(dim_unified, dim_h, type);

    cv::vconcat(w1.t(), w2.t(), W);
    
    //W = cv::Mat::ones(W.rows, W.cols, type);
    W = W.t();
    W = W.t() * W;

    cv::vconcat(train_l, cv::Mat::zeros(dim_h, num_vector, type), theta_l);
    cv::vconcat(cv::Mat::zeros(dim_l, num_vector, type), train_h, theta_h);
    adj = Knn(4);
    Gradient_descent();
}

void X::Gradient_descent() {
    P = BuildGraph(W, 1);
    cv::Mat _W, diff_W;
    //std::cout<<P<<std::endl;

    _W = W;
    //std::cout<<P;

    for (int i = 0; i < Max_iteration; i ++) {
        //diff_W = ;
        _W -= step * Calc_diff(_W);
        _W = PSDProjection(_W);
        double _Value = Calc_value_obj(_W);
        
//#ifdef __DEGUB__
        //std::printf("term: %d     value: %f\n", i, _Value);
        std::cout.precision(15);
        std::cout<<_Value<<std::endl;
//#endif
    }
    W = _W;
}

cv::Mat X::BuildGraph(const cv::Mat& W, int type) {
    cv::Mat weight = cv::Mat(theta_l.cols, theta_l.cols, CV_64FC1, double(0));
    double gamma = 0.0;
    
    type == 1 ? gamma = gamma1 : gamma = gamma2;

    size_t len = NonZeroAdj.size();
    for (int i = 0; i < len; i++) {
        cv::Mat tmp =  ((theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y)).t() * W) * (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y));
        weight.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y) = cv::exp(- tmp.at<double>(0) / (2 * gamma));

//        if (type == 1 && static_cast<int>(GetValue(train_label, dim_train_label, NonZeroAdj[i].x, NonZeroAdj[i].y)) == 1) {
//            weight.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y) = 1;
//        } else {
//            cv::Mat tmp =  ((theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y)).t() * W) * (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y));
//            //std::cout<<tmp<<std::endl;
//            weight.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y) = cv::exp(- tmp.at<double>(0) / (2 * gamma));
//        }
    }
    Normalization(weight, type);
    return weight;
}

void X::Normalization(cv::Mat& weight, int type) {
    size_t r = weight.rows;
    size_t c = weight.cols;
    //cv::Mat sum = cv::Mat(weight.rows, weight.cols, weight.type());
    cv::Mat sum;
    cv::Mat _w = weight.clone();
    
    //calc the sum of each row for weight.
    cv::reduce(weight.t(), sum, 0, CV_REDUCE_SUM);
    for (int i = 0; i < r; i ++) {
        for (int j = 0; j < c; j ++) {
            weight.at<double>(i,j) = _w.at<double>(i, j) / (sum.col(i).at<double>(0) - _w.at<double>(i, i) + lowest_double);
        }
    }
    
    //P = weight;
    
    if (type == 1) {
        size_t len = NonZeroAdj.size();
        for (int i = 0; i < len; i++) {
            if (static_cast<int>(GetValue(train_label, dim_train_label, NonZeroAdj[i].x, NonZeroAdj[i].y)) == 1) {
                weight.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y) = 1;
            }
        }
    }
}

std::vector<std::vector<bool>> X::Knn(int k) {
    int num = theta_l.cols;
    std::vector<std::vector<double>> dist(num, std::vector<double>(num));
    std::vector<std::vector<bool>> adj(num, std::vector<bool>(num));
    
    for (int i = 0; i < num; i ++) {
        for (int j = 0; j < num; j ++) {
            dist[i][j] = 0;
            adj[i][j] = false;
        }
    }
    
    for (int i = 0; i < num; i ++) {
        for (int j = 0; j < num; j ++) {
            if (dist[i][j] == 0) {
                dist[i][j] = cv::norm(theta_h.col(i) - theta_h.col(j));
                dist[j][i] = dist[i][j];
            }
        }
    }
    
    for (int i = 0; i < num; i ++) {
       
        std::vector<Pos> row;
        for (int j = 0; j < num; j ++) {
            Pos p = {dist[i][j], j};
            row.push_back(p);
        }
       
        std::sort(row.begin(), row.end(), cmp);
        
        for (int j = 0; j < k; j ++) {
            adj[i][row[j].idx] = true;
            Point p = {i, row[j].idx};
            NonZeroAdj.push_back(p);
        }
    }
    
    return adj;
}

cv::Mat X::Calc_diff(const cv::Mat& W) {
    
    cv::Mat Q = BuildGraph(W, 0);
    cv::Mat M = cv::Mat::zeros(W.rows, W.cols, W.type());
    
    size_t len = NonZeroAdj.size();
    for (int i = 0; i < len; i++) {
        M = M + lambda * (P.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y) - Q.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y)) * (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y)) * (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y)).t();
    }
    
    return (1 / (2 * gamma2)) * M;
    //return M;

}

cv::Mat X::PSDProjection(const cv::Mat& W) {
    cv::Mat eigenVector, eigenValue;
    cv::Mat S = cv::Mat::zeros(W.rows, W.cols, W.type());
    if (cv::eigen(W, eigenValue, eigenVector)) {
        int len = eigenVector.rows;
        
        for (int i = 0; i < len; i++) {
            ////cv::Mat t = eigenVector.row(i).t() * eigenVector.row(i);
            //cv::Mat t = cv::max(eigenValue.row(i).at<double>(0), 0.0) * eigenVector.row(i).t() * eigenVector.row(i);
            S = S + cv::max(eigenValue.row(i).at<double>(0), 0.0) * eigenVector.row(i).t() * eigenVector.row(i);
        }
    }

    return S;
}

double X::Calc_value_obj(const cv::Mat& W) {
    
    //constexpr double lowest_double = cv::abs(std::numeric_limits<double>::lowest());
    double lowest_double = 0.000000000000001;
    
    cv::Mat Q = BuildGraph(W, 0);

    //cv::Mat M = cv::Mat(W.rows, W.cols, W.type(), double(0));
    
    int r = P.rows;
    int c = P.cols;
    double res = 0.0;
    
    for (int i = 0; i < r; i ++) {
        for (int j = 0; j < c; j ++) {
            double t1 = log10(P.at<double>(i, j) + lowest_double) ;
            double t2 = log10(Q.at<double>(i, j) + lowest_double);
            res += P.at<double>(i, j) * log10(P.at<double>(i, j) + lowest_double) - P.at<double>(i, j) * log10(Q.at<double>(i, j) + lowest_double );
        }
    }
    return res;
}

void X::LoadTrainLabel(const char* file){
    size_t nRow, nCol;
    //const char* file = "/Users/miaoever/Project/CV/MDS/MDS/mat/cls_label_feret.mat";
    MATFile *pmat = matOpen(file, "r");
    //mxArray *pa = matGetVariable(pmat, "data");
    mxArray *pa = matGetVariable(pmat, "L");

    double* val = mxGetPr(pa);
    nRow = mxGetM(pa);
    nCol = mxGetN(pa);
    //std::cout<<val[nRow * 0 + 0]<<std::endl;
    this->train_label = val;
    dim_train_label = nRow;
}

void X::LoadTestLabel(const char* file){
    MATFile *pmat = matOpen(file, "r");
    mxArray *pa_l = matGetVariable(pmat, "label_p");
    mxArray *pa_h = matGetVariable(pmat, "label_g");
    double* val_l = mxGetPr(pa_l);
    double* val_h = mxGetPr(pa_h);
    
    //std::cout<<val[nRow * 0 + 0]<<std::endl;
    this->test_l_label = val_l;
    this->test_h_label = val_h;
}

void X::LoadTrainData(const char*file){
    
    size_t nRow_l, nCol_l, nRow_h, nCol_h;
    
    MATFile *pmat = matOpen(file, "r");
    mxArray *pa_l = matGetVariable(pmat, "train_L");
    mxArray *pa_h = matGetVariable(pmat, "train_H");
    
    val_l = mxGetPr(pa_l);
    val_h = mxGetPr(pa_h);
    nRow_l = mxGetM(pa_l);
    nCol_l = mxGetN(pa_l);
    nRow_h = mxGetM(pa_h);
    nCol_h = mxGetN(pa_h);
    
    cv::Mat Low_res((int)nCol_l, (int)nRow_l, CV_64FC1, val_l);
    cv::Mat High_res((int)nCol_h, (int)nRow_h, CV_64FC1, val_h);
    
    Low_res = Low_res.t();
    High_res = High_res.t();
    Low_res.rows = (int)nRow_l;
    Low_res.cols = (int)nCol_l;
    High_res.rows = (int)nRow_h;
    High_res.cols = (int)nCol_h;
    
    //the Low_res matrix should have the same size with High_res matrix
    assert(Low_res.cols == High_res.cols);
    
    this->train_h = High_res;
    this->train_l = Low_res;
}

void X::LoadTestData(const char*file){
    size_t nRow_l, nCol_l, nRow_h, nCol_h;
    double *val_l = nullptr;
    double *val_h = nullptr;
    MATFile *pmat = matOpen(file, "r");
    mxArray *pa_l = matGetVariable(pmat, "test_L");
    mxArray *pa_h = matGetVariable(pmat, "test_H");
    
    val_l = mxGetPr(pa_l);
    val_h = mxGetPr(pa_h);
    nRow_l = mxGetM(pa_l);
    nCol_l = mxGetN(pa_l);
    nRow_h = mxGetM(pa_h);
    nCol_h = mxGetN(pa_h);
    
    cv::Mat Low_res((int)nCol_l, (int)nRow_l, CV_64FC1, val_l);
    cv::Mat High_res((int)nRow_h, (int)nCol_h, CV_64FC1, val_h);
    
    Low_res = Low_res.t();
    High_res = High_res.t();
    Low_res.rows = (int)nRow_l;
    Low_res.cols = (int)nCol_l;
    High_res.rows = (int)nRow_h;
    High_res.cols = (int)nCol_h;
    
    //the Low_res matrix should have the same size with High_res matrix
    assert(Low_res.cols == High_res.cols);
    
    this->test_h = High_res;
    this->test_l = Low_res;
}

    





