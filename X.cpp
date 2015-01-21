//
//  X.cpp
//  MDS
//
//  Created by miaoever on 1/19/15.
//  Copyright (c) 2015 miaoever. All rights reserved.
//

#include "X.h"
#include <iostream>
#include <math.h>


void X::Train() {
    // 4 - the scale of neighborhood
    //cv::Mat adj = Knn(4);
    
    adj = Knn(4);

    int type = train_l.type();
    int num_vector = train_l.cols;
    int dim_l = train_l.rows;
    int dim_h = train_h.rows;

  
    cv::Mat w1(dim_unified, dim_l, type);
    cv::Mat w2(dim_unified, dim_h, type);
    
    cv::randu(w1, cv::Scalar::all(-1), cv::Scalar::all(1));
    cv::randu(w2, cv::Scalar::all(-1), cv::Scalar::all(1));
    cv::vconcat(w1.t(), w2.t(), W);
    W = cv::Mat::ones(W.rows, W.cols, type);
    
    cv::vconcat(train_l, cv::Mat::zeros(dim_h, num_vector, type), theta_l);
    cv::vconcat(cv::Mat::zeros(dim_l, num_vector, type), train_h, theta_h);
    
    
}

void X::Gradient_descent() {
    cv::Mat P = BuildGraph();
    cv::Mat _W, diff_W;
    
    for (int i = 0; i < Max_iteration; i ++) {
        diff_W = Calc_diff(_W);
        _W = _W - step * diff_W;
        
        PSDProjection(_W);
        double _Value = Calc_value_obj(_W);
        
#ifdef __DEGUB__
        printf("term: %d     value: %f\n", i, _Value);
#endif
    }
}

cv::Mat X::BuildGraph() {
    size_t len = NonZeroAdj.size();
    cv::Mat weight = cv::Mat(theta_l.cols, theta_l.cols, CV_64F, double(0));

    for (int i = 0; i < len; i++) {
        cv::Mat tmp = (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y)).t() * W * (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y));
        weight.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y) = exp(- tmp.at<double>(0) / (2 * gamma2));
    }
    
    Normalization(weight);
    return weight;
}

void X::Normalization(cv::Mat& weight) {
    size_t r = weight.rows;
    size_t c = weight.cols;
    cv::Mat sum = cv::Mat(weight.rows, weight.cols, weight.type());

    cv::reduce(weight, sum, 0, CV_REDUCE_SUM);
    
    for (int i = 0; i < c; i ++) {
        for (int j = 0; j < r; j ++) {
            //weight.at<double>(i,j) /=
        }
    }
    
}
