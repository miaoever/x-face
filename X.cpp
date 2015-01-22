//
//  X.cpp
//  MDS
//
//  Created by miaoever on 1/19/15.
//  Copyright (c) 2015 miaoever. All rights reserved.
//

#include "X.h"
#include <iostream>

typedef struct Pos {
    double value;
    int idx;
} Pos;

bool cmp (Pos a, Pos b) {
    return (a.value < b.value );
}

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
    P = BuildGraph(W); ///TODO - for P, the adj should be one representing the most similarity.
    cv::Mat _W, diff_W;
    
    for (int i = 0; i < Max_iteration; i ++) {
        diff_W = Calc_diff(_W);
        _W = _W - step * diff_W;
        
        _W = PSDProjection(_W);
        double _Value = Calc_value_obj(_W);
        
#ifdef __DEGUB__
        printf("term: %d     value: %f\n", i, _Value);
#endif
    }
}

cv::Mat X::BuildGraph(const cv::Mat& W) {
    size_t len = NonZeroAdj.size();
    cv::Mat weight = cv::Mat(theta_l.cols, theta_l.cols, CV_64F, double(0));

    for (int i = 0; i < len; i++) {
        cv::Mat tmp = (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y)).t() * W * (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y));
        weight.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y) = cv::exp(- tmp.at<double>(0) / (2 * gamma2));
    }
    
    Normalization(weight);
    return weight;
}

void X::Normalization(cv::Mat& weight) {
    size_t r = weight.rows;
    size_t c = weight.cols;
    //cv::Mat sum = cv::Mat(weight.rows, weight.cols, weight.type());
    cv::Mat sum;
    cv::Mat _w = weight.clone();
    
    //calc the sum of each row for weight.
    cv::reduce(weight.t(), sum, 0, CV_REDUCE_SUM);
    
    for (int i = 0; i < c; i ++) {
        for (int j = 0; j < r; j ++) {
            weight.at<double>(i,j) = _w.at<double>(i, j) / (sum.col(i).at<double>(0) - _w.at<double>(i, i));
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
                dist[i][j] = cv::norm(train_h.col(i) - train_h.col(j));
                dist[j][i] = dist[i][j];
            }
        }
    }
    
    for (int i = 0; i < num; i ++) {
       
        std::vector<Pos> row;
        for (int j = 0; j < num; j ++) {
            row[j].value = dist[i][j];
            row[j].idx = j;
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
    
    cv::Mat Q = BuildGraph(W);
    cv::Mat M = cv::Mat(W.rows, W.cols, W.type(), double(0));
    
    size_t len = NonZeroAdj.size();
    for (int i = 0; i < len; i++) {
        M = M + lambda * (P.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y) - Q.at<double>(NonZeroAdj[i].x, NonZeroAdj[i].y)) * (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y)) * (theta_l.col(NonZeroAdj[i].x) - theta_h.col(NonZeroAdj[i].y)).t();
    }
    
    cv::Mat diff_W = (1 / (2 * gamma2)) * M;
    return diff_W;
}

cv::Mat X::PSDProjection(const cv::Mat& W) {
    cv::Mat eigenVector, eigenValue;
    cv::Mat S;
    
    if (cv::eigen(W, eigenValue, eigenVector)) {
        int len = eigenVector.rows;
        
        for (int i = 0; i < len; i++) {
            S = S + cv::max(eigenValue.col(i), 0) * eigenVector.row(i).t() * eigenVector.row(i);
        }
    }

    return S;
}

double X::Calc_value_obj(const cv::Mat& W) {
    
    cv::Mat Q = BuildGraph(W);

    cv::Mat M = cv::Mat(W.rows, W.cols, W.type(), double(0));
    
    int r = P.rows;
    int c = P.cols;
    double res = 0.0;
    
    for (int i = 0; i < r; i ++) {
        for (int j = 0; j < c; j ++) {
            res += P.at<double>(i, j) * log10(P.at<double>(i, j)) - P.at<double>(i, j)* log10(Q.at<double>(i, j));
        }
    }
    
    return res;
}



