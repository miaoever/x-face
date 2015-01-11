//
//  MDS.cpp
//  MDS
//
//  Created by miaoever on 3/3/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//
#include "MDS.h"

MDS::MDS():lambda(0.5),dim_unified(30),rank(10){
    std::cout<<"Loading data ..."<<std::endl;
    LoadTrainLabel("/Users/miaoever/Project/CV/MDS/MDS/mat/cls_label_feret.mat");
    LoadTrainData("/Users/miaoever/Project/CV/MDS/MDS/mat/MDS_train.mat");
}
void MDS::RANK(int x){
    this->rank = x;
    std::cout<<this->rank<<std::endl;
}

void MDS::LoadTrainLabel(const char* file){
    size_t nRow, nCol;
    //const char* file = "/Users/miaoever/Project/CV/MDS/MDS/mat/cls_label_feret.mat";
    MATFile *pmat = matOpen(file, "r");
    mxArray *pa = matGetVariable(pmat, "data");
    double* val = mxGetPr(pa);
    nRow = mxGetM(pa);
    nCol = mxGetN(pa);
    //std::cout<<val[nRow * 0 + 0]<<std::endl;
    this->train_label = val;
    dim_train_label = nRow;
}

void MDS::LoadTestLabel(const char* file){
    MATFile *pmat = matOpen(file, "r");
    mxArray *pa_l = matGetVariable(pmat, "label_p");
    mxArray *pa_h = matGetVariable(pmat, "label_g");
    double* val_l = mxGetPr(pa_l);
    double* val_h = mxGetPr(pa_h);
    
    //std::cout<<val[nRow * 0 + 0]<<std::endl;
    this->test_l_label = val_l;
    this->test_h_label = val_h;
}

void MDS::LoadTrainData(const char*file){
    
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

void MDS::LoadTestData(const char*file){
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

double MDS::GetValue(double* array, size_t nRow, int idx_i, int idx_j){
    return array[nRow * idx_j + idx_i];
}

void MDS::LoadPreproceData(cv::Mat& A, double& B){
    std::string filename = "preproceccing.xml";
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["A"] >> A;
    fs["B"] >> B;
    fs.release();
}

void MDS::LoadProjectionMatrix(const char* file){
    std::string filename = file;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["W"] >> this->W;;
    fs.release();
}

void MDS::SavePreproceData(cv::Mat& A, double& B){
    std::string filename = "preproceccing.xml";
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "A" << A; // Write entire cv::Mat
    fs<<"B"<<B;
    fs.release();
}

void MDS::SaveProjectionMatrix(){
    std::string filename = "W.xml";
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    
    fs << "W" << W; // Write entire cv::Mat
    //fs["R"] >> R;   // Read entire cv::Mat
    fs.release();
}

void MDS::Train(int iter = 10,bool pre = false){
    //std::cout<<Low_res.col(0)<<std::endl;
    /*
    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < 50; j++) {
            std::cout<<GetValue(train_label, dim_train_label, i,j)<<" ";
        }
        std::cout<<std::endl;
    }
    */
    int type = train_l.type();
    int num_vector = train_l.cols;
    int dim_l = train_l.rows;
    int dim_h = train_h.rows;
    
    cv::Mat W, V, theta_l, theta_h, C;
    cv::Mat w1(dim_unified, dim_l, type);
    cv::Mat w2(dim_unified, dim_h, type);
    
    cv::randu(w1, cv::Scalar::all(-1), cv::Scalar::all(1));
    cv::randu(w2, cv::Scalar::all(-1), cv::Scalar::all(1));
    cv::vconcat(w1.t(), w2.t(), W);
    W = cv::Mat::ones(W.rows, W.cols, type);
    cv::vconcat(train_l, cv::Mat::zeros(dim_h, num_vector, type), theta_l);
    cv::vconcat(cv::Mat::zeros(dim_l, num_vector, type), train_h, theta_h);

    cv::Mat dist_q = cv::Mat::zeros(theta_l.rows, theta_l.rows, type);
    cv::Mat A = cv::Mat::zeros(theta_l.rows, theta_l.rows, type);
    double B = 0, a = 0, q = 0, c = 0;
    
    //std::cout<<theta_h.col(1)<<std::endl;
    
    //std::cout<<W<<std::endl;
    std::cout<<">>Preproceccing."<<std::endl;
    if (pre == false)
    {
        for (int i = 0; i < num_vector; i++){
            for (int j = 0; j < num_vector; j++){
                if (static_cast<int>(GetValue(train_label, dim_train_label, i, j)) == 1) {
                    a = (1 - lambda) + lambda;
                } else {
                    a = lambda;
                }
                //std::cout<<a<<std::endl;
                dist_q = (theta_l.col(i) - theta_h.col(j)) * (theta_l.col(i) - theta_h.col(j)).t();
                A += a * dist_q;
                double distance = cv::norm(train_h.col(i) - train_h.col(j));
                B += a * distance * distance;
            }
        }
    } else {
        LoadPreproceData(A, B);
    }
    cv::Mat _A;
    cv::invert(A, _A);
    std::cout<<">>Begin to iterate. "<<std::endl;
    std::ofstream file;
    file.open("/Users/miaoever/Documents/MATLAB/A.txt");
    file<<"[";
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            file<<A.at<double>(i,j);
            if (j != A.cols - 1)
                file <<", ";
            else
                file<<";";
        }
    }
    file<<"];";
    file.close();
    for (int k = 0; k < iter; k++){
        //cv::Mat V(W);
        W.copyTo(V);
        C = cv::Mat::zeros(A.rows, A.cols, type);
        for (int i = 0; i < num_vector; i++){
            for (int j = 0; j < num_vector; j++){
                dist_q = (theta_l.col(i) - theta_h.col(j)) * (theta_l.col(i) - theta_h.col(j)).t();
                q = cv::norm(V.t() * (theta_l.col(i) - theta_h.col(j)));
                if (q > 0)
                    c = lambda * cv::norm(train_h.col(i) - train_h.col(j)) / q;
                else
                    c = 0;
                C += c * dist_q;
            }
        }
        /*
         cv::Mat _A;
         cv::invert(A, _A);
         */
        //std::cout<<C.col(0)<<std::endl;
        W = _A * C * V;
        cv::Mat tmp = _A * C;
        
        std::cout<<tmp.col(1)<<std::endl<<"########"<<std::endl;
        
        std::cout<<_A.col(1)<<std::endl<<"########"<<std::endl;
        std::cout<<C.col(1)<<std::endl<<"########"<<std::endl;
        std::cout<<V.col(1)<<std::endl<<"########"<<std::endl;
        std::cout<<W.col(1)<<std::endl<<"########"<<std::endl;
        cv::Scalar g1 = cv::trace(W.t() * A * W);
        cv::Scalar g2 = cv::trace(V.t() * C * W);
        double g = g1[0] - 2 * g2[0] + B;
        
        std::cout<<"#"<<k + 1<<"    "<<g / (num_vector * num_vector) << std::endl;
        
    }
    this->W = W;
}

bool cmp(struct dist a, struct dist b) {
    return (a.distance < b.distance);
}

void MDS::Test(int rank, bool Load_Projection_Matrix){
    std::cout<<"Loading test data ..."<<std::endl;
    LoadTestLabel("/Users/miaoever/Project/CV/MDS/MDS/mat/test_label.mat");
    LoadTestData("/Users/miaoever/Project/CV/MDS/MDS/mat/MDS_test.mat");
    if (Load_Projection_Matrix){
        LoadProjectionMatrix("/Users/miaoever/Project/CV/MDS/MDS/mat/W.xml");
    }
    std::cout<<"Begin to testing ..."<<std::endl;
    //using namespace CMP;
    cv::Mat theta_l, theta_h;
    int num_vector = test_h.cols;
    int dim_l = test_l.rows;
    int dim_h = test_h.rows;
    int type = test_l.type();
    int cc = 0;
    int count = 0;
    
    std::cout<<this->rank<<std::endl;
    std::cout<<this->rate[0]<<std::endl;
    std::cout<<count<<std::endl;
    
    std::cout<<std::endl<<std::endl;
    
    //std::cout<<this->rank<<std::endl;
    this->rank = rank;
    for (int i = 0; i < num_vector; i++)
        std::cout<<test_l_label[i]<<std::endl;
    cv::Mat W_t = W.t();
    /*
    for (int i = 0; i < num_vector; i++) {
        //struct dist* res = new struct dist[num_vector];
        std::vector<struct dist> res(num_vector);

        cc = 0;
        //std::cout<<test_l.col(0)<<std::endl;
        cv::vconcat(test_l.col(i), cv::Mat::zeros(dim_h, 1, type), theta_l);
        for (int j = 0; j < num_vector; j++) {
            cv::vconcat(cv::Mat::zeros(dim_l, 1, type), test_h.col(j), theta_h);
            res[cc].distance= cv::norm(W_t * (theta_l - theta_h));
            res[cc++].idx = j;
        }
        std::sort(res.begin(), res.end(), cmp);
        for (int r = 0; r < rank; r++){
            for (int j = 0; j <= r; j++){
                if (test_l_label[i] == test_h_label[res[j].idx]){
                    //std::cout<<test_l_label[res[l].idx] << " "<<test_h_label[i]<<std::endl;
                    rate[r]++;
                    break;
                }
            }
        }
    }
    for (int i = 0; i < rank; i++) {
        //rate[i] /= num_vector;
        std::cout<<rate[i]<<" ";
    }*/
    
    double dist;
    for (int i = 0; i < num_vector; i++) {
        double min_dist = 100000;
        int min_index = -1;
        cv::vconcat(test_l.col(i), cv::Mat::zeros(dim_h, 1, type), theta_l);
        for (int j = 0; j < num_vector; j++){
            cv::vconcat(cv::Mat::zeros(dim_l, 1, type), test_h.col(j), theta_h);
            dist = cv::norm(W_t * (theta_l - theta_h));
            if (dist < min_dist){
                min_dist = dist;
                min_index = j;
            }

        }
        if (test_l_label[i] == test_h_label[min_index]){
            cc ++;
        }
    }
    std::cout<<cc<<std::endl;
    std::cout<<std::endl;
}