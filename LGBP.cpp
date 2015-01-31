//
//  LGBP.cpp
//  LGBP
//
//  Created by miaoever on 4/28/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

#include "LGBP.hpp"
#include "lbp.hpp"
#include "Gabor.hpp"
#include "histogram.hpp"

void lgbp::getLGBP(const Mat& src, int kernelSize, int radius, int neighbor, bool uniform, Mat& hist) {
    vector<Mat> GMP;
    int numPatterns = uniform? 58 : static_cast<int>(std::pow(2.0, static_cast<double>(8)));
    Mat tmp, tmpHist;
    int rowBlockNum = 5;
    int colBlockNum = 5;
    
    GaborFR::gaborFilter(src, kernelSize, GMP);

    lbp::ELBP(GMP[0], tmp, radius, neighbor, uniform);
    lbp::spatial_histogram(tmp, hist, numPatterns, rowBlockNum, colBlockNum);
    //lbp::histogram(tmp, hist, numPatterns);
    size_t len = GMP.size();
    for (int i = 1; i < len; i++) {
        tmp.release();
        tmpHist.release();
        lbp::ELBP(GMP[i], tmp, radius, neighbor, uniform);
        lbp::spatial_histogram(tmp, tmpHist, numPatterns, rowBlockNum, colBlockNum);
        //lbp::histogram(tmp, tmpHist, numPatterns);
        hconcat(tmpHist, hist, hist);
    }
}

double lgbp::chi_square(const Mat& histogram0, const Mat& histogram1) {
    return lbp::chi_square(histogram0, histogram1);
}
