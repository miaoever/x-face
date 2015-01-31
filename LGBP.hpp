//
//  LGBP.h
//  LGBP
//
//  Created by miaoever on 4/28/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

//#ifndef __LGBP__LGBP__
//#define __LGBP__LGBP__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
using namespace cv;

namespace lgbp {
    void getLGBP(const Mat& src, int kernelSize, int radius, int neighbor, bool uniform, Mat& hist);
    double chi_square(const Mat& histogram0, const Mat& histogram1);
}
//#endif /* defined(__LGBP__LGBP__) */
