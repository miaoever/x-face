//
//  subject.h
//  LGBP
//
//  Created by miaoever on 4/28/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

#ifndef __LGBP__subject__
#define __LGBP__subject__

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


class Subject {
public:
    Subject(const string file, string userName, string method);
    void setID(long id);
    void setUserName(string userName);
    long getUserID();
    string getUserName();
    Mat feature;
    Mat image;
private:
    long id;
    long userID;
    string userName;
    float facialPoints[10];
};
#endif /* defined(__LGBP__subject__) */
