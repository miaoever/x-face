//
//  subject.cpp
//  LGBP
//
//  Created by miaoever on 4/28/14.
//  Copyright (c) 2014 miaoever. All rights reserved.
//

#include "Subject.hpp"
#include "util.hpp"

Subject::Subject(const string file, string userName, string method) {
    
    this->image = imread(file, -1);
    this->setUserName(userName);

    if (method == "LGBP") {
        util::getImageFeature(image, this->feature);
    }
    else if (method == "RegionalLGBP") {
        resize(image, image, Size(130, 130));
        util::FeatureDetect("", image, facialPoints);
        image = util::rotateImage(image, facialPoints, 10);
        util::getRegionLGBPFeature(image, this->feature, facialPoints, SIFT_POINTS_NUM);
    } else if (method == "RegionalLGBP2") {
        resize(image, image, Size(130, 130));
        util::FeatureDetect("", image, facialPoints);
        image = util::rotateImage(image, facialPoints, 10);
        util::getRegionLGBPFeature2(image, this->feature, facialPoints, SIFT_POINTS_NUM);
    } else if (method == "RegionalSIFT") {
        resize(image, image, Size(130, 130));
        util::FeatureDetect("", image, facialPoints);
        image = util::rotateImage(image, facialPoints, 10);
        util::getRegionSIFTFeature(image, this->feature, facialPoints, SIFT_POINTS_NUM);
    }
    
    Mat rotatedImage;
    //double angle = atan( (facialPoints[1] - facialPoints[3]) / (facialPoints[0] - facialPoints[2] ));
    //util::rotateImage(image, angle * 360 / (2 * CV_PI), rotatedImage);
    
    //util::getImageFeature(file, this->feature);
    //feature = util::getSiftFeature(image, facialPoints, SIFT_POINTS_NUM);
    //util::getImageFeature(image, this->feature);
    
}

void Subject::setID(long id) {
    this->userID = id;
}
void Subject::setUserName(string userName) {
    this->userName = userName;
}
long Subject::getUserID() {
    return this->userID;
}
string Subject::getUserName() {
    return this->userName;
}