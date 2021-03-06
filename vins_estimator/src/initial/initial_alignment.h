#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
public:
    ImageFrame(){};
    ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &_points, double _t) : t{_t}, is_key_frame{false}
    {
        points = _points;
    };
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> points; //当前帧的特征点(id号+状态和位置信息(x, y, z, p_u, p_v, velocity_x, velocity_y))
    double t;                                                        //时间戳信息
    Matrix3d R;                                                      //姿态信息
    Vector3d T;                                                      //位置信息
    IntegrationBase *pre_integration;                                //上一帧到当前帧时间间隔内的imu信息集合
    bool is_key_frame;
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);