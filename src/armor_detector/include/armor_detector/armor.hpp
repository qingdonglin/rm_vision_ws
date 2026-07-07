#ifndef ARMOR_HPP
#define ARMOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

struct Light
{
    cv::Point2f top;
    cv::Point2f bottom;
    cv::Point2f center;
    float length;
    float width;
    float tilt_angle;

    // 新增：默认构造函数，解决Armor默认构造报错
    Light() = default;
    // 原有带参构造
    Light(cv::RotatedRect r);
};

struct Armor
{
    Light left_light;
    Light right_light;
    std::vector<cv::Point2f> pts; // 四角点，兼容模型多角点合并

    // 模型检测输出字段
    cv::Rect box;
    int color;      // 0:蓝 1:红 2:无 3:紫
    int number;     // 装甲数字ID
    float prob;     // 检测置信度
    cv::Mat number_img;
    double confidence;

    // 默认构造：初始化4个角点
    Armor() : pts(4) {}

    // 传统灯条配对构造：自动填充四角点
    Armor(Light l, Light r) : left_light(l), right_light(r), pts(4)
    {
        pts[0] = l.top;
        pts[1] = r.top;
        pts[2] = r.bottom;
        pts[3] = l.bottom;
    }
};

#endif