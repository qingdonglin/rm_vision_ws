#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include "armor.hpp"
#include "openvino/openvino.hpp"
#include <eigen3/Eigen/Dense>

struct ColorRange {
    int h_min, h_max;
    int s_min, s_max;
    int v_min, v_max;
};
extern ColorRange blue_range;

struct PoseResult {
    double yaw, pitch, distance;
    cv::Point3d cam_xyz;
};

class Detector
{
public:
    Detector() = default;
    ~Detector() = default;

    // ========== 传统检测接口（完整保留） ==========
    cv::Mat getColorMask(cv::Mat& img);
    cv::Mat denoise(cv::Mat& mask);
    bool isLight(const Light& light);
    std::vector<Light> findLights(const cv::Mat& rgb_img, const cv::Mat& binary_img);
    bool isArmor(const Light& l1, const Light& l2);
    bool containLight(const Light& l1, const Light& l2, const std::vector<Light>& lights);
    std::vector<Armor> matchLights(const std::vector<Light>& lights);
    
    // ========== 新增：模型检测核心接口 ==========
    bool initModel(
        const std::string& model_path,
        const std::string& classify_model_path,
        const std::string& label_path,
        const std::string& device = "AUTO",
        float conf_threshold = 0.25f,
        int top_k = 128,
        float nms_threshold = 0.3f
    );
    std::vector<Armor> detectWithModel(const cv::Mat& bgr_img);

    // ========== 绘制与位姿解算 ==========
    PoseResult solvePnPAndGetPose(cv::Point2f tl, cv::Point2f tr, cv::Point2f br, cv::Point2f bl);
    void drawResults(cv::Mat& img, std::vector<Light>& lights, std::vector<Armor>& armors, const PoseResult& input_res);
    // 模型检测结果绘制
    void drawModelResults(cv::Mat& img, std::vector<Armor>& armors, const PoseResult& input_res);

private:
    // ========== 模型检测私有成员与内部函数 ==========
    struct GridAndStride { int grid0, grid1, stride; };

    std::unique_ptr<ov::Core> ov_core_;
    std::unique_ptr<ov::CompiledModel> compiled_model_;
    std::vector<int> strides_;
    std::vector<GridAndStride> grid_strides_;

    float conf_threshold_;
    int top_k_;
    float nms_threshold_;

    static const int INPUT_W = 416;
    static const int INPUT_H = 416;
    static constexpr int NUM_CLASSES = 8;
    static constexpr int NUM_COLORS = 4;
    static constexpr float MERGE_MIN_IOU = 0.9f;
    static constexpr float MERGE_CONF_ERROR = 0.15f;

    // 数字识别
    cv::dnn::Net number_net_;
    std::vector<std::string> class_names_;

    // 内部工具函数
    cv::Mat letterbox(const cv::Mat& img, Eigen::Matrix3f& tm);
    void generateGridsAndStride();
    std::vector<Armor> decodeOutputs(const cv::Mat& output_buffer, const Eigen::Matrix3f& tm);
    std::vector<Armor> nmsMerge(std::vector<Armor>& objs);
    void extractNumberImage(const cv::Mat& src, Armor& armor);
    bool classifyNumber(Armor& armor);
};

#endif // ARMOR_DETECTOR__DETECTOR_HPP_