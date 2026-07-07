#include "armor_detector/detector.hpp"
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// 模型参数，和训练时保持一致
#define INPUT_W 416
#define INPUT_H 416
#define NUM_COLORS 2
#define NUM_CLASSES 8
#define MERGE_MIN_IOU 0.8f
#define MERGE_CONF_ERROR 0.1f
using namespace cv;
using namespace std;

ColorRange blue_range = {66, 110, 55, 255, 234, 255};
// 二值化亮度阈值：识别不到就调小，误检多就调大
const int binary_thresh = 60;

#define ARMOR_WIDTH    0.135
#define ARMOR_HEIGHT   0.055
// 单根灯条过滤阈值 isLight使用
const float min_ratio = 0.03f;       // 灯条长宽比最小值
const float max_ratio = 0.35f;        // 灯条长宽比最大值
const float max_angle = 35.0f;       // 单根灯条允许最大倾斜角度
// 两根灯条配对阈值 isArmor使用
const float min_light_ratio = 0.15f;  // 两根灯条长度最小比例
const float min_small_center_distance = 0.3f;  // 两灯中心最小间距比例
const float max_small_center_distance = 4.5f;  // 两灯中心最大间距比例
const float max_armor_angle = 40.0f; // 两灯中心点连线和水平线x轴的最大水平倾斜角

Light::Light(RotatedRect r)
{
    // 1. 区分长边（灯条长度）和短边（灯条宽度）
    float long_side = max(r.size.width, r.size.height);
    float short_side = min(r.size.width, r.size.height);
    length = long_side;
    width = short_side;

    // 2. 计算长边与x轴的夹角（修正width/height交换带来的角度偏移）
    float angle_long = r.angle;
    if (r.size.width < r.size.height) {
        angle_long += 90.0f;
    }

    // 3. 计算灯条与竖直方向的倾斜角，范围0~45°
    tilt_angle = abs(angle_long - 90.0f);
    if (tilt_angle > 45) tilt_angle = 90 - tilt_angle;

    // 4. 沿长边方向计算上下端点坐标
    float rad = angle_long * CV_PI / 180.0f;
    Point2f dir(cos(rad), sin(rad));
    top = r.center + dir * (long_side / 2.0f);
    bottom = r.center - dir * (long_side / 2.0f);
    center = r.center;

    // 保证top点y坐标小于bottom点
    if (top.y > bottom.y) {
        swap(top, bottom);
    }
}

Mat Detector::getColorMask(Mat& img)//掩码生成 getColorMask：筛选蓝色区域
{
    Mat hsv, mask;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv,
            Scalar(blue_range.h_min, blue_range.s_min, blue_range.v_min),
            Scalar(blue_range.h_max, blue_range.s_max, blue_range.v_max),
            mask);
    return mask;
}

// ========== 形态学核加宽，适配倾斜灯条，避免断成小轮廓 ==========
Mat Detector::denoise(Mat& mask)
{
    // 第1步：横向腐蚀，优先抹除水平长条灯管（最关键的去噪）
    Mat kernelH = getStructuringElement(MORPH_RECT, Size(2, 1));
    erode(mask, mask, kernelH);

    // 第2步：小核开运算，去除点状毛刺噪点
    Mat kernelOpen = getStructuringElement(MORPH_RECT, Size(2, 3));
    morphologyEx(mask, mask, MORPH_OPEN, kernelOpen);

    // 第3步：垂直膨胀，拉长真实竖直装甲灯条
    Mat kernelV = getStructuringElement(MORPH_RECT, Size(3, 9));
    dilate(mask, mask, kernelV);

    // 第4步：闭运算，填充灯条内部小黑洞
    Mat kernelClose = getStructuringElement(MORPH_RECT, Size(3, 7));
    morphologyEx(mask, mask, MORPH_CLOSE, kernelClose);

    // 第5步：面积双阈值过滤
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat cleanMask = Mat::zeros(mask.size(), CV_8UC1);
    for (auto& cnt : contours)
    {
        double area = contourArea(cnt);
        // 下限去小点，上限去大片灯光
        if (area > 40)
            drawContours(cleanMask, vector<vector<Point>>{cnt}, 0, Scalar(255), -1);
    }
    return cleanMask;
}

bool Detector::isLight(const Light& light)//单灯条过滤 isLight：几何形状校验
{
    float ratio = light.width / light.length;
    bool ratio_ok = min_ratio < ratio && ratio < max_ratio;//宽长比在 0.03~0.7 之间：必须细长，过滤方块、圆形杂物
    bool angle_ok = light.tilt_angle < max_angle;//灯条倾斜小于 60°：接近水平的长条直接剔除，只保留竖直 LED
    return ratio_ok && angle_ok;
}

//生成最终合法灯条数组（多层过滤核心）
//把完整 BGR 原图rgb_img传进函数，能读取每个像素蓝、绿、红三通道亮度值。
vector<Light> Detector::findLights(const Mat& rgb_img, const Mat& binary_img)
{
    vector<vector<Point>> contours;
    findContours(binary_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<Light> lights;

    // 遍历所有掩码轮廓
    for (const auto& contour : contours)
    {
        if (contour.size() < 5) continue;
        
        if (contourArea(contour) < 5) continue;

        RotatedRect r_rect = minAreaRect(contour);
        Light light(r_rect);
        // 第一层：几何形状过滤
        if (!isLight(light)) continue;

        // 边界保护：防止轮廓贴图片边缘，ROI越界崩溃
//对每一个轮廓，裁剪出这个轮廓在原图上对应的一小块区域，不用遍历整张图，只处理当前灯条周边像素，节省计算。
        Rect rect = r_rect.boundingRect();
        if (rect.x < 0 || rect.y < 0 ||
            rect.x + rect.width > rgb_img.cols ||
            rect.y + rect.height > rgb_img.rows)
            continue;

        // 第二层：ROI像素蓝红比值校验，杜绝淡蓝色污渍假灯

        int sum_b = 0, sum_r = 0;
        Mat roi = rgb_img(rect);
        for (int i = 0; i < roi.rows; i++)
        {
            for (int j = 0; j < roi.cols; j++)
            {
                // 只统计轮廓内部像素，排除背景
                if (pointPolygonTest(contour, Point2f(j + rect.x, i + rect.y), false) >= 0)
                {
                    sum_b += roi.at<Vec3b>(i, j)[0];
                    sum_r += roi.at<Vec3b>(i, j)[2];
                }
            }
        }
        // 蓝色通道总和大于红色?倍才放行，只有LED蓝光满足
        if (sum_b > sum_r * 1)
        {
            lights.emplace_back(light);
        }
    }
    return lights;
}
//判断两根灯条能否组成装甲
bool Detector::isArmor(const Light& l1, const Light& l2)
{
    // 0. 新增：两根灯条倾斜角必须接近，差值过大直接排除
    float angle_diff = fabs(l1.tilt_angle - l2.tilt_angle);
    if (angle_diff > 15.0f) return false;  // 倾斜差超过15°判定不是同一块装甲
    // 1. 两灯长度相似度校验
    float len_ratio = min(l1.length, l2.length) / max(l1.length, l2.length);
    if (len_ratio < min_light_ratio) return false;

    // 2. 两灯中心间距校验（相对灯平均长度归一化，远近通用）
    float avg_len = (l1.length + l2.length) / 2;
    float dist = norm(l1.center - l2.center) / avg_len;
    if (dist < min_small_center_distance || dist > max_small_center_distance) return false;

    // 3. 两灯连线水平倾斜角校验
    float delta_x = abs(l1.center.x - l2.center.x);
    float delta_y = abs(l1.center.y - l2.center.y);
    float angle = atan2(delta_y, delta_x) * 180 / CV_PI;

    // ===== 调试打印，放在angle计算完、判断阈值之前 =====
    cout << "len_ratio: " << len_ratio 
         << " | dist: " << dist 
         << " | angle: " << angle << endl;

    if (angle > max_armor_angle) return false;

    return true;
}
//嵌套灯条过滤 containLight：避免多灯条拼接假装甲
//输入两根待配对灯条 l1、l2 + 全场所有灯条数组 lights，如果这两根灯的包围框中间还夹着第三根别的灯，就返回 true，禁止它们拼成装甲
bool Detector::containLight(const Light& l1, const Light& l2, const vector<Light>& lights)
{
    vector<Point2f> pts = {l1.top, l1.bottom, l2.top, l2.bottom};
    Rect rect = boundingRect(pts);//框住 l1 和 l2 两根灯的整体包围矩形

    for (const auto& light : lights)//遍历画面里所有识别到的灯条
    {
        //跳过自己，只看别的灯
        if (light.center == l1.center || light.center == l2.center) continue;
        //第三方灯是否落在包围框内
        if (rect.contains(light.center))
            return true;
    }
    return false;
}

vector<Armor> Detector::matchLights(const vector<Light>& lights)
{
    vector<Armor> armors;
    for (int i = 0; i < lights.size(); i++)
        for (int j = i + 1; j < lights.size(); j++)
        {
            const Light& l1 = lights[i];
            const Light& l2 = lights[j];
            if (containLight(l1, l2, lights)) continue;
            if (isArmor(l1, l2))
            {
                if (l1.center.x < l2.center.x)
                    armors.emplace_back(l1, l2);
                else
                    armors.emplace_back(l2, l1);
            }
        }
    return armors;
}

vector<Point3f> getRealWorldPoints()
{
    vector<Point3f> objPoints;
    float half_w = ARMOR_WIDTH / 2;
    float half_h = ARMOR_HEIGHT / 2;
    objPoints.push_back(Point3f(-half_w, -half_h, 0));
    objPoints.push_back(Point3f(half_w, -half_h, 0));
    objPoints.push_back(Point3f(half_w, half_h, 0));
    objPoints.push_back(Point3f(-half_w, half_h, 0));
    return objPoints;
}

Mat getCameraMatrix()
{
    return (Mat_<double>(3, 3) << 650, 0, 320, 0, 650, 240, 0, 0, 1);
}

Mat getDistCoeffs()
{
    return Mat::zeros(5, 1, CV_64F);
}

PoseResult Detector::solvePnPAndGetPose(Point2f tl, Point2f tr, Point2f br, Point2f bl)
{
    vector<Point2f> imgPoints;
    imgPoints.push_back(tl);
    imgPoints.push_back(tr);
    imgPoints.push_back(br);
    imgPoints.push_back(bl);

    Mat rvec, tvec;
    solvePnP(getRealWorldPoints(), imgPoints, getCameraMatrix(), getDistCoeffs(), rvec, tvec, false, SOLVEPNP_IPPE);

    double tx = tvec.at<double>(0);
    double ty = tvec.at<double>(1);
    double tz = tvec.at<double>(2);
    double dist = sqrt(tx*tx + ty*ty + tz*tz);

    Mat R;
    Rodrigues(rvec, R);
    double yaw_deg = atan2(R.at<double>(1,0), R.at<double>(0,0)) * 180 / CV_PI;
    double pitch_deg = atan2(-R.at<double>(2,0), sqrt(R.at<double>(2,1)*R.at<double>(2,1) + R.at<double>(2,2)*R.at<double>(2,2))) * 180 / CV_PI;

    PoseResult res;
    res.yaw = yaw_deg;
    res.pitch = pitch_deg;
    res.distance = dist;
    res.cam_xyz = cv::Point3d(tx, ty, tz);
    return res;
}

void Detector::drawResults(Mat& img, vector<Light>& lights, vector<Armor>& armors, const PoseResult& input_res)
{
    for (auto& light : lights)
    {
        circle(img, light.top, 3, Scalar(255,255,255), 1);
        circle(img, light.bottom, 3, Scalar(255,255,255), 1);
        line(img, light.top, light.bottom, Scalar(255,0,255), 1);
    }

    for (auto& armor : armors)
    {
        line(img, armor.pts[0], armor.pts[1], Scalar(0,255,0), 2);
        line(img, armor.pts[1], armor.pts[2], Scalar(0,255,0), 2);
        line(img, armor.pts[2], armor.pts[3], Scalar(0,255,0), 2);
        line(img, armor.pts[3], armor.pts[0], Scalar(0,255,0), 2);
        line(img, armor.pts[0], armor.pts[2], Scalar(0,255,0), 1);
        line(img, armor.pts[1], armor.pts[3], Scalar(0,255,0), 1);

        putText(img, "yaw: " + to_string(input_res.yaw).substr(0,5) + " deg",
                Point(armor.pts[0].x, armor.pts[0].y-50),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);

        putText(img, "pitch: " + to_string(input_res.pitch).substr(0,5) + " deg",
                Point(armor.pts[0].x, armor.pts[0].y-35),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);

        putText(img, "dis: " + to_string(input_res.distance).substr(0,4) + "m",
                Point(armor.pts[0].x, armor.pts[0].y-20),
        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,255), 1);
    }

    imshow("Armor Detector", img);
    waitKey(1);
}



//  OpenVINO 模型检测实现
// 计算两个凸四边形的交并比IOU
static float polygonIou(const std::vector<cv::Point2f>& poly1, const std::vector<cv::Point2f>& poly2)
{
    std::vector<cv::Point2f> inter_poly;
    // 无交集直接返回0
    if (!cv::intersectConvexConvex(poly1, poly2, inter_poly)) {
        return 0.0f;
    }
    float inter_area = static_cast<float>(cv::contourArea(inter_poly));
    float area1 = static_cast<float>(cv::contourArea(poly1));
    float area2 = static_cast<float>(cv::contourArea(poly2));
    float union_area = area1 + area2 - inter_area;
    return union_area > 1e-6f ? inter_area / union_area : 0.0f;
}


// 角点标准化排序：输入任意顺序的4个点，输出 左上、右上、右下、左下 的标准顺序
//OpenCV 图像坐标系规则：图像左上角是坐标原点 (0,0)
//横轴 x：向右数值变大纵轴 y：向下数值变大
static std::vector<cv::Point2f> sortArmorCorners(const std::vector<cv::Point2f>& pts)
{
    std::vector<cv::Point2f> sorted(4);
    // 1. x+y 最小=左上，最大=右下
    int tl_idx = 0, br_idx = 0;
    float min_sum = pts[0].x + pts[0].y;
    float max_sum = pts[0].x + pts[0].y;
    for (int i = 1; i < 4; i++) {
        float s = pts[i].x + pts[i].y;
        if (s < min_sum) { min_sum = s; tl_idx = i; }
        if (s > max_sum) { max_sum = s; br_idx = i; }
    }
    sorted[0] = pts[tl_idx]; // 左上
    sorted[2] = pts[br_idx]; // 右下

    // 2. x-y 最大=右上，最小=左下
    int tr_idx = -1, bl_idx = -1;
    for (int i = 0; i < 4; i++) {
        if (i == tl_idx || i == br_idx) continue;
        float d = pts[i].x - pts[i].y;
        if (tr_idx == -1) {
            tr_idx = i;
            bl_idx = i;
        } else {
            if (d > (pts[tr_idx].x - pts[tr_idx].y)) tr_idx = i;
            if (d < (pts[bl_idx].x - pts[bl_idx].y)) bl_idx = i;
        }
    }
    sorted[1] = pts[tr_idx]; // 右上
    sorted[3] = pts[bl_idx]; // 左下

    return sorted;
}
//模型初始化
bool Detector::initModel(
    const std::string& model_path,
    const std::string& classify_model_path,
    const std::string& label_path,
    const std::string& device,
    float conf_threshold,
    int top_k,
    float nms_threshold)
{
    try {
        conf_threshold_ = conf_threshold;
        top_k_ = top_k;
        nms_threshold_ = nms_threshold;

        // 初始化OpenVINO核心
        ov_core_ = std::make_unique<ov::Core>();
        //读取.xml/.bin或.onnx模型文件，解析网络结构和权重。
        auto model = ov_core_->read_model(model_path);

        //配置输入输出预处理（必须调用build()才生效）
        //配置输入输出数据类型为 float32
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::f32);
        ppp.output().tensor().set_element_type(ov::element::f32);
        model = ppp.build();

        // 编译模型，把通用模型编译成当前硬件能直接执行的机器码，是 OpenVINO 加速的核心步骤。
        //LATENCY优先低延迟模式，优先保证单帧推理速度最快
        compiled_model_ = std::make_unique<ov::CompiledModel>(
            ov_core_->compile_model(model, device,
                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))
        );

        // 加载数字识别模型
        number_net_ = cv::dnn::readNetFromONNX(classify_model_path);
        if (number_net_.empty()) {
            std::cerr << "[ERROR] 数字识别模型加载失败: " << classify_model_path << std::endl;
            return false;
        }

        // 加载标签文件
        std::ifstream label_file(label_path);
        std::string line;
        class_names_.clear();
        while (std::getline(label_file, line)) {
            class_names_.push_back(line);
        }
        if (class_names_.empty()) {
            std::cerr << "[ERROR] 标签文件加载失败: " << label_path << std::endl;
            return false;
        }

        // 生成网格与步长
        //把 416×416 的输入图切成不同大小的格子，每个格子负责预测自己这块区域里有没有装甲。
        strides_ = {8, 16, 32};
        //步长 8：416/8 = 52×52 个网格，负责检测近处大装甲
        //步长 16：26×26 个网格，负责检测中距离装甲...
        generateGridsAndStride();
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] 模型初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void Detector::generateGridsAndStride()
{
    grid_strides_.clear();
    for (int s : strides_) {
        int num_w = INPUT_W / s;
        int num_h = INPUT_H / s;
        for (int g1 = 0; g1 < num_h; g1++)
            for (int g0 = 0; g0 < num_w; g0++)
                grid_strides_.push_back({g0, g1, s});
    }
}


//letterbox图像预处理
//letterbox等比例缩放，因为模型输入必须是固定尺寸 416×416
cv::Mat Detector::letterbox(const cv::Mat& img, Eigen::Matrix3f& tm)
{
    int ih = img.rows, iw = img.cols;
    float scale = std::min(INPUT_H * 1.0f / ih, INPUT_W * 1.0f / iw);
    int rh = static_cast<int>(round(ih * scale));
    int rw = static_cast<int>(round(iw * scale));
    int ph = INPUT_H - rh, pw = INPUT_W - rw;

    cv::Mat rim;
    cv::resize(img, rim, cv::Size(rw, rh));

    float hh = ph / 2.0f, hw = pw / 2.0f;

    // 还原变换矩阵
    tm << 1.0f / scale, 0, -hw / scale,
          0, 1.0f / scale, -hh / scale,
          0, 0, 1;

    int top = static_cast<int>(round(hh - 0.1f));
    int bot = static_cast<int>(round(hh + 0.1f));
    int lef = static_cast<int>(round(hw - 0.1f));
    int rig = static_cast<int>(round(hw + 0.1f));
//上下 / 左右用灰色 (114,114,114) 填充，补成正好 416×416。
    cv::copyMakeBorder(rim, rim, top, bot, lef, rig,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return rim;
}

//输出解码
std::vector<Armor> Detector::decodeOutputs(
    const cv::Mat& ob, const Eigen::Matrix3f& tm)
{
    std::vector<Armor> result;
    for (size_t ai = 0; ai < grid_strides_.size(); ai++) {
        
        float conf_raw = ob.at<float>(ai, 8);
        float conf = conf_raw;
        if (conf < conf_threshold_) continue;

        const int g0 = grid_strides_[ai].grid0;
        const int g1 = grid_strides_[ai].grid1;
        const int st = grid_strides_[ai].stride;

        // ========== 修复：颜色、分类维度读取同步修正 ==========
        double color_score, num_score;
        cv::Point color_id, num_id;
        cv::minMaxLoc(
                ob.row(ai).colRange(9, 9 + NUM_COLORS),
                nullptr, &color_score, nullptr, &color_id);
            cv::minMaxLoc(
                ob.row(ai).colRange(9 + NUM_COLORS, 9 + NUM_COLORS + NUM_CLASSES),
                nullptr, &num_score, nullptr, &num_id);

        float x1 = (ob.at<float>(ai, 0) + g0) * st;
        float y1 = (ob.at<float>(ai, 1) + g1) * st;
        float x2 = (ob.at<float>(ai, 2) + g0) * st;
        float y2 = (ob.at<float>(ai, 3) + g1) * st;
        float x3 = (ob.at<float>(ai, 4) + g0) * st;
        float y3 = (ob.at<float>(ai, 5) + g1) * st;
        float x4 = (ob.at<float>(ai, 6) + g0) * st;
        float y4 = (ob.at<float>(ai, 7) + g1) * st;

        // 还原到原图坐标
        Eigen::Matrix<float, 3, 4> apex_norm, apex_dst;
        apex_norm << x1, x2, x3, x4,
                     y1, y2, y3, y4,
                     1,   1,   1,   1;
        apex_dst = tm * apex_norm;

        Armor obj;
        obj.pts.resize(4);
        obj.pts[0] = cv::Point2f(apex_dst(0, 0), apex_dst(1, 0));
        obj.pts[1] = cv::Point2f(apex_dst(0, 1), apex_dst(1, 1));
        obj.pts[2] = cv::Point2f(apex_dst(0, 2), apex_dst(1, 2));
        obj.pts[3] = cv::Point2f(apex_dst(0, 3), apex_dst(1, 3));

         // 极角排序：计算四点中心，按相对于中心的夹角排序
 // 保证输出顺序为 左上→右上→右下→左下，与 PnP 世界点顺序严格对应
 {
     cv::Point2f c = (obj.pts[0] + obj.pts[1] + obj.pts[2] + obj.pts[3]) * 0.25f;
     std::pair<float, int> angles[4];
     for (int k = 0; k < 4; k++)
         angles[k] = {atan2(obj.pts[k].y - c.y, obj.pts[k].x - c.x), k};
     std::sort(angles, angles + 4);
     // 现在 angles 是逆时针顺序：右上→左上→左下→右下
     // 找到左上角（x+y 最小）作为顺时针起点
     cv::Point2f sorted_pts[4];
     for (int k = 0; k < 4; k++) sorted_pts[k] = obj.pts[angles[k].second];
     int start = 0;
     float min_sum = sorted_pts[0].x + sorted_pts[0].y;
     for (int k = 1; k < 4; k++) {
         float s = sorted_pts[k].x + sorted_pts[k].y;
         if (s < min_sum) { min_sum = s; start = k; }
     }
     obj.pts[0] = sorted_pts[(start + 0) % 4];  // 左上
     obj.pts[1] = sorted_pts[(start + 1) % 4];  // 右上
     obj.pts[2] = sorted_pts[(start + 2) % 4];  // 右下
     obj.pts[3] = sorted_pts[(start + 3) % 4];  // 左下
 }


         
        obj.box = cv::boundingRect(obj.pts);
        obj.color = color_id.x;
        obj.number = num_id.x;
        obj.prob = conf;

        result.push_back(obj);
    }
    return result;
}

//非极大值抑制(剔除同一装甲重复预测的重叠框，置信度优先保留最优框)+同目标角点平滑融合
std::vector<Armor> Detector::nmsMerge(std::vector<Armor>& objs)
{
    if (objs.empty()) return {};

    // 按置信度降序排序，先处理置信度最高的框
    std::sort(objs.begin(), objs.end(),
        [](const Armor& a, const Armor& b) { return a.prob > b.prob; });

    // TopK 截断，限制最大检测数量,防止画面目标过多拖慢推理速度。
    if (objs.size() > static_cast<size_t>(top_k_))
        objs.resize(top_k_);

    std::vector<Armor> result;
    //NMS 主循环：剔除重叠框
    /*取出当前最高置信的框，保留；
    遍历剩下所有框，计算 IOU（交并比，衡量重叠程度）
    IOU 超过阈值的，判定为同一个目标，直接标记删除
    重复直到所有框处理完*/
    std::vector<bool> removed(objs.size(), false);

    for (size_t i = 0; i < objs.size(); i++) {
        if (removed[i]) continue;
        result.push_back(objs[i]);// 保留当前最高置信框
        // 遍历后面所有候选框
        for (size_t j = i + 1; j < objs.size(); j++) {
            if (removed[j]) continue;
            float iou = polygonIou(objs[i].pts, objs[j].pts);
            //nms_threshold_：NMS 阈值，IOU 超过这个值判定为同一个装甲。
            if (iou > nms_threshold_) {
                removed[j] = true;// 标记这个低置信重叠框，直接丢弃
                // 同类别高重叠：合并角点取平均
                if (objs[i].number == objs[j].number &&
                    objs[i].color == objs[j].color &&
                    iou > MERGE_MIN_IOU &&
                    std::abs(objs[i].prob - objs[j].prob) < MERGE_CONF_ERROR) {
                    for (int k = 0; k < 4; k++)
                        result.back().pts.push_back(objs[j].pts[k]);
                }
            }
        }
    }

    //合并角点后求平均
    for (auto& a : result) {
        if (a.pts.size() > 4) {
            cv::Point2f pts_sum[4] = {cv::Point2f(0,0), cv::Point2f(0,0),
                                      cv::Point2f(0,0), cv::Point2f(0,0)};
            for (size_t j = 0; j < a.pts.size(); j++)
                pts_sum[j % 4] += a.pts[j];

            float divisor = a.pts.size() / 4.0f;
            a.pts.resize(4);
            for (int j = 0; j < 4; j++)
                a.pts[j] = pts_sum[j] / divisor;
        }
    }

    return result;
}
//透视矫正 + 裁剪标准化数字图，数字识别分支
void Detector::extractNumberImage(const cv::Mat& src, Armor& armor)
{
    const int light_length = 12;
    const int warp_height = 28;
    const int small_w = 32, large_w = 54;
    const cv::Size roi_size(20, 28);
//根据数字尺寸选择画布宽度
    bool is_large = (armor.number == 1 || armor.number == 7);
    int warp_width = is_large ? large_w : small_w;

    cv::Point2f src_pts[4] = {armor.pts[0], armor.pts[1], armor.pts[2], armor.pts[3]};
// dst_pts：矫正后画布四个角
    int top_y = (warp_height - light_length) / 2 - 1;
    int bot_y = top_y + light_length;
    cv::Point2f dst_pts[4] = {
        cv::Point(0, bot_y),    // 原图左上 → 矫正后左下
        cv::Point(0, top_y),    // 原图右上 → 矫正后左上
        cv::Point(warp_width - 1, top_y),// 原图右下 → 矫正后右上
        cv::Point(warp_width - 1, bot_y)    // 原图左下 → 矫正后右下
    };
//透视变换矫正倾斜，计算原图倾斜四边形 → 标准矩形的变换矩阵；，矩阵映射，把倾斜装甲拉正，消除透视畸变。
    cv::Mat ni;
    cv::warpPerspective(src, ni, cv::getPerspectiveTransform(src_pts, dst_pts),
                        cv::Size(warp_width, warp_height));

    int offset_x = (warp_width - roi_size.width) / 2;
    ni = ni(cv::Rect(offset_x, 0, roi_size.width, roi_size.height));
//灰度 + 自适应二值化 + 翻转输出
    cv::cvtColor(ni, ni, cv::COLOR_RGB2GRAY);
    cv::threshold(ni, ni, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::flip(ni, armor.number_img, 0);
}
//MLP 小型网络识别装甲数字
bool Detector::classifyNumber(Armor& armor)
{
    if (armor.number_img.empty()) return false;
    //图像归一化，适配模型输入
    cv::Mat img_f = armor.number_img.clone();
    img_f.convertTo(img_f, CV_32F, 1.0 / 255.0);

    cv::Mat blob = cv::dnn::blobFromImage(img_f);

    number_net_.setInput(blob);
    cv::Mat output = number_net_.forward();

    // Softmax归一化，把原始网络得分转为 0~1 概率，总和为 1，代表每个类别的置信度。
    cv::Mat softmax;
        // 数值稳定Softmax：先求全局最大值
    double max_val;
    cv::minMaxLoc(output.reshape(1, 1), nullptr, &max_val);
    cv::exp(output - max_val, softmax);//减去最大值是数值稳定优化，防止指数exp溢出报错。
  
    softmax /= cv::sum(softmax)[0];
//取概率最高类别，映射真实装甲数字
    double confidence;
    cv::Point class_id;
    cv::minMaxLoc(softmax.reshape(1, 1), nullptr, &confidence, nullptr, &class_id);

    armor.confidence = confidence;
    int label = class_id.x;

    // 标签索引 -> 装甲数字ID 正确映射
    // label.txt 顺序: 0=1, 1=2, 2=3, 3=4, 4=5, 5=outpost, 6=guard, 7=base, 8=negative
    switch (label) {
        case 0: armor.number = 1; break;
        case 1: armor.number = 2; break;
        case 2: armor.number = 3; break;
        case 3: armor.number = 4; break;
        case 4: armor.number = 5; break;
        case 5: armor.number = 6; break;
        case 6: armor.number = 0; break;
        case 7: armor.number = 7; break;
        default: armor.confidence = 0; return false;
    }
    return true;
}

//单帧推理流水线
std::vector<Armor> Detector::detectWithModel(const cv::Mat& bgr_img)
{
    if (!compiled_model_ || bgr_img.empty()) return {};

    try {
        // 1. 图像预处理
        Eigen::Matrix3f tm;
        cv::Mat rim = letterbox(bgr_img, tm);

        cv::Mat blob = cv::dnn::blobFromImage(
        rim, 1.0, cv::Size(INPUT_W, INPUT_H),
        cv::Scalar(0, 0, 0), true);

        // 3. OpenVINO推理
        ov::Tensor input_tensor(
            ov::element::f32,
            ov::Shape{1, 3, INPUT_W, INPUT_H},
            blob.ptr(0));//把 Blob 数据包装成 OpenVINO 的输入张量；
        //把 Blob 数据包装成 OpenVINO 的输入张量；
        auto infer_request = compiled_model_->create_infer_request();
        //创建推理请求，把输入张量塞进去；
        infer_request.set_input_tensor(input_tensor);
        //网络前向推理，所有卷积运算都在这一步完成
        infer_request.infer();

        // 4. 读取输出张量，形状是 [1, 输出维度, 总网格数]
        auto output_tensor = infer_request.get_output_tensor();
        auto shape = output_tensor.get_shape();
        cv::Mat output_buffer(shape[1], shape[2], CV_32F, output_tensor.data());

        // 5. 解码 + NMS + 数字识别
        //解码，遍历所有网格点，把模型输出的原始数值，还原成原图上的装甲四点坐标.
        auto candidates = decodeOutputs(output_buffer, tm);// 转成 OpenCV 矩阵方便读取
        auto armors = nmsMerge(candidates);

        for (auto& a : armors) {
            extractNumberImage(bgr_img, a);
            classifyNumber(a);
        }

        return armors;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] 模型推理异常: " << e.what() << std::endl;
        return {};
    }
}

void Detector::drawModelResults(cv::Mat& img, std::vector<Armor>& armors, const PoseResult& pr)
{
    for (auto& a : armors) {
        // 绘制装甲四边形
        for (int i = 0; i < 4; i++)
            cv::line(img, a.pts[i], a.pts[(i + 1) % 4],
                     cv::Scalar(0, 255, 0), 2);
        cv::line(img, a.pts[0], a.pts[2], cv::Scalar(0, 255, 0), 1);
        cv::line(img, a.pts[1], a.pts[3], cv::Scalar(0, 255, 0), 1);

        // 绘制位姿与ID信息
        cv::putText(img, "yaw:" + std::to_string(pr.yaw).substr(0, 5),
                    cv::Point(static_cast<int>(a.pts[0].x),
                              static_cast<int>(a.pts[0].y - 50)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::putText(img, "dis:" + std::to_string(pr.distance).substr(0, 4),
                    cv::Point(static_cast<int>(a.pts[0].x),
                              static_cast<int>(a.pts[0].y - 20)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        cv::putText(img, "ID:" + std::to_string(a.number) + (a.color == 0 ? "B" : "R"),
                    cv::Point(static_cast<int>(a.pts[0].x),
                              static_cast<int>(a.pts[0].y - 65)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255), 1);
    }
}