/*******************************************************
 * RTTLIO: Tightly-Coupled WiFi RTT/LiDAR/IMU Integration
 *
 * WiFi RTT ranging is integrated as a tightly-coupled factor into
 * a LiDAR/IMU sliding-window and batch factor graph optimizer.
 *
 * Based on the LiDAR/IMU framework of lili-om.
 * Li, K., Li, M., & Hanebeck, U. D. (2021). Towards high-performance
 * solid-state-lidar-inertial odometry and mapping. IEEE RA-L, 6(3).
 *******************************************************/

#include <fstream>
#include "utils/common.h"
#include "utils/math_tools.h"
#include "utils/timer.h"
#include "utils/TimerManager.h"
#include "utils/random_generator.hpp"
#include "factors/LidarKeyframeFactor.h"
#include "factors/LidarPoseFactor.h"
#include "factors/ImuFactor.h"
#include "factors/PriorFactor.h"
#include "factors/rtt.hpp"
#include "factors/Preintegration.h"
#include "factors/MarginalizationFactor.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <wifi_rtt_msgs/RTT_Raw_Array.h>

#include <std_msgs/Float32MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>

#include <random>

using namespace gtsam;

#define lossKernel 1.0
#define stateSize 3


class Estimator {
public:
    struct Solution
    {
        double unixtime;
        double timestamp; // float timestamp
        double gps_week;  // gps week
        double gps_sec;   // gps second
        double longitude;
        double latitude;
        double altitude;
        double roll;
        double pitch;
        double yaw;
    };

private:
    ros::NodeHandle nh; // node handle

    ros::Subscriber sub_surf; // sub surface features
    ros::Subscriber sub_odom; // sub odom from scan-to-scan matching
    ros::Subscriber sub_each_odom; // sub relative motion between two scan
    ros::Subscriber sub_full_cloud; // sub full clouds from lidar odomtry node
    ros::Subscriber sub_imu; // sub imu data

    ros::Publisher pub_map; // publish map
    ros::Publisher pub_odom; // pub odom mapped
    ros::Publisher pub_batch_path; // pub batch traj
    ros::Publisher pub_poses; // pub trajectories in pcl format
    ros::Publisher pub_surf; //publish surface map
    ros::Publisher pub_full; // publish raw scan with transformed pose from tc fusion

    ros::Publisher pub_sub_gl_map; // publish sub map

    nav_msgs::Odometry odom_mapping; // odom from tc fusion and lc fusion
    nav_msgs::Odometry odom_init_kf;

    /* information flag from lidar odometry */
    bool new_surf = false;
    bool new_odom = false;
    bool new_each_odom = false;
    bool new_full_cloud = false;

    /* time for the information from lidar odometry */
    double time_new_odom;

    /* batch searching range */
    int search_range = 6;
    int batch_activate_idx = 0;

    /* parameter for lidar feature selection */
    int feature_res_num;
    int rand_set_num;
    int batch_feature_res_num;
    int batch_rand_set_num;
    bool random_select;

    std::chrono::high_resolution_clock::time_point system_start_time;
    int frame_count = 0;

    pcl::PointCloud<PointType>::Ptr surf_last;
    pcl::PointCloud<PointType>::Ptr full_cloud;
    vector<pcl::PointCloud<PointType>::Ptr> full_clouds_ds;

    pcl::PointCloud<PointType>::Ptr surf_last_ds;

    vector<pcl::PointCloud<PointType>::Ptr> surf_lasts_ds;

    pcl::PointCloud<PointType>::Ptr surf_local_map;
    pcl::PointCloud<PointType>::Ptr surf_global_map;

    pcl::PointCloud<PointType>::Ptr surf_local_map_ds;
    pcl::PointCloud<PointType>::Ptr surf_global_map_ds; // downsampled

    vector<pcl::PointCloud<PointType>::Ptr> vec_surf_cur_pts; // save points
    vector<pcl::PointCloud<PointType>::Ptr> vec_surf_normal; // save normals
    vector<vector<double>> vec_surf_scores; // constant * weights

    map<int, map<int, pcl::PointCloud<PointType>::Ptr>> gl_vec_surf_cur_pts; // save points
    map<int, map<int, pcl::PointCloud<PointType>::Ptr>> gl_vec_surf_cur_pts_startend; // save points

    map<int, map<int, vector<double>>> gl_vec_surf_scores; // constant * weights
    map<int, map<int, vector<double>>> gl_vec_surf_scores_startend; // constant * weights

    map<int, map<int, vector<vector<double>>>> gl_vec_surf_normals_cents;
    map<int, map<int, vector<vector<double>>>> gl_vec_surf_normals_cents_startend;

    pcl::PointCloud<PointType>::Ptr latest_key_frames;
    pcl::PointCloud<PointType>::Ptr latest_key_frames_ds;
    pcl::PointCloud<PointType>::Ptr his_key_frames;
    pcl::PointCloud<PointType>::Ptr his_key_frames_ds;

    pcl::PointCloud<PointXYZI>::Ptr pose_keyframe; //position of keyframe (only position)
    
    // Usage for PointPoseInfo
    // position: x, y, z
    // orientation: qw - w, qx - x, qy - y, qz - z
    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_keyframe, pose_info_keyframe_batch; //pose of keyframe (should be denser)

    pcl::PointCloud<PointXYZI>::Ptr pose_each_frame; //position of each frame
    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_each_frame; //pose of each frame

    PointXYZI select_pose; //
    PointType pt_in_local, pt_in_map; // ?? locam point,

    pcl::PointCloud<PointType>::Ptr global_map; // global map to be published
    pcl::PointCloud<PointType>::Ptr global_map_ds;

    vector<pcl::PointCloud<PointType>::Ptr> surf_frames;

    deque<pcl::PointCloud<PointType>::Ptr> recent_surf_keyframes;
    int latest_frame_idx;

    pcl::KdTreeFLANN<PointType>::Ptr kd_tree_surf_local_map;
    pcl::KdTreeFLANN<PointXYZI>::Ptr kd_tree_his_key_poses;

    vector<int> pt_search_idx;
    vector<float> pt_search_sq_dists;

    /* voxelgrid filter */
    pcl::VoxelGrid<PointType> ds_filter_surf;
    pcl::VoxelGrid<PointType> ds_filter_surf_map;
    pcl::VoxelGrid<PointType> ds_filter_his_frames;
    pcl::VoxelGrid<PointType> ds_filter_global_map;

    vector<int> vec_surf_res_cnt; // how many feature correspondance in each frame
    map<int, map<int, int>> gl_vec_surf_res_cnt;
    map<int, map<int, int>> gl_vec_surf_res_cnt_startend;

    // Form of the transformation
    vector<double> abs_pose;
    vector<double> last_pose;

    mutex mutual_exclusion;

    int max_num_iter;

    string gt_path;


    /* loop closure and back end optimization related stuffs */
    bool loop_closure_on;

    gtsam::NonlinearFactorGraph global_pose_graph;
    gtsam::NonlinearFactorGraph local_pose_graph;
    gtsam::Values global_init_estimate, local_init_estimate;
    gtsam::ISAM2 *isam;
    gtsam::Values global_estimated;

    gtsam::Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise;
    gtsam::noiseModel::Diagonal::shared_ptr constraint_noise;

    // Loop closure detection related
    bool loop_to_close;
    int closest_his_idx;
    int latest_frame_idx_loop;
    bool loop_closed;

    /* back end optimization related stuffs */
    int local_map_width;
    double lc_search_radius;
    int lc_map_width;
    float lc_icp_thres;
    double surfDSRange;

    int slide_window_width; // size of the sliding window
    bool enable_batch_fusion = false;
    int sms_fusion_level = 0;

    //index of keyframe
    vector<int> keyframe_idx; // accumulate all the new keyframes
    vector<int> keyframe_id_in_frame;

    vector<double> keyframe_time; //

    vector<vector<double>> abs_poses;

    int num_kf_sliding;

    /* imu related */
    vector<sensor_msgs::ImuConstPtr> imu_buf;
    nav_msgs::Odometry::ConstPtr odom_cur;
    vector<nav_msgs::Odometry::ConstPtr> each_odom_buf;
    double time_last_imu;
    double cur_time_imu;
    bool first_imu;
    vector<Preintegration*> pre_integrations;
    Eigen::Vector3d acc_0, gyr_0, g, tmp_acc_0, tmp_gyr_0;

    /* variables to save the states inside the sliding window*/
    vector<Eigen::Vector3d> Ps; // position 
    vector<Eigen::Vector3d> Vs; // velocity 
    vector<Eigen::Matrix3d> Rs; // rotation
    vector<Eigen::Vector3d> Bas; // bias of accelemeters
    vector<Eigen::Vector3d> Bgs; // bias of gyro
    vector<vector<double>> para_speed_bias; // speed and bias?

    //extrinsic imu boady frame to lidar
    Eigen::Quaterniond q_lb;
    Eigen::Vector3d t_lb;

    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;

    Eigen::Vector3d enu_pos, enu_ypr;

    // RTT related
    std::string sol_folder;
    std::string trajectory_path, trajectory_path_3d;
    std::string error_path;

    double ql2b_w, ql2b_x, ql2b_y, ql2b_z, tl2b_x, tl2b_y, tl2b_z;

    int idx_imu;
    double gravity;

    /* measurements size */
    int measSize = 0;
    int numOfRttFactors = 0;
    wifi_rtt_msgs::RTT_Raw_Array rtt_data;
    std::map<int, wifi_rtt_msgs::RTT_Raw_Array> rtt_data_array;

    double initial_rpy_[3] = {0};
    Eigen::Quaterniond initial_Quat = Eigen::Quaterniond::Identity();

    std::vector<std::vector<double>> gt_time_poses; //vector (time, pose)

    std::string result_path, tc_sw_result_path, lc_result_path, batch_result_path, residuals_path;
    std::string result_path_evo, batch_result_path_evo, lc_result_path_evo;

    bool GTinLocal; // visualize ground truth path

    ros::Publisher pub_path_gt;
    nav_msgs::Path path_gt;
    
    //first sliding window optimazition
    bool first_opt;

    // for marginalization
    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    /* state arrays for ceres solver */
    double **tmpQuat;
    double **tmpTrans;
    double **tmpSpeedBias;

    double **gl_tmpQuat;
    double **gl_tmpTrans;
    double **gl_tmpSpeedBias;

    bool marg = true;

    vector<int> imu_idx_in_kf;
    double time_last_loop = 0;

    string imu_topic;

    double surf_dist_thres;
    double kd_max_radius;
    bool save_pcd = false;


    double lidar_const = 0;
    int mapping_interval = 1;
    double lc_time_thres = 30.0;
    int start_idx = 0;

    string frame_id = "RTTLIO";
    string data_set;
    double runtime = 0;

public:
    

    Estimator(): nh("~") {
        // This is from xrjtestWiFiRTT
        ros::param::get("sol_folder", sol_folder);
        std::cout << "path of the solution file-> " << sol_folder << "\n";

        ros::param::get("trajectory_path", trajectory_path);
        ros::param::get("trajectory_path_3d", trajectory_path_3d);

        ros::param::get("error_path", error_path);

        std::cout << "path of the trajectory_3D_path file-> " << trajectory_path_3d << "\n";
        std::cout << "path of the error_path file-> " << error_path << "\n";

        // ENU_ref << ref_lon, ref_lat, ref_alt;

        std::map<int, wifi_rtt_msgs::RTT_Raw_Array>::iterator it;
        std::map<int, wifi_rtt_msgs::RTT_Raw_Array>::iterator itEnd;

        getRawDatafromCSV(rtt_data);
        measSize = rtt_data_array.size();

        it = rtt_data_array.begin();
        itEnd = rtt_data_array.end();
        int idx = 0;

        // ceres::Problem problem;
        // ceres::Solver::Options options;
        // options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // DENSE_QR
        // options.num_threads = 1;
        // options.max_num_iterations = 150; // max_num_iter
        // options.trust_region_strategy_type = ceres::DOGLEG;
        // // options.minimizer_progress_to_stdout = true;
        // options.use_nonmonotonic_steps = false;

        // ceres::Solver::Summary summary;

        // setupStateMemory();
        // initializeNewlyAddedGraph();
        // addParameterBlocksToGraph(problem);
        // addRttRangeFactors(problem);
        // addMotionModelFactors(problem);
        // solveFactorGraph(problem, options, summary);

        // xrj_evaluateSol();

        // get result path from current namespace
        if (nh.getParam("result_path", result_path)){
          std::cout<<"result_path para"<<result_path<<std::endl;
        }
        initializeParameters();
        allocateMemory();

        pub_sub_gl_map = nh.advertise<sensor_msgs::PointCloud2>("/sub_global_cloud", 100);

        pub_path_gt = nh.advertise<nav_msgs::Path>("path_gt", 1000);

        // GTReader(gt_path);

        sub_full_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/full_point_cloud", 100, &Estimator::full_cloudHandler, this);
        sub_surf = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, &Estimator::surfaceLastHandler, this);
        sub_odom = nh.subscribe<nav_msgs::Odometry>("/odom", 10, &Estimator::odomHandler, this);
        sub_each_odom = nh.subscribe<nav_msgs::Odometry>("/each_odom", 10, &Estimator::eachOdomHandler, this);

        sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 2000, &Estimator::imuHandler, this);

        pub_map = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 2);
        pub_odom = nh.advertise<nav_msgs::Odometry>("/odom_mapped", 2);
        pub_batch_path = nh.advertise<nav_msgs::Path>("/path_batch", 5000);
        pub_poses = nh.advertise<sensor_msgs::PointCloud2>("/trajectory", 2);
        pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/map_surf_less_flat", 2);
        pub_full = nh.advertise<sensor_msgs::PointCloud2>("/raw_scan", 2);
    }

    ~Estimator() {}

    /* get raw data from CSV file */
    void getRawDatafromCSV(wifi_rtt_msgs::RTT_Raw_Array &rtt_data)
    {
        // while(1)
        {
            std::chrono::milliseconds dura(1000); // this thread sleep for any ms
            std::this_thread::sleep_for(dura);
            // load image list
            FILE *solFile;
            std::cout << "sol path" << sol_folder << std::endl;
            solFile = std::fopen(sol_folder.c_str(), "r");
            if (solFile == NULL)
            {
                printf("cannot find file: solution File \n", sol_folder.c_str());
                // ROS_BREAK();
            }
            char line[30000];

            int current_idx = 0;
            int gt_idx = 0;

            if (!rtt_data.RTT_Raws.size())
            {

                while ((fscanf(solFile, "%[^\n]", line)) != EOF)
                {
                    fgetc(solFile); // Reads in '\n' character and moves file
                                    // stream past delimiting character
                    // printf("Line = %s \n", line);
                    std::stringstream ss(line); // split into three string
                    vector<string> result;
                    while (ss.good())
                    {
                        string substr;
                        getline(ss, substr, ',');
                        result.push_back(substr);
                        std::cout << std::setprecision(17);
                    }

                    /* only record fix solution */
                    // if (strtod((result[6]).c_str(), NULL) != 1.0) {
                    //     std::cout<<"through non fix solution " <<std::endl;
                    //     continue;
                    //                 }

                    wifi_rtt_msgs::RTT_Raw rtt_raw;
                    // GroundTruth_.resize(112,3);
                    // std::map<int, wifi_rtt_msgs::RTT_Raw_Array> rtt_data_array;

                    /* results from ap */
                    rtt_raw.unixtime = strtod((result[0]).c_str(), NULL);
                    rtt_raw.timestamp = int(strtod((result[10]).c_str(), NULL));

                    if (rtt_raw.unixtime == 0)
                        rtt_raw.unixtime = 1761023969.213;
                    // for FJ data: 1723206425.776
                    // for 1123podium data 2: 1732339122.758
                    // for 1123podium data 1: 1732337538.252
                    // for 1021 parking lot: 1761023969.213

                    rtt_raw.gt_pos_x = strtod((result[1]).c_str(), NULL);
                    rtt_raw.gt_pos_y = strtod((result[2]).c_str(), NULL);
                    rtt_raw.gt_pos_z = strtod((result[3]).c_str(), NULL);
                    // gt_idx++;

                    rtt_raw.ap_pos_x = strtod((result[5]).c_str(), NULL);
                    rtt_raw.ap_pos_y = strtod((result[6]).c_str(), NULL);
                    rtt_raw.ap_pos_z = strtod((result[7]).c_str(), NULL);

                    gt_idx = rtt_raw.timestamp - 1;
                    // std::cout << "current timestamp = " << rtt_raw.timestamp << std::endl;
                    // GroundTruth_(gt_idx, 0) = rtt_raw.gt_pos_x; // gt_pos_x
                    // GroundTruth_(gt_idx, 1) = rtt_raw.gt_pos_y; // gt_pos_y
                    // GroundTruth_(gt_idx, 2) = rtt_raw.gt_pos_z; // gt_pos_z

                    rtt_raw.range = strtod((result[4]).c_str(), NULL);
                    rtt_raw.raw_range = strtod((result[9]).c_str(), NULL);

                    rtt_raw.ap_index = strtod((result[8]).c_str(), NULL);

                    // rtt_data.RTT_Raws[i].range = strtod((result[4]).c_str(), NULL);
                    // rtt_data.RTT_Raws[i].raw_range = strtod((result[9]).c_str(), NULL);

                    // rtt_data.RTT_Raws[i].ap_index = strtod((result[8]).c_str(), NULL);

                    /* results from RTKLIB */
                    // Solution_.gps_week =strtod((result[0]).c_str(), NULL);
                    // Solution_.gps_sec =strtod((result[1]).c_str(), NULL);

                    // Solution_.latitude = strtod((result[2]).c_str(), NULL);
                    // Solution_.longitude = strtod((result[3]).c_str(), NULL);
                    // Solution_.altitude = strtod((result[4]).c_str(), NULL);

                    std::cout << std::setprecision(17);
                    // m_solutions_Vec[double(Solution_.gps_sec)] = Solution_;
                    //  std::cout<<"m_NMEAVector.size() = "<<m_NMEAVector.size()<<std::endl;

                    // slice gnss data
                    if (rtt_data_array.size() == 0)
                    {
                        if (current_idx == 0)
                        {
                            current_idx = rtt_raw.timestamp;
                        }

                        if (rtt_raw.timestamp - current_idx == 1)
                        {
                            rtt_data_array[rtt_raw.timestamp - 1] = rtt_data;
                            // std::cout " current_idx: " << rtt_raw.timestamp - 1 << std::endl;
                            rtt_data.RTT_Raws.clear();
                            // GroundTruth_(gt_idx,0) = strtod((result[1]).c_str(), NULL); //gt_pos_x
                            // GroundTruth_(gt_idx,1) = strtod((result[2]).c_str(), NULL); //gt_pos_y
                            // GroundTruth_(gt_idx,2) = strtod((result[3]).c_str(), NULL); //gt_pos_z
                            // std::cout << "current GT = " << GroundTruth_(1, 0) << std::endl;
                            // gt_idx++;
                        }
                    }
                    else //! rtt_data_array.count(rtt_raw.timestamp) && (rtt_raw.timestamp != 1894)
                    {
                        if (rtt_raw.timestamp - current_idx > 1)
                        {
                            rtt_data_array[rtt_raw.timestamp - 1] = rtt_data;
                            // std::cout << " current_idx: " << rtt_raw.timestamp - 1 << std::endl;
                            rtt_data.RTT_Raws.clear();
                            current_idx = rtt_raw.timestamp - 1;
                            // GroundTruth_(gt_idx,0) = strtod((result[1]).c_str(), NULL); //gt_pos_x
                            // GroundTruth_(gt_idx,1) = strtod((result[2]).c_str(), NULL); //gt_pos_y
                            // GroundTruth_(gt_idx,2) = strtod((result[3]).c_str(), NULL); //gt_pos_z
                            // std::cout << "current GT = " << GroundTruth_(gt_idx,0) << std::endl;
                            // gt_idx++;
                        }
                        if (rtt_raw.timestamp - current_idx < 0)
                            break;
                    }

                    rtt_data.RTT_Raws.push_back(rtt_raw);
                    // std::cout << "current range: " << rtt_raw.range << std::endl;
                    // std::cout << "rtt_data.RTT_Raws size: " << rtt_data.RTT_Raws.size() << std::endl;

                    if (rtt_raw.timestamp == 317 && rtt_raw.ap_index == 4)
                    // for 1123 data: 938   for FJ data: 370
                    // for 1021 parking lot: 1764 for 1021 parking lot:317
                    {
                        rtt_data_array[rtt_raw.timestamp] = rtt_data;
                        break;
                    }
                    // {
                    //     rtt_data_array[rtt_raw.timestamp] = rtt_data;
                    //     // std::cout << " current_idx: " << rtt_raw.timestamp - 1 << std::endl;
                    //     rtt_data.RTT_Raws.clear();
                    //     current_idx = rtt_raw.timestamp - 1;
                    //     // GroundTruth_(gt_idx, 0) = strtod((result[1]).c_str(), NULL); // gt_pos_x
                    //     // GroundTruth_(gt_idx, 1) = strtod((result[2]).c_str(), NULL); // gt_pos_y
                    //     // GroundTruth_(gt_idx, 2) = strtod((result[3]).c_str(), NULL); // gt_pos_z
                    //     // gt_idx++;
                    // }
                }
                std::fclose(solFile);
                
                std::cout << "rtt data = " << rtt_data_array.begin()->second << std::endl;

                std::map<int, wifi_rtt_msgs::RTT_Raw_Array>::iterator it;
                std::map<int, wifi_rtt_msgs::RTT_Raw_Array>::iterator itEnd;
                it = rtt_data_array.begin();
                itEnd = rtt_data_array.end();
                std::cout << "rtt data initial: " << it->second << std::endl;
                std::cout << "rtt data end: " << itEnd->second << std::endl;

                while (it != itEnd) {
                    std::cout << "rtt data = " << it->second << std::endl;
                    it++;
                }
                std::cout << "rtt raw data size() = " << rtt_data_array.size() << std::endl;
                // std::cout << "GroundTruth = " << GroundTruth_ << std::endl;
                // std::cout << "GroundTruth initial = " << GroundTruth_(0,0) << std::endl;
                // std::cout << "GroundTruth final = " << GroundTruth_(111,0) << std::endl;
            }
        }

        


    }

    /* allocate the memory for the variables, very very important */
    void allocateMemory() {
        tmpQuat = new double *[slide_window_width]; // orientation
        tmpTrans = new double *[slide_window_width]; // translation
        tmpSpeedBias = new double *[slide_window_width];  // speed, bias_a, bias_g
        for (int i = 0; i < slide_window_width; ++i) {
            tmpQuat[i] = new double[4]; // w, x, y, z
            tmpTrans[i] = new double[3]; // x, y, z
            tmpSpeedBias[i] = new double[9]; // V,BA,BG
        }
        surf_last.reset(new pcl::PointCloud<PointType>());
        surf_local_map.reset(new pcl::PointCloud<PointType>());
        surf_global_map.reset(new pcl::PointCloud<PointType>());
        surf_last_ds.reset(new pcl::PointCloud<PointType>());
        surf_local_map_ds.reset(new pcl::PointCloud<PointType>());
        surf_global_map_ds.reset(new pcl::PointCloud<PointType>());
        full_cloud.reset(new pcl::PointCloud<PointType>());


        /* vectors related to the factors construction */
        for(int i = 0; i < slide_window_width; ++i) {
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent;
            tmpSurfCurrent.reset(new pcl::PointCloud<PointType>());
            vec_surf_cur_pts.push_back(tmpSurfCurrent);

            vector<double> tmpD;
            vec_surf_scores.push_back(tmpD);

            pcl::PointCloud<PointType>::Ptr tmpSurfNorm;
            tmpSurfNorm.reset(new pcl::PointCloud<PointType>());
            vec_surf_normal.push_back(tmpSurfNorm);

            vec_surf_res_cnt.push_back(0);
        }

        pose_keyframe.reset(new pcl::PointCloud<PointXYZI>());
        pose_info_keyframe.reset(new pcl::PointCloud<PointPoseInfo>());
        pose_info_keyframe_batch.reset(new pcl::PointCloud<PointPoseInfo>());

        pose_each_frame.reset(new pcl::PointCloud<PointXYZI>());
        pose_info_each_frame.reset(new pcl::PointCloud<PointPoseInfo>());

        global_map.reset(new pcl::PointCloud<PointType>());
        global_map_ds.reset(new pcl::PointCloud<PointType>());

        latest_key_frames.reset(new pcl::PointCloud<PointType>());
        latest_key_frames_ds.reset(new pcl::PointCloud<PointType>());
        his_key_frames.reset(new pcl::PointCloud<PointType>());
        his_key_frames_ds.reset(new pcl::PointCloud<PointType>());

        kd_tree_surf_local_map.reset(new pcl::KdTreeFLANN<PointType>());
        kd_tree_his_key_poses.reset(new pcl::KdTreeFLANN<PointXYZI>());
    }

    /* get parameters from the .yaml file */
    void initializeParameters() {
        gtsam::ISAM2Params isamPara;
        isamPara.relinearizeThreshold = 0.1;
        isamPara.relinearizeSkip = 1;
        isam = new gtsam::ISAM2(isamPara);

        if (!getParameter("/initialization/Euler_r", initial_rpy_[0])) {
            ROS_WARN("data_set not set, use default value: 0");
        }
        if (!getParameter("/initialization/Euler_p", initial_rpy_[1])) {
            ROS_WARN("data_set not set, use default value: 0");
        }
        if (!getParameter("/initialization/Euler_y", initial_rpy_[2])) {
            ROS_WARN("data_set not set, use default value: 0");
        }

        if (!getParameter("/initialization/gt_path", gt_path))
        {
            ROS_WARN("gt_path not set, use default value: 1");
            gt_path = "/dataset/test.csv";
        }

        poseCovariance.setZero();

        // Load parameters from yaml
        if (!getParameter("/common/data_set", data_set)) {
            ROS_WARN("data_set not set, use default value: utbm");
            data_set = "utbm";
        }

        if (!getParameter("/Estimator/enable_batch_fusion", enable_batch_fusion)) {
            ROS_WARN("enable_batch_fusion not set, use default value: false");
            enable_batch_fusion = false;
        }

        if (!getParameter("/Estimator/sms_fusion_level", sms_fusion_level)) {
            ROS_WARN("sms_fusion_level not set, use default value: 0");
            sms_fusion_level = 0;
        }

        if (!getParameter("/Estimator/surf_dist_thres", surf_dist_thres)) {
            ROS_WARN("surf_dist_thres not set, use default value: 0.1");
            surf_dist_thres = 0.1;
        }

        if (!getParameter("/Estimator/kd_max_radius", kd_max_radius)) {
            ROS_WARN("kd_max_radius not set, use default value: 1.0");
            kd_max_radius = 1.0;
        }

        if (!getParameter("/Estimator/save_pcd", save_pcd)) {
            ROS_WARN("save_pcd not set, use default value: false");
            save_pcd = false;
        }

        if (!getParameter("/Estimator/mapping_interval", mapping_interval)) {
            ROS_WARN("mapping_interval not set, use default value: 1");
            mapping_interval = 1;
        }

        if (!getParameter("/Estimator/lc_time_thres", lc_time_thres)) {
            ROS_WARN("lc_time_thres not set, use default value: 30.0");
            lc_time_thres = 30.0;
        }

        if (!getParameter("/Estimator/lidar_const", lidar_const)) {
            ROS_WARN("lidar_const not set, use default value: 1.0");
            lidar_const = 1.0;
        }

        if (!getParameter("/IMU/imu_topic", imu_topic)) {
            ROS_WARN("imu_topic not set, use default value: /imu/data");
            imu_topic = "/imu/data";
        }

        if (!getParameter("/Estimator/max_num_iter", max_num_iter)) {
            ROS_WARN("maximal iteration number of mapping optimization not set, use default value: 50");
            max_num_iter = 50;
        }

        if (!getParameter("/Estimator/loop_closure_on", loop_closure_on)) {
            ROS_WARN("loop closure detection set to false");
            loop_closure_on = false;
        }

        if (!getParameter("/Estimator/local_map_width", local_map_width)) {
            ROS_WARN("local_map_width not set, use default value: 5");
            local_map_width = 5;
        }

        if (!getParameter("/Estimator/lc_search_radius", lc_search_radius)) {
            ROS_WARN("lc_search_radius not set, use default value: 7.0");
            lc_search_radius = 7.0;
        }

        if (!getParameter("/Estimator/lc_map_width", lc_map_width)) {
            ROS_WARN("lc_map_width not set, use default value: 25");
            lc_map_width = 25;
        }

        if (!getParameter("/Estimator/lc_icp_thres", lc_icp_thres)) {
            ROS_WARN("lc_icp_thres not set, use default value: 0.3");
            lc_icp_thres = 0.3;
        }

        if (!getParameter("/Estimator/surfDSRange", surfDSRange)) {
            ROS_WARN("surfDSRange not set, use default value: 0.4");
            surfDSRange = 0.4;
        }

        if (!getParameter("/Estimator/slide_window_width", slide_window_width)) {
            ROS_WARN("slide_window_width not set, use default value: 4");
            slide_window_width = 4;
        }

        //load feature selection parameter
        if (!getParameter("/feature_selection/feature_res_num", feature_res_num)) {
            ROS_WARN("selected feature number is not set");
            feature_res_num = 60;
        }

        if (!getParameter("/feature_selection/rand_set_num", rand_set_num)) {
            ROS_WARN("point number of random set is not set");
            rand_set_num = 300;
        }

        if (!getParameter("/feature_selection/batch_feature_res_num", batch_feature_res_num)) {
            ROS_WARN("selected feature number is not set");
            batch_feature_res_num = 60;
        }

        if (!getParameter("/feature_selection/batch_rand_set_num", batch_rand_set_num)) {
            ROS_WARN("point number of random set is not set");
            batch_rand_set_num = 300;
        }

        if (!getParameter("/feature_selection/random_select", random_select)) {
            ROS_WARN("random select mode set to false");
            random_select = false;
        }

        //extrinsic parameters
        if (!getParameter("/Estimator/ql2b_w", ql2b_w))  {
            ROS_WARN("ql2b_w not set, use default value: 1");
            ql2b_w = 1;
        }

        if (!getParameter("/Estimator/ql2b_x", ql2b_x)) {
            ROS_WARN("ql2b_x not set, use default value: 0");
            ql2b_x = 0;
        }

        if (!getParameter("/Estimator/ql2b_y", ql2b_y)) {
            ROS_WARN("ql2b_y not set, use default value: 0");
            ql2b_y = 0;
        }

        if (!getParameter("/Estimator/ql2b_z", ql2b_z)) {
            ROS_WARN("ql2b_z not set, use default value: 0");
            ql2b_z = 0;
        }

        if (!getParameter("/Estimator/tl2b_x", tl2b_x)) {
            ROS_WARN("tl2b_x not set, use default value: 0");
            tl2b_x = 0;
        }

        if (!getParameter("/Estimator/tl2b_y", tl2b_y)) {
            ROS_WARN("tl2b_y not set, use default value: 0");
            tl2b_y = 0;
        }

        if (!getParameter("/Estimator/tl2b_z", tl2b_z)) {
            ROS_WARN("tl2b_z not set, use default value: 0");
            tl2b_z = 0;
        }

        if (!getParameter("/Estimator/search_range", search_range)) {
            ROS_WARN("search_range not set, use default value: 0");
            search_range = 0;
        }

        if (!getParameter("/visualization/GTinLocal", GTinLocal)) {
            ROS_WARN("GTinLocal not set, use default value: false");
            GTinLocal = false;
        }

        tc_sw_result_path = result_path + "tc_sw_result.csv";
        lc_result_path = result_path + "lc_result.csv";
//        result_path_evo = result_path + "enu_q_evo.csv";
//        batch_result_path_evo = result_path + "batch_enu_q_evo.csv";
//        lc_result_path_evo = result_path + "lc_enu_q_evo.csv";

        // clear output file
        std::ofstream tc_sw_output(tc_sw_result_path, std::ios::out);
        tc_sw_output.close();
        std::ofstream lc_output(lc_result_path, std::ios::out);
        lc_output.close();
//        std::ofstream res_evo_output(result_path_evo, std::ios::out);
//        res_evo_output.close();
//        std::ofstream res_lc_evo_output(lc_result_path_evo, std::ios::out);
//        res_lc_evo_output.close();

        last_marginalization_info = nullptr;

        idx_imu = 0;
        first_opt = false;
        cur_time_imu = -1;

        Rs.push_back(Eigen::Matrix3d::Identity());
        // double yaw_deg = -15.0; 
        // double yaw_rad = yaw_deg * M_PI / 180.0;
        // Eigen::Matrix3d R0 = Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        // Rs.push_back(R0);
        
        Ps.push_back(Eigen::Vector3d(-2, -2, 0)); // Eigen::Vector3d(-0.9, -0.2, 0.0) Eigen::Vector3d(-2.42735223, 1.81101471, 0.12618005) for 1123 _exp2  
        Vs.push_back(Eigen::Vector3d(0, 0, 0));

        Bas.push_back(Eigen::Vector3d::Zero());
        Bgs.push_back(Eigen::Vector3d(0, 0, 0));
        vector<double> tmpSpeedBias;
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);

        para_speed_bias.push_back(tmpSpeedBias);

        num_kf_sliding = 0;
        time_last_imu = 0;
        first_imu = false;

        nh.param<double>("/IMU/gravity", gravity, 9.805);
        g = Eigen::Vector3d(0, 0, gravity); //

        time_new_odom = 0;

        abs_pose.push_back(1);
        last_pose.push_back(1);

        latest_frame_idx = 0;

        vector<double> tmpOdom;
        tmpOdom.push_back(1);

        for (int i = 1; i < 7; ++i) {
            abs_pose.push_back(0);
            last_pose.push_back(0);
            tmpOdom.push_back(0);
        }
        abs_poses.push_back(tmpOdom);

        abs_pose = tmpOdom;

        ds_filter_surf.setLeafSize(surfDSRange, surfDSRange, surfDSRange);

        ds_filter_surf_map.setLeafSize(0.4, 0.4, 0.4);
        ds_filter_his_frames.setLeafSize(0.4, 0.4, 0.4);
        ds_filter_global_map.setLeafSize(0.2, 0.2, 0.2);

        odom_mapping.header.frame_id = frame_id;
        odom_init_kf.header.frame_id = frame_id;

        /* settings related to GTSAM iSAM */
        gtsam::Vector vector6p(6);
        gtsam::Vector vector6o(6);
        vector6p << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8; //1e-6, 1e-6, 1e-6
        vector6o << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        prior_noise = gtsam::noiseModel::Diagonal::Variances(vector6p);
        odom_noise = gtsam::noiseModel::Diagonal::Variances(vector6o);

        loop_to_close = false;
        loop_closed = false;

        /* extrinsic parameters */
        q_lb = Eigen::Quaterniond(ql2b_w, ql2b_x, ql2b_y, ql2b_z);
        t_lb = Eigen::Vector3d(tl2b_x, tl2b_y, tl2b_z);

        q_bl = q_lb.inverse();
        t_bl = - (q_bl * t_lb);
    }

    /* add height restricted factor*/
    bool addHeightFactors(ceres::Problem &problem)
    {
        // cout << "we have done line 2187" << endl;
        int rtt_frame_size = pose_info_keyframe_batch->points.size();
        double height = 0;                        // 5.58592007 for 1123 exp
        for (int m = 0; m < rtt_frame_size; m++) //
        {
            
            ceres::CostFunction *hs_function = new ceres::AutoDiffCostFunction<heightFactor, 1, 3>(new heightFactor(height));
            auto ID = problem.AddResidualBlock(hs_function, NULL, gl_tmpTrans[m]);
        }
    }

    void getGlobalLowerUpperIdx(double targetT, int& lower_idx, int& upper_idx)
    {
        double diff = 10000000;
        for (int i = 0; i < pose_info_keyframe_batch->points.size(); i++) {
            double t = keyframe_time[i];
            double diffTmp = fabs(t - targetT);
            if (diffTmp < diff && t < targetT) {
                lower_idx = i;
                diff = diffTmp;
            }
        }
        diff = 10000000;
        for (int i = 0; i < pose_info_keyframe_batch->points.size(); i++) {
            double t = keyframe_time[i];
            double diffTmp = fabs(t - targetT);
            if (diffTmp < diff && t > targetT) {
                upper_idx = i;
                diff = diffTmp;
            }
        }
    }

    bool addRttRangeFactors_noD(ceres::Problem &problem, double DDpsr_threshold)
    {
        int measSize = rtt_data_array.size();
        /* add pseudorange factor */
        std::map<int, wifi_rtt_msgs::RTT_Raw_Array>::iterator iter_pr;
        iter_pr = rtt_data_array.begin();
        int length = measSize;
        // cout << "we have done line 2187" << endl;
        int rtt_frame_size = pose_info_keyframe_batch->points.size();
        for (int m = 0; m < length; m++, iter_pr++) //
        {
            wifi_rtt_msgs::RTT_Raw_Array rtt_meas = (iter_pr->second);
            int sv_cnt = rtt_meas.RTT_Raws.size();
            // cout << "we have done line 2191" << endl;
            /* select the low and upper headers */
            int lower_idx = -1;
            int upper_idx = 10000000;
            //  get time by keyframe_time[lower_idx-1]
            if (rtt_meas.RTT_Raws[0].unixtime < pose_info_keyframe_batch->points[0].time || rtt_meas.RTT_Raws[0].unixtime > pose_info_keyframe_batch->points[rtt_frame_size - 1].time)
            {
                // cout << "we have done line 2193" << endl;
                // cout << "unixtime:" << rtt_meas.RTT_Raws[0].unixtime << " keyframe[0]:" << keyframe_time[0] << " keyframe[back]:" << keyframe_time.back() << endl;
                continue;
            }
            // cout << "back: " << keyframe_time.back() << endl;
            getGlobalLowerUpperIdx(rtt_meas.RTT_Raws[0].unixtime, lower_idx, upper_idx);

            int leftKey = lower_idx;
            int rightKey = upper_idx;

            cout << "unixtime:" << rtt_meas.RTT_Raws[0].unixtime << "leftKey: " << leftKey << " rightKey: " << rightKey << " keyframe_size: " << keyframe_time.size() << endl;
            double ts_ratio = (rtt_meas.RTT_Raws[0].unixtime - keyframe_time[leftKey]) / (keyframe_time[rightKey] - keyframe_time[leftKey]);
            cout << "ts_ratio: " << ts_ratio << endl;
            int factor_index = -1;

            for (int i = 0; i < sv_cnt; i++)
            {
                factor_index++;

                double s_g_x = 0, s_g_y = 0, s_g_z = 0;
                double pseudorange = 0;

                int ap_index = rtt_meas.RTT_Raws[i].ap_index;
                s_g_x = rtt_meas.RTT_Raws[i].ap_pos_x;
                s_g_y = rtt_meas.RTT_Raws[i].ap_pos_y;
                s_g_z = rtt_meas.RTT_Raws[i].ap_pos_z;

                pseudorange = rtt_meas.RTT_Raws[i].range;

                ceres::CostFunction *ps_function = new ceres::AutoDiffCostFunction<rttFactor_noD, 1, 3, 3>(new rttFactor_noD(ap_index, s_g_x, s_g_y, s_g_z, pseudorange, ts_ratio));
                // auto ID = problem.AddResidualBlock(ps_function, NULL, state_array[m]);

                /* eavaluate risiduals */
                double **parameters_evo;
                double *residuals_evo = new double;

                std::vector<int> block_sizes = ps_function->parameter_block_sizes();
                std::vector<Eigen::VectorXd> parameters_eigen;
                parameters_evo = new double *[block_sizes.size()];
                parameters_eigen.resize(block_sizes.size());

                for (int l = 0; l < static_cast<int>(block_sizes.size()); l++)
                {
                    parameters_eigen[l].resize(block_sizes[l]);
                    parameters_evo[l] = parameters_eigen[l].data();
                }

                Eigen::VectorXd residuals_evo_size;
                residuals_evo_size.resize(1);
                residuals_evo = residuals_evo_size.data();

                parameters_evo[0][0] = gl_tmpTrans[leftKey][0];  //;gt_time_poses[lower_idx_gt][1]  tmpTrans[leftKey][0]
                parameters_evo[0][1] = gl_tmpTrans[leftKey][1];  //;gt_time_poses[lower_idx_gt][2]  tmpTrans[leftKey][1]
                parameters_evo[0][2] = gl_tmpTrans[leftKey][2];  //;gt_time_poses[lower_idx_gt][3]  tmpTrans[leftKey][2]
                parameters_evo[1][0] = gl_tmpTrans[rightKey][0]; //;gt_time_poses[upper_idx_gt][1] tmpTrans[rightKey][0]
                parameters_evo[1][1] = gl_tmpTrans[rightKey][1]; //;gt_time_poses[upper_idx_gt][2] tmpTrans[rightKey][1]
                parameters_evo[1][2] = gl_tmpTrans[rightKey][2]; //;gt_time_poses[upper_idx_gt][3] tmpTrans[rightKey][2]

                ps_function->Evaluate(parameters_evo, residuals_evo, NULL);
                std::vector<double> residuals_evo_values(residuals_evo, residuals_evo + 1);
                // cout << "residuals: " << residuals_evo_values[0] << endl;
                // 将原来的 cout 行替换为下面代码
                // static std::ofstream csv_out;
                // if (!csv_out.is_open())
                // {
                //     std::string file = result_path;
                //     if (!file.empty() && file.back() != '/')
                //         file += '/';
                //     file += "output.csv";
                //     // 第一次打开时清空旧文件，之后以 append 写入
                //     csv_out.open(file, std::ios::out | std::ios::trunc);
                //     if (!csv_out.is_open())
                //     {
                //         std::cerr << "Failed to open CSV file: " << file << std::endl;
                //     }
                // }
                // // 记录：第一列 = 时间 (rtt_meas.RTT_Raws[0].unixtime)
                // //       第二列 = i （AP 编号的循环索引）
                // //       第三列 = residual
                // double csv_time = 0.0;
                // if (!rtt_meas.RTT_Raws.empty())
                //     csv_time = rtt_meas.RTT_Raws[0].unixtime;
                // csv_out << csv_time << "," << i << "," << residuals_evo_values[0] << "\n";
                // csv_out.flush();

                if (fabs(residuals_evo_values[0]) < DDpsr_threshold)
                {
                    auto ID = problem.AddResidualBlock(ps_function, NULL, gl_tmpTrans[leftKey], gl_tmpTrans[rightKey]);
                    numOfRttFactors++;
                }

                // --- DEBUG: 打印 residual 和 雅可比 检查敏感性 ---
                // 打印 residual
                ROS_INFO("RTT residual (time %.3f, sat %d, idx %d): %f", 
                         rtt_meas.RTT_Raws[0].unixtime, ap_index, i, residuals_evo_values[0]);
 
                // 计算雅可比（residuals x params），因为 residual size=1，jacobian 每块长度等于 block_size
                std::vector<double*> jacobians_dbg(block_sizes.size(), nullptr);
                std::vector<double> jac0(block_sizes[0]), jac1(block_sizes.size()>1? block_sizes[1]:0);
                jacobians_dbg[0] = jac0.data();
                if (block_sizes.size()>1) jacobians_dbg[1] = jac1.data();
                // Evaluate with jacobians
                ps_function->Evaluate(parameters_evo, residuals_evo, jacobians_dbg.data());
                double norm_j0 = 0.0;
                for (int jj=0; jj<block_sizes[0]; ++jj) norm_j0 += fabs(jac0[jj]);
                double norm_j1 = 0.0;
                if (block_sizes.size()>1) for (int jj=0; jj<block_sizes[1]; ++jj) norm_j1 += fabs(jac1[jj]);
                ROS_INFO("RTT jacobian abs-sum: left=%f right=%f", norm_j0, norm_j1);
            }
            // cout << "we have done line 2223" << endl;
        }

        return true;
    }

    void addLIOFactor() {
        int keyframe_size = pose_keyframe->points.size();
        int proc_kf_idx = keyframe_size - slide_window_width;
        //add poses to global graph
        if (keyframe_size == slide_window_width) {
            gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(pose_info_keyframe->points[0].qw,
                                                           pose_info_keyframe->points[0].qx,
                                                           pose_info_keyframe->points[0].qy,
                                                           pose_info_keyframe->points[0].qz);
            gtsam::Point3 transition = gtsam::Point3(pose_keyframe->points[0].x,
                                                     pose_keyframe->points[0].y,
                                                     pose_keyframe->points[0].z);

            // Initialization for global pose graph
            local_pose_graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(rotation, transition), prior_noise));
            local_init_estimate.insert(0, gtsam::Pose3(rotation, transition));
        }

            /* insert all the dense regular frames between two keyframes */
        else if(keyframe_size > slide_window_width) {
            gtsam::Rot3 rotationLast = gtsam::Rot3::Quaternion(pose_info_keyframe->points[proc_kf_idx-1].qw,
                                                               pose_info_keyframe->points[proc_kf_idx-1].qx,
                                                               pose_info_keyframe->points[proc_kf_idx-1].qy,
                                                               pose_info_keyframe->points[proc_kf_idx-1].qz);
            gtsam::Point3 transitionLast = gtsam::Point3(pose_keyframe->points[proc_kf_idx-1].x,
                                                         pose_keyframe->points[proc_kf_idx-1].y,
                                                         pose_keyframe->points[proc_kf_idx-1].z);

            gtsam::Rot3 rotationCur = gtsam::Rot3::Quaternion(pose_info_keyframe->points[proc_kf_idx].qw,
                                                              pose_info_keyframe->points[proc_kf_idx].qx,
                                                              pose_info_keyframe->points[proc_kf_idx].qy,
                                                              pose_info_keyframe->points[proc_kf_idx].qz);
            gtsam::Point3 transitionCur = gtsam::Point3(pose_keyframe->points[proc_kf_idx].x,
                                                        pose_keyframe->points[proc_kf_idx].y,
                                                        pose_keyframe->points[proc_kf_idx].z);
            gtsam::Pose3 poseFrom = gtsam::Pose3(rotationLast, transitionLast);
            gtsam::Pose3 poseTo = gtsam::Pose3(rotationCur, transitionCur);

            local_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(proc_kf_idx - 1,
                                                                    proc_kf_idx,
                                                                    poseFrom.between(poseTo),
                                                                    odom_noise));
            local_init_estimate.insert(proc_kf_idx, poseTo);
        }
    }

    /* 3D LiDAR Aided GNSS sliding window */
    void optimizeSlidingWindowWithLandMark() {
        if(slide_window_width < 1) return;
        if(keyframe_idx.size() < slide_window_width) return;

//        Timer t_optimize_slidingwindow("optimizeSlidingWindowWithLandMark");

        first_opt = true;

        int windowSize = keyframe_idx[keyframe_idx.size()-1] - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1; // always equals to slide_window_width

        kd_tree_surf_local_map->setInputCloud(surf_local_map_ds);
        sensor_msgs::PointCloud2::Ptr sw_clouds_map_ptr(new sensor_msgs::PointCloud2);
        pcl::toROSMsg(*surf_local_map_ds, *sw_clouds_map_ptr);
        sw_clouds_map_ptr->header.frame_id = "RTTLIO";
        pub_sub_gl_map.publish(*sw_clouds_map_ptr);

        //multiple iterations enables the re-searching of correspondance
        int iteration_num = 1;
        double DDpsr_threshold[iteration_num] = {10};
        double psr_threshold[iteration_num] = {10};

        for (int iterCount = 0; iterCount < iteration_num; ++iterCount) {

            ceres::LossFunction *lossFunction = new ceres::HuberLoss(lossKernel);
            ceres::Manifold *quatParameterization = new ceres::QuaternionManifold();
            ceres::Problem problem;

            /* init vectors for evaluation */
            std::vector<ceres::ResidualBlockId> imuPIIDs;

            //eigen to double: initialize the states
            for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); i++){

                if (iterCount == 0) {
                    Eigen::Quaterniond tmpQ(Rs[i]);
                    tmpQuat[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][0] =  tmpQ.w();
                    tmpQuat[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][1] =  tmpQ.x();
                    tmpQuat[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][2] =  tmpQ.y();
                    tmpQuat[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][3] =  tmpQ.z();
                    tmpTrans[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][0] = Ps[i][0];
                    tmpTrans[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][1] = Ps[i][1];
                    tmpTrans[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][2] = Ps[i][2];

                    abs_poses[i][0] = tmpQ.w();
                    abs_poses[i][1] = tmpQ.x();
                    abs_poses[i][2] = tmpQ.y();
                    abs_poses[i][3] = tmpQ.z();
                    abs_poses[i][4] = Ps[i][0];
                    abs_poses[i][5] = Ps[i][1];
                    abs_poses[i][6] = Ps[i][2];

                    for (int j = 0; j < 9; j++) {
                        tmpSpeedBias[i - keyframe_idx[keyframe_idx.size() - slide_window_width]][j] = para_speed_bias[i][j];
                    }
                }

                //add lidar parameters
                problem.AddParameterBlock(tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 3);
                problem.AddParameterBlock(tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 4, quatParameterization);

                //add IMU parameters
                problem.AddParameterBlock(tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 9);

            }

            abs_pose = abs_poses.back();
#if 1
            /* add the marginalization factor */
            if (last_marginalization_info) {
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                problem.AddResidualBlock(marginalization_factor, NULL,
                                         last_marginalization_parameter_blocks);
            }


            /* prior factor for imu preintegration typically this is not used 
             * this is to increase the stability of IMU bias estimation, this can further be improved by set the upper bound of the IMU bias, refer to the FGO project
            */
            if(!marg) {
                //add prior factor
                for(int i = 0; i < slide_window_width - 1; i++) {

                    vector<double> tmps;
                    for(int j = 0; j < 9; j++) {
                        tmps.push_back(tmpSpeedBias[i][j]);
                    }
                    ceres::CostFunction *speedBiasPriorFactor = SpeedBiasPriorFactorAutoDiff::Create(tmps);
                    problem.AddResidualBlock(speedBiasPriorFactor, NULL, tmpSpeedBias[i]);
                }

            }
#endif

#if 1 // IMU factor
            /* add IMU pre-integration factor */
            double sum_imu_residual = 0;
            for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx < keyframe_idx.back(); ++idx) {
                ImuFactor *imuFactor = new ImuFactor(pre_integrations[idx+1]);
                ImuFactor *imuFactor_test = new ImuFactor(pre_integrations[idx+1]);

                auto ID = problem.AddResidualBlock(imuFactor, NULL, tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpSpeedBias[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpTrans[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpQuat[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpSpeedBias[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
            }
#endif

#if 1 // scan to map lidar constraints
            /* add the LiDAR plannar factor */
            double sum_plane_residual = 0;
            for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx <= keyframe_idx.back(); idx++) {
                Eigen::Quaterniond Q2 = Eigen::Quaterniond(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]);
                Eigen::Vector3d T2 = Eigen::Vector3d(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2]);

                Eigen::Quaterniond Q2_ = Eigen::Quaterniond(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]);
                Eigen::Vector3d T2_ = Eigen::Vector3d(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2]);


                Q2 = Q2 * q_lb.inverse(); // orientation of imu to local world frame
                T2 = T2 - Q2 * t_lb; // translation of imu to local world frame

                int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width];

                if (surf_local_map_ds->points.size() > 50) {
                    findCorrespondingSurfFeatures(idx-1, Q2, T2);
                    featureSelection(idx-1, Q2_, T2_);

                    /* add plannar feature factorS */
                    for (int i = 0; i < vec_surf_res_cnt[idVec]; ++i) {
                        Eigen::Vector3d currentPt(vec_surf_cur_pts[idVec]->points[i].x,
                                                  vec_surf_cur_pts[idVec]->points[i].y,
                                                  vec_surf_cur_pts[idVec]->points[i].z);
                        Eigen::Vector3d norm(vec_surf_normal[idVec]->points[i].x,
                                             vec_surf_normal[idVec]->points[i].y,
                                             vec_surf_normal[idVec]->points[i].z);
                        double normInverse = vec_surf_normal[idVec]->points[i].intensity;
                        ceres::CostFunction *costFunction = LidarPlaneNormFactor::Create(currentPt, norm, q_lb, t_lb,
                                                                                         normInverse,
                                                                                         vec_surf_scores[idVec][i]);

                        auto ID = problem.AddResidualBlock(costFunction, lossFunction, tmpTrans[idx - keyframe_idx[
                                                                   keyframe_idx.size() - slide_window_width]],
                                                           tmpQuat[idx - keyframe_idx[keyframe_idx.size() -
                                                                                      slide_window_width]]);
                    }
                }
                else {
                    ROS_WARN("Not enough feature points from the map");
                }

            }
#endif


            /* setup the solver related options */
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // DENSE_QR
            options.num_threads = 1;
            options.max_num_iterations = 15; //max_num_iter
            options.trust_region_strategy_type = ceres::DOGLEG;
            options.minimizer_progress_to_stdout = false;
            options.use_nonmonotonic_steps = false;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            /* unify the quaterniond */
            for(int i = 0; i < windowSize; i++) {
                if(tmpQuat[i][0] < 0) {
                    Eigen::Quaterniond tmp(tmpQuat[i][0],
                            tmpQuat[i][1],
                            tmpQuat[i][2],
                            tmpQuat[i][3]);
                    tmp = unifyQuaternion(tmp);
                    tmpQuat[i][0] = tmp.w();
                    tmpQuat[i][1] = tmp.x();
                    tmpQuat[i][2] = tmp.y();
                    tmpQuat[i][3] = tmp.z();
                }
                Eigen::Quaterniond tmp(tmpQuat[i][0],
                                       tmpQuat[i][1],
                                       tmpQuat[i][2],
                                       tmpQuat[i][3]);
                tmp = unifyQuaternion(tmp);
                Eigen::Vector3d sky_point_vect = tmp * Eigen::Vector3d(0, 0, 1);
            }

            
        }

        MarginalizationInfo *marginalization_info = new MarginalizationInfo();

        if (last_marginalization_info) {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
                if (last_marginalization_parameter_blocks[i] == tmpTrans[0] ||
                        last_marginalization_parameter_blocks[i] == tmpQuat[0] ||
                        last_marginalization_parameter_blocks[i] == tmpSpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->AddResidualBlockInfo(residual_block_info);
        }

        /* marginalization for the prior imu pre-integration: typically this is not used */
        if(!marg) {
            //add prior factor
            for(int i = 0; i < slide_window_width - 1; i++) {

                vector<double*> tmp;
                tmp.push_back(tmpTrans[i]);
                tmp.push_back(tmpQuat[i]);

                vector<int> drop_set;
                if(i == 0) {
                    drop_set.push_back(0);
                    drop_set.push_back(1);
                }

                vector<double> tmps;
                for(int j = 0; j < 9; j++) {
                    tmps.push_back(tmpSpeedBias[i][j]);
                }

                vector<double*> tmp1;
                tmp1.push_back(tmpSpeedBias[i]);

                vector<int> drop_set1;
                if(i == 0) {
                    drop_set1.push_back(0);
                }
                ceres::CostFunction *speedBiasPriorFactor = SpeedBiasPriorFactorAutoDiff::Create(tmps);
                ResidualBlockInfo *residual_block_info1 = new ResidualBlockInfo(speedBiasPriorFactor, NULL,
                                                                                tmp1,
                                                                                drop_set1);

                marginalization_info->AddResidualBlockInfo(residual_block_info1);
            }

            marg = true;
        }

        //marginalization of imu
        ImuFactor *imuFactor = new ImuFactor(pre_integrations[keyframe_idx[keyframe_idx.size()-slide_window_width]+1]);

        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imuFactor, NULL,
                                                                       vector<double *>{
                                                                               tmpTrans[0],
                                                                               tmpQuat[0],
                                                                               tmpSpeedBias[0],
                                                                               tmpTrans[1],
                                                                               tmpQuat[1],
                                                                               tmpSpeedBias[1]
                                                                       },
                                                                       vector<int>{0, 1, 2});

        marginalization_info->AddResidualBlockInfo(residual_block_info);

#if 1
        //marginalization of lidar factors
        for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx <= keyframe_idx.back(); idx++) {
            ceres::LossFunction *lossFunction = new ceres::HuberLoss(lossKernel);
            // ceres::LossFunction *lossFunction = new ceres::CauchyLoss(2.0);
            int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width];
            if (surf_local_map_ds->points.size() > 50) {
                vector<double*> tmp;
                tmp.push_back(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
                tmp.push_back(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);

                for (int i = 0; i < vec_surf_res_cnt[idVec]; ++i) {
                    Eigen::Vector3d currentPt(vec_surf_cur_pts[idVec]->points[i].x,
                                              vec_surf_cur_pts[idVec]->points[i].y,
                                              vec_surf_cur_pts[idVec]->points[i].z);
                    Eigen::Vector3d norm(vec_surf_normal[idVec]->points[i].x,
                                         vec_surf_normal[idVec]->points[i].y,
                                         vec_surf_normal[idVec]->points[i].z);
                    double normInverse = vec_surf_normal[idVec]->points[i].intensity;

                    //LidarPlaneNormAnalyticFactor *costFunction = new LidarPlaneNormAnalyticFactor(currentPt, norm, normInverse);
                    ceres::CostFunction *costFunction = LidarPlaneNormFactor::Create(currentPt, norm, q_lb, t_lb, normInverse, vec_surf_scores[idVec][i]); //vec_surf_scores[idVec][i] * 1000 / vec_surf_res_cnt[idVec]

                    vector<int> drop_set;
                    if(idx == keyframe_idx[keyframe_idx.size()-slide_window_width]) {
                        drop_set.push_back(0);
                        drop_set.push_back(1);
                    }
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(costFunction, lossFunction,
                                                                                   tmp,
                                                                                   drop_set);
                    marginalization_info->AddResidualBlockInfo(residual_block_info);
                }

            }

            vec_surf_cur_pts[idVec]->clear();
            vec_surf_normal[idVec]->clear();
            vec_surf_res_cnt[idVec] = 0;
            vec_surf_scores[idVec].clear();
        }
#endif

        marginalization_info->PreMarginalize();
        marginalization_info->Marginalize();

#if 1
        /* shift for the states to be optimized */
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i < windowSize; ++i) {
            addr_shift[reinterpret_cast<long>(tmpTrans[i])] = tmpTrans[i-1];
            addr_shift[reinterpret_cast<long>(tmpQuat[i])] = tmpQuat[i-1];
            addr_shift[reinterpret_cast<long>(tmpSpeedBias[i])] = tmpSpeedBias[i-1];


        }

        vector<double *> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);


        if (last_marginalization_info) {
            delete last_marginalization_info;
        }
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
#endif

        //double to eigen
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); ++i){
            double dp0 = Ps[i][0] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            double dp1 = Ps[i][1] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            double dp2 = Ps[i][2] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            double pnorm = sqrt(dp0*dp0+dp1*dp1+dp2*dp2);

            double dv0 = Vs[i][0] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            double dv1 = Vs[i][1] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            double dv2 = Vs[i][2] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            double vnorm = sqrt(dv0*dv0+dv1*dv1+dv2*dv2);

            double dba1 = para_speed_bias[i][3] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];
            double dba2 = para_speed_bias[i][4] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4];
            double dba3 = para_speed_bias[i][5] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5];
            double dbg1 = para_speed_bias[i][6] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6];
            double dbg2 = para_speed_bias[i][7] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7];
            double dbg3 = para_speed_bias[i][8] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8];

            Eigen::Vector3d ba_tmp (tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3],
                                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4],
                                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5]);
            Eigen::Vector3d bg_tmp (tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6],
                                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7],
                                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8]);

            Eigen::Quaterniond dq = Eigen::Quaterniond (tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]).normalized().inverse() *
                    Eigen::Quaterniond(Rs[i]);
            double qnorm = dq.vec().norm();

            /* The trans difference between the initialized and optimized state */
            if(pnorm < 100) {
                abs_poses[i][4] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                abs_poses[i][5] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                abs_poses[i][6] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];

                Ps[i][0] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                Ps[i][1] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                Ps[i][2] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            } else {
                ROS_WARN("bad optimization result of p!!!!!!!!!!!!!");
            }

            /* The trans difference between the initialized and optimized state */
            if(qnorm < 10) {
                abs_poses[i][0] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                abs_poses[i][1] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                abs_poses[i][2] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
                abs_poses[i][3] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];

                Rs[i] = Eigen::Quaterniond (tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]).normalized().toRotationMatrix();
            } else
                ROS_WARN("bad optimization result of q!!!!!!!!!!!!!");

            /* The vel difference between the initialized and optimized state */
            if(vnorm < 100) {
                for(int j = 0; j < 3; j++) {
                    para_speed_bias[i][j] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][j];
                }
                Vs[i][0] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                Vs[i][1] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                Vs[i][2] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            } else {
                ROS_WARN("bad optimization result of v!!!!!!!!!!!!!");
            }

            /* The bias of acc difference between the initialized and optimized state */
            if(abs(dba1) < 22) {
                para_speed_bias[i][3] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];
                Bas[i][0] = para_speed_bias[i][3];
            } else
//                ROS_WARN("bad ba1!!!!!!!!!!");

            if(abs(dba2) < 22) {
                para_speed_bias[i][4] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4];
                Bas[i][1] = para_speed_bias[i][4];
            } else
//                ROS_WARN("bad ba2!!!!!!!!!!");

            if(abs(dba3) < 22) {
                para_speed_bias[i][5] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5];
                Bas[i][2] = para_speed_bias[i][5];
            } else
//                ROS_WARN("bad ba3!!!!!!!!!!");

            if(abs(dbg1) < 22) {
                para_speed_bias[i][6] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6];
                Bgs[i][0] = para_speed_bias[i][6];
            } else
//                ROS_WARN("bad bg1!!!!!!!!!!");

            if(abs(dbg2) < 22) {
                para_speed_bias[i][7] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7];
                Bgs[i][1] = para_speed_bias[i][7];
            } else
//                ROS_WARN("bad bg2!!!!!!!!!!");

            if(abs(dbg3) < 22) {
                para_speed_bias[i][8] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8];
                Bgs[i][2] = para_speed_bias[i][8];
            }
//            else
//                ROS_WARN("bad bg3!!!!!!!!!!");

        }

        updatePose();
        /* construct batch lidar feature association */
        batchFeatureAssociation();
    }

    /* 3D LiDAR Aided GNSS batch optimization */
    void optimizeBatchWithLandMark() {
        if (pose_info_keyframe->points.size() < 6) return;
        if (enable_batch_fusion) {
            if ((pose_info_keyframe->points.size() - batch_activate_idx) < 1) {
                return;
            }
            else {
                batch_activate_idx = pose_info_keyframe->points.size();
            }
        }

        //        Timer t_optimize_batch("optimizeBatchWithLandMark");
        *pose_info_keyframe_batch = *pose_info_keyframe; //pose of each frame
        int cur_batch_size = pose_info_keyframe_batch->points.size();

        gl_tmpQuat = new double *[cur_batch_size]; // orientation
        gl_tmpTrans = new double *[cur_batch_size]; // translation
        gl_tmpSpeedBias = new double *[cur_batch_size];  // speed, bias_a, bias_g
        start_idx = 0;

        std::cout << &start_idx << std::endl;

        /* generate global map for LiDAR matching */
        surf_global_map.reset(new pcl::PointCloud<PointType>());
        std::cout << &start_idx << std::endl;
        surf_global_map_ds.reset(new pcl::PointCloud<PointType>());
        //multiple iterations enables the re-searching of correspondence
        int iteration_num = 5;
        std::cout << &start_idx << std::endl;
        double DDpsr_threshold[iteration_num] = {1000000000, 10, 8, 6, 5}; //= {1000000000, 10, 8, 6, 4, 3, 2.5}; // 1000000000, 10, 8, 6, 4, 3, 2.5
        std::cout << &start_idx << std::endl;
        for (int iterCount = 0; iterCount < iteration_num; iterCount++) { // 1

            ceres::LossFunction *lossFunction = NULL; //new ceres::HuberLoss(lossKernel)
            ceres::Manifold *quatParameterization = new ceres::QuaternionManifold();
            ceres::Problem problem;

            
            //eigen to double: initialize the states
            for (int i = start_idx; i < cur_batch_size; i++){

                gl_tmpQuat[i] = new double[4]; // w, x, y, z
                gl_tmpTrans[i] = new double[3]; // x, y, z
                gl_tmpSpeedBias[i] = new double[9]; // V,BA,BG

                Eigen::Quaterniond tmpQ(Rs[i]);
                // gt init
                gl_tmpQuat[i][0]  = pose_info_keyframe_batch->points[i].qw;
                gl_tmpQuat[i][1]  = pose_info_keyframe_batch->points[i].qx;
                gl_tmpQuat[i][2]  = pose_info_keyframe_batch->points[i].qy;
                gl_tmpQuat[i][3]  = pose_info_keyframe_batch->points[i].qz;
                gl_tmpTrans[i][0] = pose_info_keyframe_batch->points[i].x;
                gl_tmpTrans[i][1] = pose_info_keyframe_batch->points[i].y;
                gl_tmpTrans[i][2] = pose_info_keyframe_batch->points[i].z;

                for (int j=0; j<9; j++) {
                    gl_tmpSpeedBias[i][j] = para_speed_bias[i+1][j];
                }

                //add lidar parameters
                problem.AddParameterBlock(gl_tmpTrans[i], 3);
                problem.AddParameterBlock(gl_tmpQuat[i], 4, quatParameterization);

                if (i == 0){
                    problem.SetParameterBlockConstant(gl_tmpTrans[i]);
                    problem.SetParameterBlockConstant(gl_tmpQuat[i]);
                }
                
                //add IMU parameters
                problem.AddParameterBlock(gl_tmpSpeedBias[i], 9);

            }
        
            

            
            // relative attitude constraint
            for (int i = start_idx; i < cur_batch_size; i++) {
                Eigen::Quaterniond qi(pose_info_keyframe->points[i].qw,
                                      pose_info_keyframe->points[i].qx,
                                      pose_info_keyframe->points[i].qy,
                                      pose_info_keyframe->points[i].qz);
                qi = unifyQuaternion(qi);

                Eigen::Vector3d pi = Eigen::Vector3d(pose_info_keyframe->points[i].x,
                                                     pose_info_keyframe->points[i].y,
                                                     pose_info_keyframe->points[i].z);
                Eigen::Vector3d p_tmp = pi;
                int factor_count = 0;
                //backward attitude constraint
                for (int j=i; j>=start_idx; j-=1) {
                    if (factor_count == search_range) {
                        factor_count = 0;
                        break;
                    }
                    if (j == i) continue;
                    Eigen::Quaterniond qj(pose_info_keyframe->points[j].qw,
                                          pose_info_keyframe->points[j].qx,
                                          pose_info_keyframe->points[j].qy,
                                          pose_info_keyframe->points[j].qz);
                    Eigen::Vector3d pj = Eigen::Vector3d(pose_info_keyframe->points[j].x,
                                                         pose_info_keyframe->points[j].y,
                                                         pose_info_keyframe->points[j].z);
                    double p_range = (p_tmp - pj).norm();
                    if (p_range > 5/search_range) { //6
                        p_tmp = pj;
                        Eigen::Quaterniond const_diff = qi.inverse() * qj;
                        ceres::CostFunction* delta_q_factor = new ceres::AutoDiffCostFunction<delta_q_factor_auto,3,4,4>
                                (new delta_q_factor_auto(const_diff));
                        auto ID = problem.AddResidualBlock(delta_q_factor, NULL, gl_tmpQuat[i], gl_tmpQuat[j]);
                        factor_count++;
                    }
                }

                //forward attitude constraint
                for (int j=i; j<cur_batch_size; j++) {
                    if (factor_count == search_range) {
                        factor_count = 0;
                        break;
                    }
                    if (j == i) continue;
                    Eigen::Quaterniond qj(pose_info_keyframe->points[j].qw,
                                          pose_info_keyframe->points[j].qx,
                                          pose_info_keyframe->points[j].qy,
                                          pose_info_keyframe->points[j].qz);
                    Eigen::Vector3d pj = Eigen::Vector3d(pose_info_keyframe->points[j].x,
                                                         pose_info_keyframe->points[j].y,
                                                         pose_info_keyframe->points[j].z);
                    double p_range = (p_tmp - pj).norm();
                    if (p_range > 5/search_range) { // 6
                        p_tmp = pj;
                        Eigen::Quaterniond const_diff = qi.inverse() * qj;
                        ceres::CostFunction* delta_q_factor = new ceres::AutoDiffCostFunction<delta_q_factor_auto,3,4,4>
                                (new delta_q_factor_auto(const_diff));
                        auto ID = problem.AddResidualBlock(delta_q_factor, NULL, gl_tmpQuat[i], gl_tmpQuat[j]);
                        factor_count++;
                    }
                }
            }
            
            // /* add parameter block for receiver clock drift */

            /* add scan-to-multiscan factor */
            if (sms_fusion_level == 0) {
                /* add relative pose factor */
                //forward backward batch constraint
                for (int idx = start_idx + search_range; idx < cur_batch_size; idx++) {
                    for (int ms_i = 1; ms_i < search_range; ms_i++) {
                        Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_info_keyframe->points[idx].x,
                                                                   pose_info_keyframe->points[idx].y,
                                                                   pose_info_keyframe->points[idx].z) -
                                                   Eigen::Vector3d(pose_info_keyframe->points[idx - ms_i].x,
                                                                   pose_info_keyframe->points[idx - ms_i].y,
                                                                   pose_info_keyframe->points[idx - ms_i].z);
                        tmpTrans = Eigen::Quaterniond(pose_info_keyframe->points[idx - ms_i].qw,
                                                      pose_info_keyframe->points[idx - ms_i].qx,
                                                      pose_info_keyframe->points[idx - ms_i].qy,
                                                      pose_info_keyframe->points[idx - ms_i].qz).inverse() * tmpTrans;

                        Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_keyframe->points[idx - ms_i].qw,
                                                                        pose_info_keyframe->points[idx - ms_i].qx,
                                                                        pose_info_keyframe->points[idx - ms_i].qy,
                                                                        pose_info_keyframe->points[idx -
                                                                                                   ms_i].qz).inverse() *
                                                     Eigen::Quaterniond(pose_info_keyframe->points[idx].qw,
                                                                        pose_info_keyframe->points[idx].qx,
                                                                        pose_info_keyframe->points[idx].qy,
                                                                        pose_info_keyframe->points[idx].qz);
                        ceres::CostFunction *relativePose_factor = LidarPoseFactorBatchRelativeAutoDiff::Create(tmpQuat,
                                                                                                                tmpTrans);
                        problem.AddResidualBlock(relativePose_factor, NULL, gl_tmpTrans[idx - ms_i],
                                                 gl_tmpQuat[idx - ms_i], gl_tmpTrans[idx], gl_tmpQuat[idx]);
                    }
                }
                for (int idx = start_idx; idx < cur_batch_size - search_range; idx++) {
                    for (int ms_i = 1; ms_i < search_range; ms_i++) {
                        Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_info_keyframe->points[idx + ms_i].x,
                                                                   pose_info_keyframe->points[idx + ms_i].y,
                                                                   pose_info_keyframe->points[idx + ms_i].z) -
                                                   Eigen::Vector3d(pose_info_keyframe->points[idx].x,
                                                                   pose_info_keyframe->points[idx].y,
                                                                   pose_info_keyframe->points[idx].z);
                        tmpTrans = Eigen::Quaterniond(pose_info_keyframe->points[idx].qw,
                                                      pose_info_keyframe->points[idx].qx,
                                                      pose_info_keyframe->points[idx].qy,
                                                      pose_info_keyframe->points[idx].qz).inverse() * tmpTrans;

                        Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_keyframe->points[idx].qw,
                                                                        pose_info_keyframe->points[idx].qx,
                                                                        pose_info_keyframe->points[idx].qy,
                                                                        pose_info_keyframe->points[idx].qz).inverse() *
                                                     Eigen::Quaterniond(pose_info_keyframe->points[idx + ms_i].qw,
                                                                        pose_info_keyframe->points[idx + ms_i].qx,
                                                                        pose_info_keyframe->points[idx + ms_i].qy,
                                                                        pose_info_keyframe->points[idx + ms_i].qz);
                        ceres::CostFunction *relativePose_factor = LidarPoseFactorBatchRelativeAutoDiff::Create(tmpQuat,
                                                                                                                tmpTrans);
                        problem.AddResidualBlock(relativePose_factor, NULL, gl_tmpTrans[idx], gl_tmpQuat[idx],
                                                 gl_tmpTrans[idx + ms_i], gl_tmpQuat[idx + ms_i]);
                    }
                }
                // cout << "we have done line 3280" << endl;
                // vehicle static constraint
                if (0) {
                    int static_keyframe_idx = start_idx;
                    for (int idx = start_idx; idx < cur_batch_size; idx++) {
                        if (idx == static_keyframe_idx) continue;
                        Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_info_keyframe->points[idx].x,
                                                                   pose_info_keyframe->points[idx].y,
                                                                   pose_info_keyframe->points[idx].z) -
                                                   Eigen::Vector3d(pose_info_keyframe->points[static_keyframe_idx].x,
                                                                   pose_info_keyframe->points[static_keyframe_idx].y,
                                                                   pose_info_keyframe->points[static_keyframe_idx].z);
                        tmpTrans = Eigen::Quaterniond(pose_info_keyframe->points[static_keyframe_idx].qw,
                                                      pose_info_keyframe->points[static_keyframe_idx].qx,
                                                      pose_info_keyframe->points[static_keyframe_idx].qy,
                                                      pose_info_keyframe->points[static_keyframe_idx].qz).inverse() *
                                   tmpTrans;
                        if (tmpTrans.norm() < 0.05) {
                            Eigen::Quaterniond tmpQuat =
                                    Eigen::Quaterniond(pose_info_keyframe->points[static_keyframe_idx].qw,
                                                       pose_info_keyframe->points[static_keyframe_idx].qx,
                                                       pose_info_keyframe->points[static_keyframe_idx].qy,
                                                       pose_info_keyframe->points[static_keyframe_idx].qz).inverse() *
                                    Eigen::Quaterniond(pose_info_keyframe->points[idx].qw,
                                                       pose_info_keyframe->points[idx].qx,
                                                       pose_info_keyframe->points[idx].qy,
                                                       pose_info_keyframe->points[idx].qz);
                            ceres::CostFunction *relativePose_factor = LidarPoseFactorBatchRelativeAutoDiff::Create(
                                    tmpQuat, tmpTrans);
                            problem.AddResidualBlock(relativePose_factor, NULL, gl_tmpTrans[static_keyframe_idx],
                                                     gl_tmpQuat[static_keyframe_idx], gl_tmpTrans[idx],
                                                     gl_tmpQuat[idx]);
                        } else static_keyframe_idx = start_idx;
                    }
                }

            }
            else if (sms_fusion_level == 1) {

                /* add IMU pre-integration factor */
                for (int idx = start_idx; idx < cur_batch_size-1; ++idx) {
                    ImuFactor *imuFactor = new ImuFactor(pre_integrations[idx+1]);
                    auto ID = problem.AddResidualBlock(imuFactor, NULL, gl_tmpTrans[idx],
                                                       gl_tmpQuat[idx],
                                                       gl_tmpSpeedBias[idx],
                                                       gl_tmpTrans[idx+1],
                                                       gl_tmpQuat[idx+1],
                                                       gl_tmpSpeedBias[idx+1]);
                }

                /* add the LiDAR plannar factor */
                for (int idx = start_idx; idx < cur_batch_size; idx++) {
                    int idVec = idx; // start from 0

                    // start of search keyframes
                    int search_idx_start;
                    if (idx >= search_range + start_idx && idx < cur_batch_size - 1 - search_range) {
                        search_idx_start = idx - search_range;
                    }
                    else if (idx < search_range + start_idx) {
                        search_idx_start = start_idx;
                    }
                    else if (idx >= cur_batch_size - 1 - search_range) {
                        search_idx_start = cur_batch_size - 2*search_range - 1;
                    }
                    if (idx > cur_batch_size - 1 - search_range || idx < search_range + start_idx ) {
                        for (int j = search_idx_start; j <= search_idx_start + 2*search_range; j++) {
                            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent(new pcl::PointCloud<PointType>());
                            gl_vec_surf_cur_pts_startend[idx][j] = tmpSurfCurrent;
                            gl_vec_surf_res_cnt_startend[idx][j] = 0;
                            vector<double> tmpD;
                            gl_vec_surf_scores_startend[idx][j] = tmpD;
//                            vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> tmp_pnc;
                            vector<vector<double>> tmp_pnc;
                            gl_vec_surf_normals_cents_startend[idx][j] = tmp_pnc;
                        }
                        findGlobalCorrespondingSurfFeatures_Batch(idx, search_idx_start);
                        globalFeatureSelection_Batch(idx, search_idx_start);
                        /* add plannar feature factorS */
                        Eigen::Quaterniond q_(1., 0., 0., 0.);
                        Eigen::Vector3d t_(0., 0., 0.);
                        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
                            if (search_idx == idx) continue;
                            for (int i = 0; i < gl_vec_surf_res_cnt_startend[idVec][search_idx]; ++i) {
                                Eigen::Vector3d currentPt(gl_vec_surf_cur_pts_startend[idVec][search_idx]->points[i].x,
                                                          gl_vec_surf_cur_pts_startend[idVec][search_idx]->points[i].y,
                                                          gl_vec_surf_cur_pts_startend[idVec][search_idx]->points[i].z);
                                Eigen::Matrix<double, 6, 1> norm_cent;
                                norm_cent << gl_vec_surf_normals_cents_startend[idVec][search_idx][i][0],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][1],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][2],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][3],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][4],
                                        gl_vec_surf_normals_cents_startend[idVec][search_idx][i][5];

                                ceres::CostFunction *costFunction = BinaryLidarPlaneNormFactor::Create(currentPt, norm_cent, gl_vec_surf_scores_startend[idVec][search_idx][i]);

                                auto ID = problem.AddResidualBlock(costFunction, lossFunction, gl_tmpTrans[idx], gl_tmpQuat[idx], gl_tmpTrans[search_idx], gl_tmpQuat[search_idx]);
                            }
                        }
                    }
                    else {
                        /* add plannar feature factorS */
                        Eigen::Quaterniond q_(1., 0., 0., 0.);
                        Eigen::Vector3d t_(0., 0., 0.);
                        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
                            if (search_idx == idx) continue;
                            for (int i = 0; i < gl_vec_surf_res_cnt[idVec][search_idx]; ++i) {
                                Eigen::Vector3d currentPt(gl_vec_surf_cur_pts[idVec][search_idx]->points[i].x,
                                                          gl_vec_surf_cur_pts[idVec][search_idx]->points[i].y,
                                                          gl_vec_surf_cur_pts[idVec][search_idx]->points[i].z);
                                Eigen::Matrix<double, 6, 1> norm_cent;
                                norm_cent << gl_vec_surf_normals_cents[idVec][search_idx][i][0],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][1],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][2],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][3],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][4],
                                        gl_vec_surf_normals_cents[idVec][search_idx][i][5];
                                ceres::CostFunction *costFunction = BinaryLidarPlaneNormFactor::Create(currentPt, norm_cent, gl_vec_surf_scores[idVec][search_idx][i]); //vec_surf_scores[idVec][i] * 1000 / vec_surf_res_cnt[idVec]
                                auto ID = problem.AddResidualBlock(costFunction, lossFunction, gl_tmpTrans[idx], gl_tmpQuat[idx], gl_tmpTrans[search_idx], gl_tmpQuat[search_idx]); // lossFunction
                            }
                        }
                    }
                }
            }

            /* write a csv file of residual*/

            /* add RTT Range factor */
            addHeightFactors(problem);
            // addRttRangeFactors(problem, DDpsr_threshold[iterCount]);
            addRttRangeFactors_noD(problem, DDpsr_threshold[iterCount]);
            
            // /* eavaluate risiduals */
            // double **parameters_evo;
            // double *residuals_evo = new double;

            // std::vector<int> block_sizes = ps_function->parameter_block_sizes();
            // std::vector<Eigen::VectorXd> parameters_eigen;
            // parameters_evo = new double *[block_sizes.size()];
            // parameters_eigen.resize(block_sizes.size());

            // for (int l = 0; l < static_cast<int>(block_sizes.size()); l++)
            // {
            //     parameters_eigen[l].resize(block_sizes[l]);
            //     parameters_evo[l] = parameters_eigen[l].data();
            // }

            // Eigen::VectorXd residuals_evo_size;
            // residuals_evo_size.resize(1);
            // residuals_evo = residuals_evo_size.data();

            // parameters_evo[0][0] = gl_tmpTrans[leftKey][0];  //;gt_time_poses[lower_idx_gt][1]  tmpTrans[leftKey][0]
            // parameters_evo[0][1] = gl_tmpTrans[leftKey][1];  //;gt_time_poses[lower_idx_gt][2]  tmpTrans[leftKey][1]
            // parameters_evo[0][2] = gl_tmpTrans[leftKey][2];  //;gt_time_poses[lower_idx_gt][3]  tmpTrans[leftKey][2]
            // parameters_evo[1][0] = gl_tmpTrans[rightKey][0]; //;gt_time_poses[upper_idx_gt][1] tmpTrans[rightKey][0]
            // parameters_evo[1][1] = gl_tmpTrans[rightKey][1]; //;gt_time_poses[upper_idx_gt][2] tmpTrans[rightKey][1]
            // parameters_evo[1][2] = gl_tmpTrans[rightKey][2]; //;gt_time_poses[upper_idx_gt][3] tmpTrans[rightKey][2]

            // ps_function->Evaluate(parameters_evo, residuals_evo);
            // std::vector<double> residuals_evo_values(residuals_evo, residuals_evo);


            /* setup the solver related options */
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // DENSE_QR
            options.num_threads = 12;
            options.max_num_iterations = max_num_iter;
            options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
            options.dogleg_type = ceres::DoglegType::SUBSPACE_DOGLEG;
            options.minimizer_progress_to_stdout = true;
            options.use_nonmonotonic_steps = true;
            cout << "we have done line 3624" << endl;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            /* unify the quaterniond */
            for(int i = start_idx; i < cur_batch_size; i++) {
                if(gl_tmpQuat[i][0] < 0) {
                    Eigen::Quaterniond tmp(gl_tmpQuat[i][0],
                                           gl_tmpQuat[i][1],
                                           gl_tmpQuat[i][2],
                                           gl_tmpQuat[i][3]);
                    tmp = unifyQuaternion(tmp);
                    gl_tmpQuat[i][0] = tmp.w();
                    gl_tmpQuat[i][1] = tmp.x();
                    gl_tmpQuat[i][2] = tmp.y();
                    gl_tmpQuat[i][3] = tmp.z();
                }
            }

            //double to eigen
            for (int i = start_idx; i < cur_batch_size; ++i){
                pose_info_keyframe_batch->points[i].x = gl_tmpTrans[i][0];
                pose_info_keyframe_batch->points[i].y = gl_tmpTrans[i][1];
                pose_info_keyframe_batch->points[i].z = gl_tmpTrans[i][2];
                pose_info_keyframe_batch->points[i].qw = gl_tmpQuat[i][0];
                pose_info_keyframe_batch->points[i].qx = gl_tmpQuat[i][1];
                pose_info_keyframe_batch->points[i].qy = gl_tmpQuat[i][2];
                pose_info_keyframe_batch->points[i].qz = gl_tmpQuat[i][3];

//                for (int j=0; j<9; j++) {
//                    para_speed_bias[i+1][j] = gl_tmpSpeedBias[i][j];
//                }
            }

#if 1 //pub batch trajectory
            if (1) {// (iterCount == iteration_num - 1)
                nav_msgs::Path path_batch;
                path_batch.header.frame_id = frame_id;
                for (int i = keyframe_idx[start_idx]; i < cur_batch_size; ++i) {
                    geometry_msgs::PoseStamped batch_pose_msg;
                    batch_pose_msg.header = path_batch.header;
                    batch_pose_msg.pose.orientation.w = pose_info_keyframe_batch->points[i - 1].qw;
                    batch_pose_msg.pose.orientation.x = pose_info_keyframe_batch->points[i - 1].qx;
                    batch_pose_msg.pose.orientation.y = pose_info_keyframe_batch->points[i - 1].qy;
                    batch_pose_msg.pose.orientation.z = pose_info_keyframe_batch->points[i - 1].qz;
                    batch_pose_msg.pose.position.x = pose_info_keyframe_batch->points[i - 1].x;
                    batch_pose_msg.pose.position.y = pose_info_keyframe_batch->points[i - 1].y;
                    batch_pose_msg.pose.position.z = pose_info_keyframe_batch->points[i - 1].z;
                    path_batch.poses.push_back(batch_pose_msg);
                }
                pub_batch_path.publish(path_batch);

#if 1 /* write final result of pose*/
//                batch_result_path_evo = result_path + std::to_string(iterCount) + "GLIO_batch_enu_q_evo.csv";
//                std::ofstream batch_res_output_evo(batch_result_path_evo, std::ios::out);
//                batch_res_output_evo.close();
//                batch_result_path = result_path + std::to_string(iterCount) + "GLIO_batch_enu.csv";
                batch_result_path = result_path + std::to_string(iterCount) + "tc_batch_result.csv";
                std::ofstream batch_res_output_ws(batch_result_path, std::ios::out);
                batch_res_output_ws.close();

                for (int i = keyframe_idx[0]; i < cur_batch_size; ++i) {
                    /* results for evo */
//                    ofstream fout_batch_evo(batch_result_path_evo, ios::app); //GROUND_TRUTH_PATH_EVO
//                    fout_batch_evo.setf(ios::fixed, ios::floatfield);
//                    fout_batch_evo.precision(8);
//                    fout_batch_evo << keyframe_time[i - 1] << ' ';
//                    fout_batch_evo << pose_info_keyframe_batch->points[i - 1].x << ' '
//                                   << pose_info_keyframe_batch->points[i - 1].y << ' '
//                                   << pose_info_keyframe_batch->points[i - 1].z << ' '
//                                   << pose_info_keyframe_batch->points[i-1].qx << ' '
//                                   << pose_info_keyframe_batch->points[i-1].qy << ' '
//                                   << pose_info_keyframe_batch->points[i-1].qz << ' '
//                                   << pose_info_keyframe_batch->points[i-1].qw << '\n';
//                    fout_batch_evo.close();

                    /* write result to file */
                    Eigen::Vector3d batch_Ps_i = Eigen::Vector3d(pose_info_keyframe_batch->points[i - 1].x,
                                                                 pose_info_keyframe_batch->points[i - 1].y,
                                                                 pose_info_keyframe_batch->points[i - 1].z);
                    enu_pos = batch_Ps_i;
                    Eigen::Matrix3d batch_Rs_i = Eigen::Quaterniond (pose_info_keyframe_batch->points[i-1].qw,
                                                                     pose_info_keyframe_batch->points[i-1].qx,
                                                                     pose_info_keyframe_batch->points[i-1].qy,
                                                                     pose_info_keyframe_batch->points[i-1].qz).normalized().toRotationMatrix();
                    enu_ypr = Utility::R2ypr(batch_Rs_i);

                    ofstream tc_batch_output(batch_result_path, ios::app);
                    tc_batch_output.setf(ios::fixed, ios::floatfield);
                    tc_batch_output.precision(8);
                    tc_batch_output << keyframe_time[i - 1] << ','
                                      << enu_ypr.x() << ','
                                      << enu_ypr.y() << ','
                                      << enu_ypr.z() << ','
                                      << enu_pos[0] << ','
                                      << enu_pos[1] << ','
                                      << enu_pos[2] << '\n';
                    tc_batch_output.close();
                }
//                cout << "---------------- Write traj of iteration "<< iterCount << " for evo" << endl;
#endif
            }
#endif
            
        }

        gl_vec_surf_cur_pts_startend.clear();
        gl_vec_surf_res_cnt_startend.clear();
        gl_vec_surf_scores_startend.clear();
        gl_vec_surf_normals_cents_startend.clear();

//        t_optimize_batch.tic_toc();

        exit(0);
    }

    /* construct batch lidar feature association */
    void batchFeatureAssociation() {
        int idx = keyframe_idx.size() - search_range - 1;
        if (keyframe_idx.size() < 2*search_range || idx < search_range) return;
        int search_idx_start = idx - search_range;

        for (int j = search_idx_start; j <= search_idx_start + 2*search_range; j++) {
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent(new pcl::PointCloud<PointType>());
            gl_vec_surf_cur_pts[idx][j] = tmpSurfCurrent;
            gl_vec_surf_res_cnt[idx][j] = 0;
            vector<double> tmpD;
            gl_vec_surf_scores[idx][j] = tmpD;
//            vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> tmp_pnc;
            vector<vector<double>> tmp_pnc;
            gl_vec_surf_normals_cents[idx][j] = tmp_pnc;
        }
        findGlobalCorrespondingSurfFeaturesAdd_Batch(idx, search_idx_start);
        globalFeatureSelectionAdd_Batch(idx, search_idx_start);

        return;
    }

    /* update the pose */
    void updatePose() {
        abs_pose = abs_poses.back();
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx[keyframe_idx.size()-1]; ++i){
            pose_keyframe->points[i-1].x = abs_poses[i][4];
            pose_keyframe->points[i-1].y = abs_poses[i][5];
            pose_keyframe->points[i-1].z = abs_poses[i][6];

            pose_info_keyframe->points[i-1].x = abs_poses[i][4];
            pose_info_keyframe->points[i-1].y = abs_poses[i][5];
            pose_info_keyframe->points[i-1].z = abs_poses[i][6];
            pose_info_keyframe->points[i-1].qw = abs_poses[i][0];
            pose_info_keyframe->points[i-1].qx = abs_poses[i][1];
            pose_info_keyframe->points[i-1].qy = abs_poses[i][2];
            pose_info_keyframe->points[i-1].qz = abs_poses[i][3];
        }
    }

    void optimizeLocalGraph(vector<double*> paraEach) {
        ceres::Manifold *quatParameterization = new ceres::QuaternionManifold();
        ceres::Problem problem;

        int numPara = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width] - keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1;
        if(numPara==0) return;

        double dQuat[numPara][4];
        double dTrans[numPara][3];

        for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
            i < keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {
            dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0] = pose_each_frame->points[i].x;
            dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1] = pose_each_frame->points[i].y;
            dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2] = pose_each_frame->points[i].z;

            dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0] = pose_info_each_frame->points[i].qw;
            dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1] = pose_info_each_frame->points[i].qx;
            dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2] = pose_info_each_frame->points[i].qy;
            dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][3] = pose_info_each_frame->points[i].qz;

            problem.AddParameterBlock(dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1], 3);
            problem.AddParameterBlock(dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1], 4, quatParameterization);
        }



        ceres::CostFunction *LeftFactor = LidarPoseLeftFactorAutoDiff::Create(Eigen::Quaterniond(paraEach[1][0], paraEach[1][1], paraEach[1][2], paraEach[1][3]),
                Eigen::Vector3d(paraEach[0][0], paraEach[0][1], paraEach[0][2]),
                Eigen::Quaterniond(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].qw,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].qx,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].qy,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].qz),
                Eigen::Vector3d(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].x,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].y,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width - 1].z));
        problem.AddResidualBlock(LeftFactor, NULL, dTrans[0], dQuat[0]);
        for(int i = 0; i < numPara - 1; i++) {
            ceres::CostFunction *Factor = LidarPoseFactorAutoDiff::Create(Eigen::Quaterniond(paraEach[2*i+1][0], paraEach[2*i+1][1], paraEach[2*i+1][2], paraEach[2*i+1][3]),
                    Eigen::Vector3d(paraEach[2*i][0], paraEach[2*i][1], paraEach[2*i][2]));
            problem.AddResidualBlock(Factor, NULL, dTrans[i], dQuat[i], dTrans[i+1], dQuat[i+1]);
        }

        ceres::CostFunction *RightFactor = LidarPoseRightFactorAutoDiff::Create(Eigen::Quaterniond(paraEach.back()[0], paraEach.back()[1], paraEach.back()[2], paraEach.back()[3]),
                Eigen::Vector3d(paraEach[paraEach.size()-2][0], paraEach[paraEach.size()-2][1], paraEach[paraEach.size()-2][2]),
                Eigen::Quaterniond(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].qw,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].qx,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].qy,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].qz),
                Eigen::Vector3d(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].x,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].y,
                pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width].z));
        problem.AddResidualBlock(RightFactor, NULL, dTrans[numPara-1], dQuat[numPara-1]);

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 15;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
            i < keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {
            pose_each_frame->points[i].x = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0];
            pose_each_frame->points[i].y = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1];
            pose_each_frame->points[i].z = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2];

            pose_info_each_frame->points[i].x = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0];
            pose_info_each_frame->points[i].y = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1];
            pose_info_each_frame->points[i].z = dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2];
            pose_info_each_frame->points[i].qw = dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0];
            pose_info_each_frame->points[i].qx = dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1];
            pose_info_each_frame->points[i].qy = dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2];
            pose_info_each_frame->points[i].qz = dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][3];
        }
    }

    void buildLocalMapWithLandMark() {
        // Initialization
        if (pose_keyframe->points.size() < 1) {
            PointPoseInfo Tbl;
            Tbl.qw = q_bl.w();
            Tbl.qx = q_bl.x();
            Tbl.qy = q_bl.y();
            Tbl.qz = q_bl.z();
            Tbl.x = t_bl.x();
            Tbl.y = t_bl.y();
            Tbl.z = t_bl.z();
            //ROS_INFO("Initialization for local map building");
            *surf_local_map += *transformCloud(surf_last, &Tbl);
            return;
        }

        if (recent_surf_keyframes.size() < local_map_width) {
            recent_surf_keyframes.clear();

            for (int i = pose_keyframe->points.size() - 1; i >= 0; --i) {
                if ((int)pose_keyframe->points[i].intensity < 0) continue;
                if (pose_keyframe->points.size() > local_map_width && i <= pose_keyframe->points.size() - local_map_width) break;
                int idx = (int)pose_keyframe->points[i].intensity;

                Eigen::Quaterniond q_po(pose_info_keyframe->points[idx].qw,
                                        pose_info_keyframe->points[idx].qx,
                                        pose_info_keyframe->points[idx].qy,
                                        pose_info_keyframe->points[idx].qz);

                Eigen::Vector3d t_po(pose_info_keyframe->points[idx].x,
                                     pose_info_keyframe->points[idx].y,
                                     pose_info_keyframe->points[idx].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                recent_surf_keyframes.push_front(transformCloud(surf_frames[idx], &Ttmp));

                if (recent_surf_keyframes.size() >= local_map_width)
                    break;
            }
        }
        // If already more then 50 frames, pop the frames at the beginning
        else {
            if (latest_frame_idx != pose_keyframe->points.size() - 1) {
                recent_surf_keyframes.pop_front();
                latest_frame_idx = pose_keyframe->points.size() - 1;

                Eigen::Quaterniond q_po(pose_info_keyframe->points[latest_frame_idx].qw,
                                        pose_info_keyframe->points[latest_frame_idx].qx,
                                        pose_info_keyframe->points[latest_frame_idx].qy,
                                        pose_info_keyframe->points[latest_frame_idx].qz);

                Eigen::Vector3d t_po(pose_info_keyframe->points[latest_frame_idx].x,
                                     pose_info_keyframe->points[latest_frame_idx].y,
                                     pose_info_keyframe->points[latest_frame_idx].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                recent_surf_keyframes.push_back(transformCloud(surf_frames[latest_frame_idx], &Ttmp));

            }
        }

        surf_local_map->points.clear();
        for (int i = 0; i < recent_surf_keyframes.size(); ++i) {
            *surf_local_map += *recent_surf_keyframes[i];
        }
    }

    void downSampleCloud() {

        ds_filter_surf_map.setInputCloud(surf_local_map);
        ds_filter_surf_map.filter(*surf_local_map_ds);

        pcl::PointCloud<PointType>::Ptr fullDS(new pcl::PointCloud<PointType>());
        ds_filter_surf_map.setInputCloud(full_cloud);
        ds_filter_surf_map.filter(*fullDS);
//        full_clouds_ds.push_back(fullDS);

        surf_last_ds->clear();
        ds_filter_surf.setInputCloud(surf_last);
        ds_filter_surf.filter(*surf_last_ds);
    }

    void findCorrespondingSurfFeatures(int idx, Eigen::Quaterniond q, Eigen::Vector3d t) {
//        Timer t_feature_association("findCorrespondingSurfFeatures");

        double nearst_dist = 0; int count_ = 0;
        bool fCSF = false;
        int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;
        vec_surf_res_cnt[idVec] = 0;
        int fail_max_radius = 0;
        int fail_plane_fit = 0;
        int fail_weight = 0;
        for (int i = 0; i < surf_frames[idx]->points.size(); ++i) {
            pt_in_local = surf_frames[idx]->points[i];

            transformPoint(&pt_in_local, &pt_in_map, q, t);
            kd_tree_surf_local_map->nearestKSearch(pt_in_map, 5, pt_search_idx, pt_search_sq_dists);

            Eigen::Matrix<double, 5, 3> matA0 = Eigen::Matrix<double, 5, 3>::Ones();
            Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();
            if (pt_search_sq_dists[4] < kd_max_radius) { // last one lasgest
                nearst_dist += fabs(pt_search_sq_dists[4]);
                count_++;
                for (int j = 0; j < 5; ++j) {
                    matA0(j, 0) = surf_local_map_ds->points[pt_search_idx[j]].x;
                    matA0(j, 1) = surf_local_map_ds->points[pt_search_idx[j]].y;
                    matA0(j, 2) = surf_local_map_ds->points[pt_search_idx[j]].z;
                }

                // Get the norm of the plane using linear solver based on QR composition
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double normInverse = 1 / norm.norm();
                norm.normalize(); // get the unit norm

                // Make sure that the plan is fit
                bool planeValid = true;
                for (int j = 0; j < 5; ++j) {
                    if (fabs(norm.x() * surf_local_map_ds->points[pt_search_idx[j]].x +
                             norm.y() * surf_local_map_ds->points[pt_search_idx[j]].y +
                             norm.z() * surf_local_map_ds->points[pt_search_idx[j]].z + normInverse) > surf_dist_thres) {
                        planeValid = false;
                        break;
                    }
                }

                // if one eigenvalue is significantly larger than the other two
                if (planeValid) {
                    float pd = norm.x() * pt_in_map.x + norm.y() * pt_in_map.y + norm.z() *pt_in_map.z + normInverse;
                    float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(pt_in_map.x * pt_in_map.x + pt_in_map.y * pt_in_map.y + pt_in_map.z * pt_in_map.z));

                    if(weight > 0.3) {
                        PointType normal;
                        normal.x = weight * norm.x();
                        normal.y = weight * norm.y();
                        normal.z = weight * norm.z();
                        normal.intensity = weight * normInverse;

                        vec_surf_cur_pts[idVec]->push_back(pt_in_local);
                        vec_surf_normal[idVec]->push_back(normal);

                        ++vec_surf_res_cnt[idVec];
                        vec_surf_scores[idVec].push_back(lidar_const*weight);
                        fCSF = true;
                    }
                    else {
                        fail_weight++;
                    }
                }
                else {
                    fail_plane_fit++;
                }
            }
            else {
                fail_max_radius++;
            }
        }
//        t_feature_association.tic_toc();
    }

    void findGlobalCorrespondingSurfFeatures_Batch(int idx, int search_idx_start) {
//        Timer t_feature_association("findGlobalCorrespondingSurfFeatures_Batch");
        int idVec = idx;
        pcl::PointCloud<PointType>::Ptr surf_local_cur_frame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_global_cur_frame_map(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_global_oth_frame_map(new pcl::PointCloud<PointType>());
        *surf_global_cur_frame_map = *transformCloud(surf_frames[idx], &pose_info_keyframe->points[idx]);
        *surf_local_cur_frame = *surf_frames[idx];

        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
            if (search_idx == idx) continue;

            pcl::PointCloud<PointType>::Ptr surf_local_search_frame(new pcl::PointCloud<PointType>());
            *surf_local_search_frame = *surf_frames[search_idx];

            *surf_global_oth_frame_map = *transformCloud(surf_local_search_frame, &pose_info_keyframe->points[search_idx]);
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent(new pcl::PointCloud<PointType>());
            gl_vec_surf_cur_pts_startend[idx][search_idx] = tmpSurfCurrent;

            pcl::KdTreeFLANN<PointType>::Ptr kd_tree_surf_local_map_batch;
            kd_tree_surf_local_map_batch.reset(new pcl::KdTreeFLANN<PointType>());
            kd_tree_surf_local_map_batch->setInputCloud(surf_global_oth_frame_map);

            int sfi_size = surf_local_cur_frame->points.size();
            int sgf_map_size = surf_global_cur_frame_map->points.size();

            for (int i = 0; i < surf_global_cur_frame_map->points.size(); i++) {
                int idxx = i;
                PointType pt_in_local_;
                double sf_idx_i_x = surf_local_cur_frame->points[i].x;
                pt_in_local_ = surf_local_cur_frame->points[i];
                PointType pt_in_gl_map_;
                pt_in_gl_map_ = surf_global_cur_frame_map->points[i];
                vector<double> normal_cent(6, 0);
                vector<int> pt_search_idx_batch;
                vector<float> pt_search_sq_dists_batch;
                kd_tree_surf_local_map_batch->nearestKSearch(pt_in_gl_map_, 5, pt_search_idx_batch, pt_search_sq_dists_batch);
                Eigen::Matrix<double, 5, 3> matA0 = Eigen::Matrix<double, 5, 3>::Zero();
                Eigen::Matrix<double, 5, 3> matA0_local = Eigen::Matrix<double, 5, 3>::Zero();
                Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();
                Eigen::Matrix<double, 5, 1> matB0_local = - Eigen::Matrix<double, 5, 1>::Ones();
                if (pt_search_sq_dists_batch[4] < 1.5) { // last one lasgest
                    double cent_x = 0; double cent_y = 0; double cent_z = 0;
                    for (int j = 0; j < 5; ++j) {
                        matA0(j, 0) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].x;
                        matA0(j, 1) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].y;
                        matA0(j, 2) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].z;
                        matA0_local(j, 0) = surf_local_search_frame->points[pt_search_idx_batch[j]].x;
                        matA0_local(j, 1) = surf_local_search_frame->points[pt_search_idx_batch[j]].y;
                        matA0_local(j, 2) = surf_local_search_frame->points[pt_search_idx_batch[j]].z;
                        cent_x += matA0_local(j, 0);
                        cent_y += matA0_local(j, 1);
                        cent_z += matA0_local(j, 2);
                    }
                    normal_cent[3] = cent_x/5.;
                    normal_cent[4] = cent_y/5.;
                    normal_cent[5] = cent_z/5.;
                    // Get the norm of the plane using linear solver based on QR composition
                    Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                    double normInverse = 1 / norm.norm();
                    norm.normalize(); // get the unit norm
                    Eigen::Vector3d norm_local = matA0_local.colPivHouseholderQr().solve(matB0_local);
                    norm_local.normalize(); // get the unit norm
                    // Make sure that the plan is fit
                    bool planeValid = true;
                    for (int j = 0; j < 5; ++j) {
                        if (fabs(norm.x() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].x +
                                 norm.y() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].y +
                                 norm.z() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].z + normInverse) > 0.18) {
                            planeValid = false;
                            break;
                        }
                    }
                    // if one eigenvalue is significantly larger than the other two
                    if (planeValid) {
                        float pd = norm.x() * pt_in_gl_map_.x + norm.y() * pt_in_gl_map_.y + norm.z() *pt_in_gl_map_.z + normInverse;
                        float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(pt_in_gl_map_.x * pt_in_gl_map_.x + pt_in_gl_map_.y * pt_in_gl_map_.y + pt_in_gl_map_.z * pt_in_gl_map_.z));
                        if(weight > 0.3) {
                            PointType normal;
                            normal.x = weight * norm.x();
                            normal.y = weight * norm.y();
                            normal.z = weight * norm.z();
                            normal.intensity = weight * normInverse;
                            normal_cent[0] = norm_local.x();
                            normal_cent[1] = norm_local.y();
                            normal_cent[2] = norm_local.z();
                            gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.push_back(pt_in_local_);
                            ++gl_vec_surf_res_cnt_startend[idVec][search_idx];
                            gl_vec_surf_scores_startend[idVec][search_idx].push_back(2.5*weight);
                            gl_vec_surf_normals_cents_startend[idVec][search_idx].push_back(normal_cent);
                        }
                    }
                }
            }
        }
//        t_feature_association.tic_toc();
    }

    void findGlobalCorrespondingSurfFeaturesAdd_Batch(int idx, int search_idx_start) {
//        Timer t_feature_association("findGlobalCorrespondingSurfFeaturesAdd_Batch");
        int idVec = idx;
        pcl::PointCloud<PointType>::Ptr surf_global_cur_frame_map(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_global_oth_frame_map(new pcl::PointCloud<PointType>());
        *surf_global_cur_frame_map = *transformCloud(surf_frames[idx], &pose_info_keyframe->points[idx]);
        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
            if (search_idx == idx) continue;

            *surf_global_oth_frame_map = *transformCloud(surf_frames[search_idx], &pose_info_keyframe->points[search_idx]);
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent(new pcl::PointCloud<PointType>());
            gl_vec_surf_cur_pts[idVec][search_idx] = tmpSurfCurrent;

            pcl::KdTreeFLANN<PointType>::Ptr kd_tree_surf_local_map_batch;
            kd_tree_surf_local_map_batch.reset(new pcl::KdTreeFLANN<PointType>());
            kd_tree_surf_local_map_batch->setInputCloud(surf_global_oth_frame_map);
            for (int i = 0; i < surf_global_cur_frame_map->points.size(); ++i) {
                PointType pt_in_local_;
                pt_in_local_ = surf_frames[idx]->points[i];
                PointType pt_in_gl_map_;
                pt_in_gl_map_ = surf_global_cur_frame_map->points[i];
                vector<double> normal_cent(6, 0);
                vector<int> pt_search_idx_batch;
                vector<float> pt_search_sq_dists_batch;
                kd_tree_surf_local_map_batch->nearestKSearch(pt_in_gl_map_, 5, pt_search_idx_batch, pt_search_sq_dists_batch);
                Eigen::Matrix<double, 5, 3> matA0 = Eigen::Matrix<double, 5, 3>::Zero();
                Eigen::Matrix<double, 5, 3> matA0_local = Eigen::Matrix<double, 5, 3>::Zero();
                Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();
                Eigen::Matrix<double, 5, 1> matB0_local = - Eigen::Matrix<double, 5, 1>::Ones();
                if (pt_search_sq_dists_batch[4] < 1.5) { // last one lasgest
                    double cent_x = 0; double cent_y = 0; double cent_z = 0;
                    for (int j = 0; j < 5; ++j) {
                        matA0(j, 0) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].x;
                        matA0(j, 1) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].y;
                        matA0(j, 2) = surf_global_oth_frame_map->points[pt_search_idx_batch[j]].z;
                        matA0_local(j, 0) = surf_frames[search_idx]->points[pt_search_idx_batch[j]].x;
                        matA0_local(j, 1) = surf_frames[search_idx]->points[pt_search_idx_batch[j]].y;
                        matA0_local(j, 2) = surf_frames[search_idx]->points[pt_search_idx_batch[j]].z;
                        cent_x += matA0_local(j, 0);
                        cent_y += matA0_local(j, 1);
                        cent_z += matA0_local(j, 2);
                    }
                    normal_cent[3] = cent_x/5.;
                    normal_cent[4] = cent_y/5.;
                    normal_cent[5] = cent_z/5.;
                    // Get the norm of the plane using linear solver based on QR composition
                    Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                    double normInverse = 1 / norm.norm();
                    norm.normalize(); // get the unit norm
                    Eigen::Vector3d norm_local = matA0_local.colPivHouseholderQr().solve(matB0_local);
                    norm_local.normalize(); // get the unit norm
                    // Make sure that the plan is fit
                    bool planeValid = true;
                    for (int j = 0; j < 5; ++j) {
                        if (fabs(norm.x() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].x +
                                 norm.y() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].y +
                                 norm.z() * surf_global_oth_frame_map->points[pt_search_idx_batch[j]].z + normInverse) > 0.18) {
                            planeValid = false;
                            break;
                        }
                    }
                    // if one eigenvalue is significantly larger than the other two
                    if (planeValid) {
                        float pd = norm.x() * pt_in_gl_map_.x + norm.y() * pt_in_gl_map_.y + norm.z() *pt_in_gl_map_.z + normInverse;
                        float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(pt_in_gl_map_.x * pt_in_gl_map_.x + pt_in_gl_map_.y * pt_in_gl_map_.y + pt_in_gl_map_.z * pt_in_gl_map_.z));
                        if(weight > 0.3) {
                            PointType normal;
                            normal.x = weight * norm.x();
                            normal.y = weight * norm.y();
                            normal.z = weight * norm.z();
                            normal.intensity = weight * normInverse;
                            normal_cent[0] = norm_local.x();
                            normal_cent[1] = norm_local.y();
                            normal_cent[2] = norm_local.z();
                            gl_vec_surf_cur_pts[idVec][search_idx]->points.push_back(pt_in_local_);
                            ++gl_vec_surf_res_cnt[idVec][search_idx];
                            gl_vec_surf_scores[idVec][search_idx].push_back(2.5*weight);
                            gl_vec_surf_normals_cents[idVec][search_idx].push_back(normal_cent);
                        }
                    }
                }
            }
        }
//        t_feature_association.tic_toc();
    }

    void featureSelection (int idx, Eigen::Quaterniond q, Eigen::Vector3d t) {
//        Timer t_feature_select("FeatureSelector");

        int idVec = idx-keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;

        int surf_pts_size = vec_surf_cur_pts[idVec]->points.size();
        vec_surf_cur_pts[idVec]->resize(surf_pts_size);
        vec_surf_normal[idVec]->resize(surf_pts_size);
        if (surf_pts_size < 1) return;
        int org_rand_set_num = rand_set_num;
        int org_feature_res_num = feature_res_num;

        if (surf_pts_size - 1 < feature_res_num) {
            return;
            feature_res_num = surf_pts_size - 1;
            rand_set_num = surf_pts_size - 1;
        }
        if (surf_pts_size - 1 < rand_set_num) {
            rand_set_num = surf_pts_size - 1;
        }
        if (surf_pts_size - feature_res_num < rand_set_num) {
            rand_set_num = surf_pts_size - feature_res_num - 1;
        }
        /* init feature selection para */
        double sum_LogDeterminant = 0;
        Eigen::Matrix<double, 6, 6> JTJ_selected_feature_sum = Eigen::Matrix<double, 6, 6>::Zero();

        /* init temp variables */                       
        pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp_less (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp_less (new pcl::PointCloud<PointType>());
        vector<double> surf_score_temp;
        int surf_res_cnt_temp = 0;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians_set;
        std::vector<double *> parameter_blocks_;

        /* init ceres problem parameter block */
        double Trans_[3];
        Trans_[0] = t[0];
        Trans_[1] = t[1];
        Trans_[2] = t[2];

        double Euler_[3];
        Eigen::Vector3d euler_2 = Eigen::Vector3d(0, 0, 0);
        toEulerAngle(q, euler_2);
        Euler_[0] = euler_2[0]; Euler_[1] = euler_2[1]; Euler_[2] = euler_2[2];

        /* random numbers generator */
        common::RandomGeneratorInt<int> rgi_;

        while (surf_cur_pt_temp->points.size() < feature_res_num && random_select) {
            int *rand_ids;
            rand_ids = rgi_.geneRandArrayNoRepeat(0, vec_surf_cur_pts[idVec]->points.size() - 1, rand_set_num);

            /* Search best feature points in random set*/
            double temp_res_LogDeterminant = -1;
            int selected_id = -1;

            for (int i=0; i<rand_set_num; i++) {
                int id = rand_ids[i];
                selected_id = id;
            }
            delete[]rand_ids;

            surf_cur_pt_temp->points.push_back(vec_surf_cur_pts[idVec]->points[selected_id]);
            surf_cur_normal_temp->points.push_back(vec_surf_normal[idVec]->points[selected_id]);
            surf_score_temp.push_back(vec_surf_scores[idVec][selected_id]);
            surf_res_cnt_temp++;

            pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
            inliers->indices.push_back(selected_id);

            pcl::ExtractIndices<PointType> extract1;
            extract1.setInputCloud(vec_surf_cur_pts[idVec]);
            extract1.setIndices(inliers);
            extract1.setNegative(true);
            extract1.filter(*vec_surf_cur_pts[idVec]);

            extract1.setInputCloud(vec_surf_normal[idVec]);
            extract1.setIndices(inliers);
            extract1.setNegative(true);
            extract1.filter(*vec_surf_normal[idVec]);

            vec_surf_scores[idVec].erase(vec_surf_scores[idVec].begin() + selected_id);
        }

        vec_surf_cur_pts[idVec]->clear();
        vec_surf_cur_pts[idVec] = surf_cur_pt_temp;
        vec_surf_normal[idVec]->clear();
        vec_surf_normal[idVec] = surf_cur_normal_temp;
        vec_surf_scores[idVec].clear();
        vec_surf_scores[idVec] = surf_score_temp;
        vec_surf_res_cnt[idVec] = surf_res_cnt_temp;

        feature_res_num = org_feature_res_num;
        rand_set_num = org_rand_set_num;
//        t_feature_select.tic_toc();
    }

    void globalFeatureSelection_Batch (int idx, int search_idx_start) {
//        Timer t_feature_select("globalFeatureSelection_Batch");

        int idVec = idx;

        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
            if (search_idx == idx) continue;
            pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp_less (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp_less (new pcl::PointCloud<PointType>());
            vector<double> surf_score_temp;
            int surf_res_cnt_temp = 0;
            vector<vector<double>> surf_normals_cents_temp;

            /* random numbers generator */
            common::RandomGeneratorInt<int> rgi_;

            int org_batch_feature_res_num = batch_feature_res_num;
            int org_batch_rand_set_num = batch_rand_set_num;

            if (gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - 1 < batch_feature_res_num ||
                    gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() < 50) {
                return;
            }
            if (gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - 1 < batch_rand_set_num) {
                batch_rand_set_num = gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - 1;
            }
            if (gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - batch_feature_res_num < batch_rand_set_num) {
                batch_rand_set_num = gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - batch_feature_res_num - 1;
            }
            while (surf_cur_pt_temp->points.size() < batch_feature_res_num) {
                int *rand_ids;
                rand_ids = rgi_.geneRandArrayNoRepeat(0, gl_vec_surf_cur_pts_startend[idVec][search_idx]->points.size() - 1, batch_rand_set_num);

                /* Search best feature points in random set*/
                int selected_id = -1;

                for (int i=0; i<batch_feature_res_num; i++) {
                    int id = rand_ids[i];
                    selected_id = id;
                    surf_cur_pt_temp->points.push_back(gl_vec_surf_cur_pts_startend[idVec][search_idx]->points[selected_id]);
                    surf_score_temp.push_back(gl_vec_surf_scores_startend[idVec][search_idx][selected_id]);
                    surf_res_cnt_temp++;
                    surf_normals_cents_temp.push_back(gl_vec_surf_normals_cents_startend[idVec][search_idx][selected_id]);
                }
                gl_vec_surf_cur_pts_startend[idVec][search_idx]->clear();
                gl_vec_surf_cur_pts_startend[idVec][search_idx] = surf_cur_pt_temp;
                gl_vec_surf_scores_startend[idVec][search_idx].clear();
                gl_vec_surf_scores_startend[idVec][search_idx] = surf_score_temp;
                gl_vec_surf_res_cnt_startend[idVec][search_idx] = surf_res_cnt_temp;
                gl_vec_surf_normals_cents_startend[idVec][search_idx].clear();
                gl_vec_surf_normals_cents_startend[idVec][search_idx] = surf_normals_cents_temp;
                delete[] rand_ids;
            }
            batch_feature_res_num = org_batch_feature_res_num;
            batch_rand_set_num = org_batch_rand_set_num;
        }

//        t_feature_select.tic_toc();

    }

    void globalFeatureSelectionAdd_Batch (int idx, int search_idx_start) {
//        Timer t_feature_select("globalFeatureSelectionAdd_Batch");

        int idVec = idx;

        for (int search_idx = search_idx_start; search_idx <= search_idx_start + 2*search_range; search_idx++) {
            if (search_idx == idx) continue;
            pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_pt_temp_less (new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr surf_cur_normal_temp_less (new pcl::PointCloud<PointType>());
            vector<double> surf_score_temp;
            int surf_res_cnt_temp = 0;
            vector<vector<double>> surf_normals_cents_temp;

            /* random numbers generator */
            common::RandomGeneratorInt<int> rgi_;

            int org_batch_feature_res_num = batch_feature_res_num;
            int org_batch_rand_set_num = batch_rand_set_num;

            if (gl_vec_surf_cur_pts[idVec][search_idx]->points.size() <= batch_feature_res_num) {
                continue;
            }
            while (surf_cur_pt_temp->points.size() < batch_feature_res_num) {
                int *rand_ids;
                if (gl_vec_surf_cur_pts[idVec][search_idx]->points.size() - 1 <= batch_feature_res_num)
                    cout << "rand set larger than org set " << endl;
                rand_ids = rgi_.geneRandArrayNoRepeat(0, gl_vec_surf_cur_pts[idVec][search_idx]->points.size() - 1, batch_feature_res_num);


                /* Search best feature points in random set*/
                int selected_id = -1;

                for (int i=0; i<batch_feature_res_num; i++) {
                    int id = rand_ids[i];
                    selected_id = id;
                    surf_cur_pt_temp->points.push_back(gl_vec_surf_cur_pts[idVec][search_idx]->points[selected_id]);
                    surf_score_temp.push_back(gl_vec_surf_scores[idVec][search_idx][selected_id]);
                    surf_res_cnt_temp++;
                    surf_normals_cents_temp.push_back(gl_vec_surf_normals_cents[idVec][search_idx][selected_id]);
                }

                gl_vec_surf_cur_pts[idVec][search_idx]->clear();
                gl_vec_surf_cur_pts[idVec][search_idx] = surf_cur_pt_temp;
                gl_vec_surf_scores[idVec][search_idx].clear();
                gl_vec_surf_scores[idVec][search_idx] = surf_score_temp;
                gl_vec_surf_res_cnt[idVec][search_idx] = surf_res_cnt_temp;
                gl_vec_surf_normals_cents[idVec][search_idx].clear();
                gl_vec_surf_normals_cents[idVec][search_idx] = surf_normals_cents_temp;

                delete[] rand_ids;
            }
            batch_feature_res_num = org_batch_feature_res_num;
            batch_rand_set_num = org_batch_rand_set_num;
        }

//        t_feature_select.tic_toc();

    }

    static void toEulerAngle(const Quaterniond& q, Eigen::Vector3d &euler)
    {
        // roll (x-axis rotation)
        double sinr_cosp = +2.0 * (q.w() * q.x() + q.y() * q.z());
        double cosr_cosp = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        euler[0] = atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
        if (fabs(sinp) >= 1)
        euler[1] = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        else
        euler[1] = asin(sinp);

        // yaw (z-axis rotation)
        double siny_cosp = +2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        euler[2] = atan2(siny_cosp, cosy_cosp);
    }

    void saveKeyFramesAndFactors() {
//        Timer t_sFF("saveKeyFramesAndFactors");
        abs_poses.push_back(abs_pose);
        keyframe_id_in_frame.push_back(each_odom_buf.size()-1); // each_odom_buf max is 50, for example, 1, 5, 10, 15, ..

        pcl::PointCloud<PointType>::Ptr surfEachFrame(new pcl::PointCloud<PointType>());

        *surfEachFrame = *surf_last_ds;
        surf_frames.push_back(surfEachFrame);

        //record index of kayframe on imu preintegration poses
        keyframe_idx.push_back(abs_poses.size()-1);

        keyframe_time.push_back(odom_cur->header.stamp.toSec()); // 3Hz roughly

        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

        int i = idx_imu;
        Eigen::Quaterniond tmpOrient;
        double timeodom_cur = odom_cur->header.stamp.toSec();
        if(imu_buf[i]->header.stamp.toSec() > timeodom_cur)
            ROS_WARN("Timestamp not synchronized, please check your hardware!");
        while(imu_buf[i]->header.stamp.toSec() < timeodom_cur) {
            double t = imu_buf[i]->header.stamp.toSec();
            if (cur_time_imu < 0)
                cur_time_imu = t;
            double dt = t - cur_time_imu;
            cur_time_imu = imu_buf[i]->header.stamp.toSec();
            dx = imu_buf[i]->linear_acceleration.x;
            dy = imu_buf[i]->linear_acceleration.y;
            dz = imu_buf[i]->linear_acceleration.z;
            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            rx = imu_buf[i]->angular_velocity.x;
            ry = imu_buf[i]->angular_velocity.y;
            rz = imu_buf[i]->angular_velocity.z;

            tmpOrient = Eigen::Quaterniond(imu_buf[i]->orientation.w,
                                           imu_buf[i]->orientation.x,
                                           imu_buf[i]->orientation.y,
                                           imu_buf[i]->orientation.z);
            processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
            i++;
            if(i >= imu_buf.size())
                break;
        }
        imu_idx_in_kf.push_back(i - 1);

        if(i < imu_buf.size()) {
            double dt1 = timeodom_cur - cur_time_imu;
            double dt2 = imu_buf[i]->header.stamp.toSec() - timeodom_cur;

            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);

            Eigen::Quaterniond orient1 = Eigen::Quaterniond(imu_buf[i]->orientation.w,
                                                            imu_buf[i]->orientation.x,
                                                            imu_buf[i]->orientation.y,
                                                            imu_buf[i]->orientation.z);
            tmpOrient = tmpOrient.slerp(w2, orient1);

            dx = w1 * dx + w2 * imu_buf[i]->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_buf[i]->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_buf[i]->linear_acceleration.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            rx = w1 * rx + w2 * imu_buf[i]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[i]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[i]->angular_velocity.z;
            processIMU(dt1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
        }
        cur_time_imu = timeodom_cur;
        vector<double> tmpSpeedBias;
        tmpSpeedBias.push_back(Vs.back().x());
        tmpSpeedBias.push_back(Vs.back().y());
        tmpSpeedBias.push_back(Vs.back().z());
        tmpSpeedBias.push_back(Bas.back().x());
        tmpSpeedBias.push_back(Bas.back().y());
        tmpSpeedBias.push_back(Bas.back().z());
        tmpSpeedBias.push_back(Bgs.back().x());
        tmpSpeedBias.push_back(Bgs.back().y());
        tmpSpeedBias.push_back(Bgs.back().z());
        para_speed_bias.push_back(tmpSpeedBias);
        idx_imu = i;

        PointXYZI latestPose;
        PointPoseInfo latestPoseInfo;
        latestPose.x = Ps.back().x();
        latestPose.y = Ps.back().y();
        latestPose.z = Ps.back().z();
        latestPose.intensity = pose_keyframe->points.size();
        pose_keyframe->push_back(latestPose);

        latestPoseInfo.x = Ps.back().x();
        latestPoseInfo.y = Ps.back().y();
        latestPoseInfo.z = Ps.back().z();
        Eigen::Quaterniond qs_last(Rs.back());
        latestPoseInfo.qw = qs_last.w();
        latestPoseInfo.qx = qs_last.x();
        latestPoseInfo.qy = qs_last.y();
        latestPoseInfo.qz = qs_last.z();
        latestPoseInfo.idx = pose_keyframe->points.size();
        latestPoseInfo.time = time_new_odom;

        pose_info_keyframe->push_back(latestPoseInfo);

        //optimize sliding window
        num_kf_sliding++;
        if(num_kf_sliding >= 1 || !first_opt) {
            optimizeSlidingWindowWithLandMark();
            num_kf_sliding = 0;
        }

        /* optimize local factor graph */
        if (pose_keyframe->points.size() == slide_window_width) {
            pose_each_frame->push_back(pose_keyframe->points[0]);
            pose_info_each_frame->push_back(pose_info_keyframe->points[0]);
        }
        else if(pose_keyframe->points.size() > slide_window_width) {
            int ii = imu_idx_in_kf[imu_idx_in_kf.size() - slide_window_width - 1];
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            Eigen::Vector3d Ptmp = Ps[Ps.size() - slide_window_width];
            Eigen::Vector3d Vtmp = Vs[Ps.size() - slide_window_width];
            Eigen::Matrix3d Rtmp = Rs[Ps.size() - slide_window_width];
            Eigen::Vector3d Batmp = Eigen::Vector3d::Zero();
            Eigen::Vector3d Bgtmp = Eigen::Vector3d::Zero();

            for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
                i < keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {

                double dt1 = each_odom_buf[i-1]->header.stamp.toSec() - imu_buf[ii]->header.stamp.toSec();
                double dt2 = imu_buf[ii+1]->header.stamp.toSec() - each_odom_buf[i-1]->header.stamp.toSec();

                double w1 = dt2 / (dt1 + dt2);
                double w2 = dt1 / (dt1 + dt2);
                dx = w1 * imu_buf[ii]->linear_acceleration.x + w2 * imu_buf[ii+1]->linear_acceleration.x;
                dy = w1 * imu_buf[ii]->linear_acceleration.y + w2 * imu_buf[ii+1]->linear_acceleration.y;
                dz = w1 * imu_buf[ii]->linear_acceleration.z + w2 * imu_buf[ii+1]->linear_acceleration.z;

                rx = w1 * imu_buf[ii]->angular_velocity.x + w2 * imu_buf[ii+1]->angular_velocity.x;
                ry = w1 * imu_buf[ii]->angular_velocity.y + w2 * imu_buf[ii+1]->angular_velocity.y;
                rz = w1 * imu_buf[ii]->angular_velocity.z + w2 * imu_buf[ii+1]->angular_velocity.z;
                Eigen::Vector3d a0(dx, dy, dz);
                Eigen::Vector3d gy0(rx, ry, rz);
                ii++;
                double integStartTime = each_odom_buf[i-1]->header.stamp.toSec();

                while(imu_buf[ii]->header.stamp.toSec() < each_odom_buf[i]->header.stamp.toSec()) {
                    double t = imu_buf[ii]->header.stamp.toSec();
                    double dt = t - integStartTime;
                    integStartTime = imu_buf[ii]->header.stamp.toSec();
                    dx = imu_buf[ii]->linear_acceleration.x;
                    dy = imu_buf[ii]->linear_acceleration.y;
                    dz = imu_buf[ii]->linear_acceleration.z;

                    rx = imu_buf[ii]->angular_velocity.x;
                    ry = imu_buf[ii]->angular_velocity.y;
                    rz = imu_buf[ii]->angular_velocity.z;

                    if(dx > 15.0) dx = 15.0;
                    if(dy > 15.0) dy = 15.0;
                    if(dz > 18.0) dz = 18.0;

                    if(dx < -15.0) dx = -15.0;
                    if(dy < -15.0) dy = -15.0;
                    if(dz < -18.0) dz = -18.0;

                    Eigen::Vector3d a1(dx, dy, dz);
                    Eigen::Vector3d gy1(rx, ry, rz);

                    Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                    Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                    Rtmp *= deltaQ(un_gyr * dt).toRotationMatrix();
                    Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                    Ptmp += dt * Vtmp + 0.5 * dt * dt * un_acc;
                    Vtmp += dt * un_acc;

                    a0 = a1;
                    gy0 = gy1;

                    ii++;
                }

                dt1 = each_odom_buf[i]->header.stamp.toSec() - imu_buf[ii-1]->header.stamp.toSec();
                dt2 = imu_buf[ii]->header.stamp.toSec() - each_odom_buf[i]->header.stamp.toSec();
                w1 = dt2 / (dt1 + dt2);
                w2 = dt1 / (dt1 + dt2);
                dx = w1 * dx + w2 * imu_buf[ii]->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_buf[ii]->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_buf[ii]->linear_acceleration.z;

                rx = w1 * rx + w2 * imu_buf[ii]->angular_velocity.x;
                ry = w1 * ry + w2 * imu_buf[ii]->angular_velocity.y;
                rz = w1 * rz + w2 * imu_buf[ii]->angular_velocity.z;

                if(dx > 15.0) dx = 15.0;
                if(dy > 15.0) dy = 15.0;
                if(dz > 18.0) dz = 18.0;

                if(dx < -15.0) dx = -15.0;
                if(dy < -15.0) dy = -15.0;
                if(dz < -18.0) dz = -18.0;

                Eigen::Vector3d a1(dx, dy, dz);
                Eigen::Vector3d gy1(rx, ry, rz);

                Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                Rtmp *= deltaQ(un_gyr * dt1).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ptmp += dt1 * Vtmp + 0.5 * dt1 * dt1 * un_acc;
                Vtmp += dt1 * un_acc;

                ii--;

                Eigen::Quaterniond qqq(Rtmp);

                PointXYZI latestPose;
                PointPoseInfo latestPoseInfo;
                latestPose.x = Ptmp.x();
                latestPose.y = Ptmp.y();
                latestPose.z = Ptmp.z();
                pose_each_frame->push_back(latestPose);

                latestPoseInfo.x = Ptmp.x();
                latestPoseInfo.y = Ptmp.y();
                latestPoseInfo.z = Ptmp.z();
                latestPoseInfo.qw = qqq.w();
                latestPoseInfo.qx = qqq.x();
                latestPoseInfo.qy = qqq.y();
                latestPoseInfo.qz = qqq.z();
                latestPoseInfo.time = each_odom_buf[i]->header.stamp.toSec();
                pose_info_each_frame->push_back(latestPoseInfo);
            }

            pose_each_frame->push_back(pose_keyframe->points[pose_keyframe->points.size() - slide_window_width]);
            pose_info_each_frame->push_back(pose_info_keyframe->points[pose_keyframe->points.size() - slide_window_width]);
            int j = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width];

            double dt1 = each_odom_buf[j-1]->header.stamp.toSec() - imu_buf[ii]->header.stamp.toSec();
            double dt2 = imu_buf[ii+1]->header.stamp.toSec() - each_odom_buf[j-1]->header.stamp.toSec();
            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);
            dx = w1 * imu_buf[ii]->linear_acceleration.x + w2 * imu_buf[ii+1]->linear_acceleration.x;
            dy = w1 * imu_buf[ii]->linear_acceleration.y + w2 * imu_buf[ii+1]->linear_acceleration.y;
            dz = w1 * imu_buf[ii]->linear_acceleration.z + w2 * imu_buf[ii+1]->linear_acceleration.z;

            rx = w1 * imu_buf[ii]->angular_velocity.x + w2 * imu_buf[ii+1]->angular_velocity.x;
            ry = w1 * imu_buf[ii]->angular_velocity.y + w2 * imu_buf[ii+1]->angular_velocity.y;
            rz = w1 * imu_buf[ii]->angular_velocity.z + w2 * imu_buf[ii+1]->angular_velocity.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            Eigen::Vector3d a0(dx, dy, dz);
            Eigen::Vector3d gy0(rx, ry, rz);
            ii++;
            double integStartTime = each_odom_buf[j-1]->header.stamp.toSec();

            while(imu_buf[ii]->header.stamp.toSec() < each_odom_buf[j]->header.stamp.toSec()) {
                double t = imu_buf[ii]->header.stamp.toSec();
                double dt = t - integStartTime;
                integStartTime = imu_buf[ii]->header.stamp.toSec();
                dx = imu_buf[ii]->linear_acceleration.x;
                dy = imu_buf[ii]->linear_acceleration.y;
                dz = imu_buf[ii]->linear_acceleration.z;

                rx = imu_buf[ii]->angular_velocity.x;
                ry = imu_buf[ii]->angular_velocity.y;
                rz = imu_buf[ii]->angular_velocity.z;

                if(dx > 15.0) dx = 15.0;
                if(dy > 15.0) dy = 15.0;
                if(dz > 18.0) dz = 18.0;

                if(dx < -15.0) dx = -15.0;
                if(dy < -15.0) dy = -15.0;
                if(dz < -18.0) dz = -18.0;

                Eigen::Vector3d a1(dx, dy, dz);
                Eigen::Vector3d gy1(rx, ry, rz);

                Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                Rtmp *= deltaQ(un_gyr * dt).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ptmp += dt * Vtmp + 0.5 * dt * dt * un_acc;
                Vtmp += dt * un_acc;

                a0 = a1;
                gy0 = gy1;

                ii++;
            }

            dt1 = each_odom_buf[j]->header.stamp.toSec() - imu_buf[ii-1]->header.stamp.toSec();
            dt2 = imu_buf[ii]->header.stamp.toSec() - each_odom_buf[j]->header.stamp.toSec();
            w1 = dt2 / (dt1 + dt2);
            w2 = dt1 / (dt1 + dt2);
            dx = w1 * dx + w2 * imu_buf[ii]->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_buf[ii]->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_buf[ii]->linear_acceleration.z;

            rx = w1 * rx + w2 * imu_buf[ii]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[ii]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[ii]->angular_velocity.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            Eigen::Vector3d a1(dx, dy, dz);
            Eigen::Vector3d gy1(rx, ry, rz);

            Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
            Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
            Rtmp *= deltaQ(un_gyr * dt1).toRotationMatrix();
            Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
            Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            Ptmp += dt1 * Vtmp + 0.5 * dt1 * dt1 * un_acc;
            Vtmp += dt1 * un_acc;

            vector<double*> paraBetweenEachFrame;
            int numPara = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width] - keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1];
            double dQuat[numPara][4];
            double dTrans[numPara][3];
            for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
                i < keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {
                Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_each_frame->points[i].x,
                                                           pose_each_frame->points[i].y,
                                                           pose_each_frame->points[i].z) -
                        Eigen::Vector3d(pose_each_frame->points[i-1].x,
                        pose_each_frame->points[i-1].y,
                        pose_each_frame->points[i-1].z);
                tmpTrans = Eigen::Quaterniond(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz).inverse() * tmpTrans;
                dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0] = tmpTrans.x();
                dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1] = tmpTrans.y();
                dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2] = tmpTrans.z();
                paraBetweenEachFrame.push_back(dTrans[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1]);

                Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz).inverse() *
                        Eigen::Quaterniond(pose_info_each_frame->points[i].qw,
                                           pose_info_each_frame->points[i].qx,
                                           pose_info_each_frame->points[i].qy,
                                           pose_info_each_frame->points[i].qz);
                dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][0] = tmpQuat.w();
                dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][1] = tmpQuat.x();
                dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][2] = tmpQuat.y();
                dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1][3] = tmpQuat.z();
                paraBetweenEachFrame.push_back(dQuat[i-keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1]-1]);
//                Eigen::Vector3d euler;
//                toEulerAngle(Eigen::Quaterniond(tmpQuat.w(), tmpQuat.x(), tmpQuat.y(), tmpQuat.z()), euler);
            }
            int jj = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width];

            Eigen::Vector3d tmpTrans = Ptmp - Eigen::Vector3d(pose_each_frame->points[jj-1].x,
                    pose_each_frame->points[jj-1].y,
                    pose_each_frame->points[jj-1].z);
            tmpTrans = Eigen::Quaterniond(pose_info_each_frame->points[jj-1].qw,
                    pose_info_each_frame->points[jj-1].qx,
                    pose_info_each_frame->points[jj-1].qy,
                    pose_info_each_frame->points[jj-1].qz).inverse() * tmpTrans;

            dTrans[numPara-1][0] = tmpTrans.x();
            dTrans[numPara-1][1] = tmpTrans.y();
            dTrans[numPara-1][2] = tmpTrans.z();
            paraBetweenEachFrame.push_back(dTrans[numPara-1]);

            Eigen::Quaterniond qtmp(Rtmp);
            Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_each_frame->points[jj-1].qw,
                    pose_info_each_frame->points[jj-1].qx,
                    pose_info_each_frame->points[jj-1].qy,
                    pose_info_each_frame->points[jj-1].qz).inverse() * qtmp;
            dQuat[numPara-1][0] = tmpQuat.w();
            dQuat[numPara-1][1] = tmpQuat.x();
            dQuat[numPara-1][2] = tmpQuat.y();
            dQuat[numPara-1][3] = tmpQuat.z();
            paraBetweenEachFrame.push_back(dQuat[numPara-1]);

            optimizeLocalGraph(paraBetweenEachFrame);
        }

        if (time_new_odom > 1761024069)
        // 1732337783 for 1123_exp_1
        // 1732339403 for 1123_exp_2
        // 1723206610 for fj_data
        // 1761024069 for 1021_exp_1
        {
            optimizeBatchWithLandMark();
        }

        if (!loop_closure_on)
            return;

        //add poses to global graph
        if (pose_keyframe->points.size() == slide_window_width) {
            gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(pose_info_each_frame->points[0].qw,
                                                           pose_info_each_frame->points[0].qx,
                                                           pose_info_each_frame->points[0].qy,
                                                           pose_info_each_frame->points[0].qz);
            gtsam::Point3 transition = gtsam::Point3(pose_each_frame->points[0].x,
                                                     pose_each_frame->points[0].y,
                                                     pose_each_frame->points[0].z);

            // Initialization for global pose graph
            global_pose_graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(rotation, transition), prior_noise));
            global_init_estimate.insert(0, gtsam::Pose3(rotation, transition));

            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size()-slide_window_width][i];
            }
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];
        }

            /* insert all the dense regular frames between two keyframes */
        else if(pose_keyframe->points.size() > slide_window_width) {
            for(int i = keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width - 1] + 1;
                i <= keyframe_id_in_frame[pose_keyframe->points.size() - slide_window_width]; i++) {

                gtsam::Rot3 rotationLast = gtsam::Rot3::Quaternion(pose_info_each_frame->points[i-1].qw,
                                                                   pose_info_each_frame->points[i-1].qx,
                                                                   pose_info_each_frame->points[i-1].qy,
                                                                   pose_info_each_frame->points[i-1].qz);
                gtsam::Point3 transitionLast = gtsam::Point3(pose_each_frame->points[i-1].x,
                                                             pose_each_frame->points[i-1].y,
                                                             pose_each_frame->points[i-1].z);

                gtsam::Rot3 rotationCur = gtsam::Rot3::Quaternion(pose_info_each_frame->points[i].qw,
                                                                  pose_info_each_frame->points[i].qx,
                                                                  pose_info_each_frame->points[i].qy,
                                                                  pose_info_each_frame->points[i].qz);
                gtsam::Point3 transitionCur = gtsam::Point3(pose_each_frame->points[i].x,
                                                            pose_each_frame->points[i].y,
                                                            pose_each_frame->points[i].z);
                gtsam::Pose3 poseFrom = gtsam::Pose3(rotationLast, transitionLast);
                gtsam::Pose3 poseTo = gtsam::Pose3(rotationCur, transitionCur);

                global_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1,
                                                                         i,
                                                                         poseFrom.between(poseTo),
                                                                         odom_noise));
                global_init_estimate.insert(i, poseTo);
            }
        }

        isam->update(global_pose_graph, global_init_estimate);
        isam->update();

        global_pose_graph.resize(0);
        global_init_estimate.clear();

        if (pose_keyframe->points.size() > slide_window_width) {
            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size()-slide_window_width][i];
            }
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];
        }
//        t_sFF.tic_toc();

    }

    /* update the globally optimized pose estimation */
    void correctPoses() {
        if (loop_closed == true) {
            recent_surf_keyframes.clear();

            int numPoses = global_estimated.size();

            vector<Eigen::Quaterniond> quaternionRel;
            vector<Eigen::Vector3d> transitionRel;

            for(int i = abs_poses.size() - slide_window_width; i < abs_poses.size() - 1; i++) {
                Eigen::Quaterniond quaternionFrom(abs_poses[i][0],
                                                  abs_poses[i][1],
                                                  abs_poses[i][2],
                                                  abs_poses[i][3]);
                Eigen::Vector3d transitionFrom(abs_poses[i][4],
                                               abs_poses[i][5],
                                               abs_poses[i][6]);

                Eigen::Quaterniond quaternionTo(abs_poses[i+1][0],
                                                abs_poses[i+1][1],
                                                abs_poses[i+1][2],
                                                abs_poses[i+1][3]);
                Eigen::Vector3d transitionTo(abs_poses[i+1][4],
                                             abs_poses[i+1][5],
                                             abs_poses[i+1][6]);

                quaternionRel.push_back(quaternionFrom.inverse() * quaternionTo);
                transitionRel.push_back(quaternionFrom.inverse() * (transitionTo - transitionFrom));
            }

            for (int i = 0; i < numPoses; ++i) {
                pose_each_frame->points[i].x = global_estimated.at<gtsam::Pose3>(i).translation().x();
                pose_each_frame->points[i].y = global_estimated.at<gtsam::Pose3>(i).translation().y();
                pose_each_frame->points[i].z = global_estimated.at<gtsam::Pose3>(i).translation().z();

                pose_info_each_frame->points[i].x = pose_each_frame->points[i].x;
                pose_info_each_frame->points[i].y = pose_each_frame->points[i].y;
                pose_info_each_frame->points[i].z = pose_each_frame->points[i].z;
                pose_info_each_frame->points[i].qw = global_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().w();
                pose_info_each_frame->points[i].qx = global_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().x();
                pose_info_each_frame->points[i].qy = global_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().y();
                pose_info_each_frame->points[i].qz = global_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().z();
            }

            for(int i = 0; i <= pose_keyframe->points.size() - slide_window_width; i++) {
                pose_keyframe->points[i].x = pose_each_frame->points[keyframe_id_in_frame[i]].x;
                pose_keyframe->points[i].y = pose_each_frame->points[keyframe_id_in_frame[i]].y;
                pose_keyframe->points[i].z = pose_each_frame->points[keyframe_id_in_frame[i]].z;

                pose_info_keyframe->points[i].x = pose_each_frame->points[keyframe_id_in_frame[i]].x;
                pose_info_keyframe->points[i].y = pose_each_frame->points[keyframe_id_in_frame[i]].y;
                pose_info_keyframe->points[i].z = pose_each_frame->points[keyframe_id_in_frame[i]].z;
                pose_info_keyframe->points[i].qw = pose_info_each_frame->points[keyframe_id_in_frame[i]].qw;
                pose_info_keyframe->points[i].qx = pose_info_each_frame->points[keyframe_id_in_frame[i]].qx;
                pose_info_keyframe->points[i].qy = pose_info_each_frame->points[keyframe_id_in_frame[i]].qy;
                pose_info_keyframe->points[i].qz = pose_info_each_frame->points[keyframe_id_in_frame[i]].qz;

                abs_poses[i+1][0] = pose_info_keyframe->points[i].qw;
                abs_poses[i+1][1] = pose_info_keyframe->points[i].qx;
                abs_poses[i+1][2] = pose_info_keyframe->points[i].qy;
                abs_poses[i+1][3] = pose_info_keyframe->points[i].qz;
                abs_poses[i+1][4] = pose_info_keyframe->points[i].x;
                abs_poses[i+1][5] = pose_info_keyframe->points[i].y;
                abs_poses[i+1][6] = pose_info_keyframe->points[i].z;

                Rs[i+1] = Eigen::Quaterniond(abs_poses[i+1][0],
                                             abs_poses[i+1][1],
                                             abs_poses[i+1][2],
                                             abs_poses[i+1][3]).toRotationMatrix();

                Ps[i+1][0] = abs_poses[i+1][4];
                Ps[i+1][1] = abs_poses[i+1][5];
                Ps[i+1][2] = abs_poses[i+1][6];
            }

            for(int i = abs_poses.size() - slide_window_width; i < abs_poses.size() - 1; i++) {
                Eigen::Quaterniond integratedQuaternion(abs_poses[i][0],
                                                        abs_poses[i][1],
                                                        abs_poses[i][2],
                                                        abs_poses[i][3]);
                Eigen::Vector3d integratedTransition(abs_poses[i][4],
                                                     abs_poses[i][5],
                                                     abs_poses[i][6]);

                integratedTransition = integratedTransition + integratedQuaternion * transitionRel[i - abs_poses.size() + slide_window_width];
                integratedQuaternion = integratedQuaternion * quaternionRel[i - abs_poses.size() + slide_window_width];

                abs_poses[i+1][0] = integratedQuaternion.w();
                abs_poses[i+1][1] = integratedQuaternion.x();
                abs_poses[i+1][2] = integratedQuaternion.y();
                abs_poses[i+1][3] = integratedQuaternion.z();
                abs_poses[i+1][4] = integratedTransition.x();
                abs_poses[i+1][5] = integratedTransition.y();
                abs_poses[i+1][6] = integratedTransition.z();

                Rs[i+1] = Eigen::Quaterniond(abs_poses[i+1][0],
                                             abs_poses[i+1][1],
                                             abs_poses[i+1][2],
                                             abs_poses[i+1][3]).toRotationMatrix();

                Ps[i+1][0] = abs_poses[i+1][4];
                Ps[i+1][1] = abs_poses[i+1][5];
                Ps[i+1][2] = abs_poses[i+1][6];

                pose_keyframe->points[i].x = abs_poses[i+1][4];
                pose_keyframe->points[i].y = abs_poses[i+1][5];
                pose_keyframe->points[i].z = abs_poses[i+1][6];

                pose_info_keyframe->points[i].x = abs_poses[i+1][4];
                pose_info_keyframe->points[i].y = abs_poses[i+1][5];
                pose_info_keyframe->points[i].z = abs_poses[i+1][6];
                pose_info_keyframe->points[i].qw = abs_poses[i+1][0];
                pose_info_keyframe->points[i].qx = abs_poses[i+1][1];
                pose_info_keyframe->points[i].qy = abs_poses[i+1][2];
                pose_info_keyframe->points[i].qz = abs_poses[i+1][3];
            }

            abs_pose = abs_poses.back();
            for (int i = 0; i < 7; ++i) {
                last_pose[i] = abs_poses[abs_poses.size() - slide_window_width][i];
            }

            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];

            loop_closed = false;
            marg = false;
        }
    }

    void full_cloudHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn) {
        full_cloud->clear();
        pcl::fromROSMsg(*pointCloudIn, *full_cloud);
        pcl::PointCloud<PointType>::Ptr full(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*full_cloud, *full);
        new_full_cloud = true;
    }

    void surfaceLastHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn) {
        surf_last->clear();
        pcl::fromROSMsg(*pointCloudIn, *surf_last);
        new_surf = true;
    }

    void odomHandler(const nav_msgs::Odometry::ConstPtr& odomIn) {
        time_new_odom = odomIn->header.stamp.toSec();
        odom_cur = odomIn;
        new_odom = true;
    }

    void eachOdomHandler(const nav_msgs::Odometry::ConstPtr& odomIn) {
        each_odom_buf.push_back(odomIn);
        if(each_odom_buf.size() > 5000)
            each_odom_buf[each_odom_buf.size() - 5001] = nullptr;
        new_each_odom = true;
    }

    void imuHandler(const sensor_msgs::ImuConstPtr& ImuIn) {
        time_last_imu = ImuIn->header.stamp.toSec();

        sensor_msgs::ImuPtr ImuIn_tmp (new sensor_msgs::Imu(*ImuIn));
        imu_buf.push_back(ImuIn_tmp);
        if(imu_buf.size() > 60000)
            imu_buf[imu_buf.size() - 60001] = nullptr;

        if (cur_time_imu < 0)
            cur_time_imu = time_last_imu;

        if (!first_imu) {
            Eigen::Quaterniond quat(ImuIn->orientation.w,
                                    ImuIn->orientation.x,
                                    ImuIn->orientation.y,
                                    ImuIn->orientation.z);
            Rs[0] = quat.toRotationMatrix();

            abs_poses[0][0] = ImuIn->orientation.w;
            abs_poses[0][1] = ImuIn->orientation.x;
            abs_poses[0][2] = ImuIn->orientation.y;
            abs_poses[0][3] = ImuIn->orientation.z;

            if(true)
            {
                tf2::Quaternion gt_init_q;
                gt_init_q.setRPY(initial_rpy_[0] * 3.1415926/180, initial_rpy_[1] * 3.1415926/180, initial_rpy_[2] * 3.1415926/180);
                gt_init_q.normalize();
                initial_Quat = Eigen::Quaterniond(gt_init_q[3],gt_init_q[0],gt_init_q[1],gt_init_q[2]);
                Rs[0] = initial_Quat.toRotationMatrix();

                abs_poses[0][0] = initial_Quat.w();
                abs_poses[0][1] = initial_Quat.x();
                abs_poses[0][2] = initial_Quat.y();
                abs_poses[0][3] = initial_Quat.z();
            }

            first_imu = true;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            dx = ImuIn->linear_acceleration.x;
            dy = ImuIn->linear_acceleration.y;
            dz = ImuIn->linear_acceleration.z;
            rx = ImuIn->angular_velocity.x;
            ry = ImuIn->angular_velocity.y;
            rz = ImuIn->angular_velocity.z;

            Eigen::Vector3d linear_acceleration(dx, dy, dz);
            Eigen::Vector3d angular_velocity(rx, ry, rz);
            acc_0 = linear_acceleration;
            gyr_0 = angular_velocity;
            pre_integrations.push_back(new Preintegration(acc_0, gyr_0, Bas[0], Bgs[0]));
            pre_integrations.back()->g_vec_ = -g;
        }
    }

    void transformPoint(PointType const *const pi, PointType *const po) {
        Eigen::Quaterniond quaternion(abs_pose[0], abs_pose[1], abs_pose[2], abs_pose[3]);
        Eigen::Vector3d transition(abs_pose[4], abs_pose[5], abs_pose[6]);
        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;
        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
    }

    void transformPoint(PointType const *const pi, PointType *const po, Eigen::Quaterniond quaternion, Eigen::Vector3d transition) {
        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;
        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);
        for (int i = 0; i < numPts; ++i) {
            PointType ptIn = cloudIn->points[i];
            PointType ptOut;
            transformPoint(&ptIn, &ptOut);
            cloudOut->points[i] = ptOut;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, PointPoseInfo * PointInfoIn) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        Eigen::Quaterniond quaternion(PointInfoIn->qw, PointInfoIn->qx, PointInfoIn->qy, PointInfoIn->qz);
        Eigen::Vector3d transition(PointInfoIn->x, PointInfoIn->y, PointInfoIn->z);
        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);
        for (int i = 0; i < numPts; ++i) {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;
            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;
            cloudOut->points[i] = pt;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, Eigen::Quaterniond quaternion, Eigen::Vector3d transition) {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);
        for (int i = 0; i < numPts; ++i) {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;
            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;
            cloudOut->points[i] = pt;
        }
        return cloudOut;
    }

    void processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity) {
        if(pre_integrations.size() < abs_poses.size()) {
            pre_integrations.push_back(new Preintegration(acc_0, gyr_0, Bas.back(), Bgs.back()));
            pre_integrations.back()->g_vec_ = -g;
            Bas.push_back(Bas.back());
            Bgs.push_back(Bgs.back());
            Rs.push_back(Rs.back());
            Ps.push_back(Ps.back());
            Vs.push_back(Vs.back());
        }

        Eigen::Vector3d un_acc_0 = Rs.back() * (acc_0 - Bas.back()) - g;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs.back();
        Rs.back() *= deltaQ(un_gyr * dt).toRotationMatrix();
        Eigen::Vector3d un_acc_1 = Rs.back() * (linear_acceleration - Bas.back()) - g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps.back() += dt * Vs.back() + 0.5 * dt * dt * un_acc;
        Vs.back() += dt * un_acc;

        pre_integrations.back()->push_back(dt, linear_acceleration, angular_velocity);

        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    void publishOdometry() {

        if(pose_info_keyframe->points.size() >= slide_window_width) {
            time_new_odom = keyframe_time[pose_info_keyframe->points.size()-slide_window_width];
            odom_mapping.header.stamp = ros::Time().fromSec(time_new_odom);
            odom_mapping.pose.pose.orientation.w = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].qw;

            odom_mapping.pose.pose.orientation.x = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].qx;

            odom_mapping.pose.pose.orientation.y = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].qy;
            odom_mapping.pose.pose.orientation.z = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].qz;
            odom_mapping.pose.pose.position.x = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].x;
            odom_mapping.pose.pose.position.y = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].y;
            odom_mapping.pose.pose.position.z = pose_info_keyframe->points[pose_info_keyframe->points.size()-slide_window_width].z;
            
            odom_mapping.twist.twist.linear.x = Bas.back().x(); // bias of imu acc
            odom_mapping.twist.twist.linear.y = Bas.back().y();
            odom_mapping.twist.twist.linear.z = Bas.back().z();
            odom_mapping.twist.twist.angular.x = Bgs.back().x();
            odom_mapping.twist.twist.angular.y = Bgs.back().y();
            odom_mapping.twist.twist.angular.z = Bgs.back().z();
            pub_odom.publish(odom_mapping);

            /* results for evo */
//            ofstream fout_evo(result_path_evo, ios::app); //GROUND_TRUTH_PATH_EVO
//            fout_evo.setf(ios::fixed, ios::floatfield);
//            fout_evo.precision(8);
//            fout_evo  << time_new_odom << ' ';
//            fout_evo.precision(8);
//            fout_evo  << odom_mapping.pose.pose.position.x << ' '
//                        << odom_mapping.pose.pose.position.y << ' '
//                        << odom_mapping.pose.pose.position.z << ' '
//                        << odom_mapping.pose.pose.orientation.x << ' '
//                        << odom_mapping.pose.pose.orientation.y << ' '
//                        << odom_mapping.pose.pose.orientation.z << ' '
//                        << odom_mapping.pose.pose.orientation.w << '\n';
//            fout_evo.close();

            /* pose estimation */
            enu_pos = Ps[Ps.size()-slide_window_width];
            enu_ypr = Utility::R2ypr(Rs[Rs.size()-1]);

            /* write result to file */
            ofstream tc_sw_output(tc_sw_result_path, ios::app);
            tc_sw_output.setf(ios::fixed, ios::floatfield);
            tc_sw_output.precision(8);
            tc_sw_output << time_new_odom << ','
                        << enu_ypr.x() << ','
                        << enu_ypr.y() << ','
                        << enu_ypr.z() << ','
                        << enu_pos[0] << ','
                        << enu_pos[1] << ','
                        << enu_pos[2] << '\n';
            tc_sw_output.close();


            // publish local-imu body tf
             static tf::TransformBroadcaster br;
             tf::Transform transform_enu_world;
             tf::Quaternion tf_q;

             // publish world-map tf
             tf::Transform transform_map_world;
             transform_map_world.setOrigin(tf::Vector3(odom_mapping.pose.pose.position.x, odom_mapping.pose.pose.position.y, odom_mapping.pose.pose.position.z));
             tf_q.setW(odom_mapping.pose.pose.orientation.w);
             tf_q.setX(odom_mapping.pose.pose.orientation.x);
             tf_q.setY(odom_mapping.pose.pose.orientation.y);
             tf_q.setZ(odom_mapping.pose.pose.orientation.z);
             transform_map_world.setRotation(tf_q);
             br.sendTransform(tf::StampedTransform(transform_map_world, odom_mapping.header.stamp, "RTTLIO", "RTTLIO_dyna"));

            
        }

        sensor_msgs::PointCloud2 msgs;

        if (pub_poses.getNumSubscribers() && pose_info_keyframe->points.size() >= slide_window_width) {
            pcl::toROSMsg(*pose_each_frame, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_poses.publish(msgs);

        }


        PointPoseInfo Tbl;
        Tbl.qw = q_bl.w();
        Tbl.qx = q_bl.x();
        Tbl.qy = q_bl.y();
        Tbl.qz = q_bl.z();
        Tbl.x = t_bl.x();
        Tbl.y = t_bl.y();
        Tbl.z = t_bl.z();

        // publish the surf feature points in lidar_init frame

        if (pub_surf.getNumSubscribers()) {
            for (int i = 0; i < surf_last_ds->points.size(); ++i) {
                transformPoint(&surf_last_ds->points[i], &surf_last_ds->points[i], q_bl, t_bl);
                transformPoint(&surf_last_ds->points[i], &surf_last_ds->points[i]);
            }
            pcl::PointCloud<PointType>::Ptr surf_res(new pcl::PointCloud<PointType>());
            Eigen::Matrix4f tf_initial = Eigen::Matrix4f::Identity();
            tf2::Quaternion gt_init_q;
            gt_init_q.setRPY(0 * 3.1415926/180, 0 * 3.1415926/180, 0 * 3.1415926/180);
            gt_init_q.normalize();
            tf_initial.block(0,0,3,3) = Eigen::Quaternionf(gt_init_q[3],gt_init_q[0],gt_init_q[1],gt_init_q[2]).toRotationMatrix();
            pcl::transformPointCloud(*surf_last_ds, *surf_res, tf_initial);
            pcl::toROSMsg(*surf_res, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_surf.publish(msgs);
        }

        if (pub_full.getNumSubscribers()) {
            for (int i = 0; i < full_cloud->points.size(); ++i) {
                transformPoint(&full_cloud->points[i], &full_cloud->points[i], q_bl, t_bl);
                transformPoint(&full_cloud->points[i], &full_cloud->points[i]);
            }
            pcl::toROSMsg(*full_cloud, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_full.publish(msgs);
        }

    }

    void publishLCOdometry() {

        std::ofstream lc_output(lc_result_path, std::ios::out);
        lc_output.close();
//        std::ofstream res_lc_evo_output(lc_result_path_evo, std::ios::out);
//        res_lc_evo_output.close();

        nav_msgs::Path lc_enu_path;
        lc_enu_path.header.frame_id = frame_id;
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses - 1; ++i)
        {
            double time_frame = pose_info_keyframe->points[i].time;
            geometry_msgs::PoseStamped lc_pose;
            lc_pose.header.stamp = ros::Time().fromSec(time_frame);
            tf::Quaternion q_tmp = tf::createQuaternionFromRPY(isamCurrentEstimate.at<Pose3>(i).rotation().roll(),
                                                               isamCurrentEstimate.at<Pose3>(i).rotation().pitch(),
                                                               isamCurrentEstimate.at<Pose3>(i).rotation().yaw());
            lc_pose.pose.orientation.w = q_tmp.w();
            lc_pose.pose.orientation.x = q_tmp.x();
            lc_pose.pose.orientation.y = q_tmp.y();
            lc_pose.pose.orientation.z = q_tmp.z();

            Eigen::Vector3d i_pos (isamCurrentEstimate.at<Pose3>(i).translation().x(),
                                   isamCurrentEstimate.at<Pose3>(i).translation().y(),
                                   isamCurrentEstimate.at<Pose3>(i).translation().z());

            lc_pose.pose.position.x = i_pos[0];
            lc_pose.pose.position.y = i_pos[1];
            lc_pose.pose.position.z = i_pos[2];

            lc_enu_path.poses.push_back(lc_pose);

            Eigen::Vector3d tmp_pos (lc_pose.pose.position.x,
                                     lc_pose.pose.position.y,
                                     lc_pose.pose.position.z);
            Eigen::Matrix3d tmp_rot = Eigen::Quaterniond (q_tmp.w(), q_tmp.x(), q_tmp.y(), q_tmp.z()).toRotationMatrix();

            /* results for evo */
//            ofstream fout_evo(lc_result_path_evo, ios::app); //
//            fout_evo.setf(ios::fixed, ios::floatfield);
//            fout_evo.precision(8);
//            fout_evo  << time_frame << ' ';
//            fout_evo.precision(8);
//            fout_evo  << lc_pose.pose.position.x << ' '
//                      << lc_pose.pose.position.y << ' '
//                      << lc_pose.pose.position.z << ' '
//                      << lc_pose.pose.orientation.x << ' '
//                      << lc_pose.pose.orientation.y << ' '
//                      << lc_pose.pose.orientation.z << ' '
//                      << lc_pose.pose.orientation.w << '\n';
//            fout_evo.close();

            enu_pos = tmp_pos;
            enu_ypr = Utility::R2ypr(tmp_rot);

            /* write result to file */
            ofstream lc_output(lc_result_path, ios::app);
            lc_output.setf(ios::fixed, ios::floatfield);
            lc_output.precision(8);
            lc_output << time_frame << ','
                        << enu_ypr.x() << ','
                        << enu_ypr.y() << ','
                        << enu_ypr.z() << ','
                        << enu_pos[0] << ','
                        << enu_pos[1] << ','
                        << enu_pos[2] << '\n';
            lc_output.close();
        }

    }

    void clearCloud() {
        surf_local_map->clear();
        surf_local_map_ds->clear();

        // if(surf_lasts_ds.size() > slide_window_width + 5) {
        //     surf_lasts_ds[surf_lasts_ds.size() - slide_window_width - 6]->clear();
        // }

        // if(pre_integrations.size() > slide_window_width + 5) {
        //     pre_integrations[pre_integrations.size() - slide_window_width - 6] = nullptr;
        // }

        // if(last_marginalization_parameter_blocks.size() > slide_window_width + 5) {
        //     last_marginalization_parameter_blocks[last_marginalization_parameter_blocks.size() - slide_window_width - 6] = nullptr;
        // }

        if(surf_lasts_ds.size() > 3 * slide_window_width + 1) {
            surf_lasts_ds[surf_lasts_ds.size() - 3 * slide_window_width - 2]->clear();
        }

//        if(surf_frames.size() > batch_fusion_width*1.5) {
//            surf_frames[surf_frames.size() - batch_fusion_width*1.5 - 1]->clear();
//        }

//        if(pre_integrations.size() > 2 * slide_window_width + 1) {
//            pre_integrations[pre_integrations.size() - 2 * slide_window_width - 2] = nullptr;
//        }

        /* window size of 10 */
//        if(last_marginalization_parameter_blocks.size() > 2 * slide_window_width + 1) {
//            last_marginalization_parameter_blocks[last_marginalization_parameter_blocks.size() - 2 * slide_window_width - 2] = nullptr;
//        }

        /* window size of 3 */
        // if(last_marginalization_parameter_blocks.size() > slide_window_width + 5) {
        //     last_marginalization_parameter_blocks[last_marginalization_parameter_blocks.size() - slide_window_width - 6] = nullptr;
        // }
    }

    void loopClosureThread() {
        if (!loop_closure_on)
            return;

        ros::Rate rate(1);
        while (ros::ok()) {
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure() {
        latest_key_frames->clear();
        latest_key_frames_ds->clear();
        his_key_frames->clear();
        his_key_frames_ds->clear();

        std::lock_guard<std::mutex> lock(mutual_exclusion);

        // Look for the closest key frames
        std::vector<int> pt_search_idxLoop;
        std::vector<float> pt_search_sq_distsLoop;

        kd_tree_his_key_poses->setInputCloud(pose_keyframe);
        kd_tree_his_key_poses->radiusSearch(select_pose, lc_search_radius, pt_search_idxLoop, pt_search_sq_distsLoop, 0);

        closest_his_idx = -1;
        for (int i = 0; i < pt_search_idxLoop.size(); ++i) {
            int idx = pt_search_idxLoop[i];
            if (abs(pose_info_keyframe->points[idx].time - time_new_odom) > lc_time_thres) {
                closest_his_idx = idx;
                break;
            }
        }

        if (closest_his_idx == -1)
            return false;
        else if(abs(time_last_loop - time_new_odom) < 0.2)
            return false;

//        ROS_INFO("******************* Loop closure ready to detect! *******************");

        // Combine the corner and surf frames to form the latest frame
        latest_frame_idx_loop = pose_keyframe->points.size() - slide_window_width;

        for (int j = 0; j < 6; ++j) {
            if (latest_frame_idx_loop-j < 0)
                continue;
            Eigen::Quaterniond q_po(pose_info_keyframe->points[latest_frame_idx_loop-j].qw,
                                    pose_info_keyframe->points[latest_frame_idx_loop-j].qx,
                                    pose_info_keyframe->points[latest_frame_idx_loop-j].qy,
                                    pose_info_keyframe->points[latest_frame_idx_loop-j].qz);

            Eigen::Vector3d t_po(pose_info_keyframe->points[latest_frame_idx_loop-j].x,
                                 pose_info_keyframe->points[latest_frame_idx_loop-j].y,
                                 pose_info_keyframe->points[latest_frame_idx_loop-j].z);

            Eigen::Quaterniond q_tmp = q_po * q_bl;
            Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

            *latest_key_frames += *transformCloud(surf_frames[latest_frame_idx_loop-j], q_tmp, t_tmp);
        }

        ds_filter_his_frames.setInputCloud(latest_key_frames);
        ds_filter_his_frames.filter(*latest_key_frames_ds);


        // Form the history frame for loop closure detection
        for (int j = -lc_map_width; j <= lc_map_width; ++j) {
            if (closest_his_idx + j < 0 || closest_his_idx + j > latest_frame_idx_loop)
                continue;

            Eigen::Quaterniond q_po(pose_info_keyframe->points[closest_his_idx+j].qw,
                                    pose_info_keyframe->points[closest_his_idx+j].qx,
                                    pose_info_keyframe->points[closest_his_idx+j].qy,
                                    pose_info_keyframe->points[closest_his_idx+j].qz);

            Eigen::Vector3d t_po(pose_info_keyframe->points[closest_his_idx+j].x,
                                 pose_info_keyframe->points[closest_his_idx+j].y,
                                 pose_info_keyframe->points[closest_his_idx+j].z);

            Eigen::Quaterniond q_tmp = q_po * q_bl;
            Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

            *his_key_frames += *transformCloud(surf_frames[closest_his_idx+j], q_tmp, t_tmp);
        }

        ds_filter_his_frames.setInputCloud(his_key_frames);
        ds_filter_his_frames.filter(*his_key_frames_ds);

        return true;
    }

    void performLoopClosure() {
        if (pose_keyframe->points.empty())
            return;

        if (!loop_to_close) {
            if (detectLoopClosure())
                loop_to_close = true;
            if (!loop_to_close)
                return;
        }

        loop_to_close = false;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(30);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(5);

        icp.setInputSource(latest_key_frames_ds);
        icp.setInputTarget(his_key_frames_ds);
        pcl::PointCloud<PointType>::Ptr alignedCloud(new pcl::PointCloud<PointType>());
        icp.align(*alignedCloud);

        //std::cout << "ICP converg flag:" << icp.hasConverged() << ". Fitness score: " << icp.getFitnessScore() << endl;

        if (!icp.hasConverged() || icp.getFitnessScore() > lc_icp_thres)
            return;

        Timer t_loop("Loop Closure");
//        ROS_INFO("******************* Loop closure detected! *******************");

        Eigen::Matrix4d correctedTranform;
        correctedTranform = icp.getFinalTransformation().cast<double>();
        Eigen::Quaterniond quaternionIncre(correctedTranform.block<3, 3>(0, 0));
        Eigen::Vector3d transitionIncre(correctedTranform.block<3, 1>(0, 3));
        Eigen::Quaterniond quaternionToCorrect(pose_info_keyframe->points[latest_frame_idx_loop].qw,
                                               pose_info_keyframe->points[latest_frame_idx_loop].qx,
                                               pose_info_keyframe->points[latest_frame_idx_loop].qy,
                                               pose_info_keyframe->points[latest_frame_idx_loop].qz);
        Eigen::Vector3d transitionToCorrect(pose_info_keyframe->points[latest_frame_idx_loop].x,
                                            pose_info_keyframe->points[latest_frame_idx_loop].y,
                                            pose_info_keyframe->points[latest_frame_idx_loop].z);

        Eigen::Quaterniond quaternionCorrected = quaternionIncre * quaternionToCorrect;
        Eigen::Vector3d transitionCorrected = quaternionIncre * transitionToCorrect + transitionIncre;

        gtsam::Rot3 rotationFrom = gtsam::Rot3::Quaternion(quaternionCorrected.w(), quaternionCorrected.x(), quaternionCorrected.y(), quaternionCorrected.z());
        gtsam::Point3 transitionFrom = gtsam::Point3(transitionCorrected.x(), transitionCorrected.y(), transitionCorrected.z());

        gtsam::Rot3 rotationTo = gtsam::Rot3::Quaternion(pose_info_keyframe->points[closest_his_idx].qw,
                                                         pose_info_keyframe->points[closest_his_idx].qx,
                                                         pose_info_keyframe->points[closest_his_idx].qy,
                                                         pose_info_keyframe->points[closest_his_idx].qz);
        gtsam::Point3 transitionTo = gtsam::Point3(pose_info_keyframe->points[closest_his_idx].x,
                                                   pose_info_keyframe->points[closest_his_idx].y,
                                                   pose_info_keyframe->points[closest_his_idx].z);

        gtsam::Pose3 poseFrom = gtsam::Pose3(rotationFrom, transitionFrom);
        gtsam::Pose3 poseTo = gtsam::Pose3(rotationTo, transitionTo);
        gtsam::Vector vector6(6);
        double noiseScore = icp.getFitnessScore();
        vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraint_noise = gtsam::noiseModel::Diagonal::Variances(vector6);

        std::lock_guard<std::mutex> lock(mutual_exclusion);

        global_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(keyframe_id_in_frame[latest_frame_idx_loop],
                                                                 keyframe_id_in_frame[closest_his_idx],
                                                                 poseFrom.between(poseTo),
                                                                 constraint_noise));
        isam->update(global_pose_graph);
        isam->update();
        global_pose_graph.resize(0);

        loop_closed = true;

        global_estimated = isam->calculateEstimate();
        correctPoses();

        if (last_marginalization_info) {
            delete last_marginalization_info;
        }
        last_marginalization_info = nullptr;

        time_last_loop = pose_info_keyframe->points[latest_frame_idx_loop].time;

//        ROS_INFO("******************* Loop closure finished! *******************");
        //t_loop.tic_toc();
    }

    void publishCompleteMap() {
        if (pose_keyframe->points.size() > 10) {
            for (int i = 0; i < pose_info_keyframe->points.size(); i = i + mapping_interval) {
                Eigen::Quaterniond q_po(pose_info_keyframe->points[i].qw,
                                        pose_info_keyframe->points[i].qx,
                                        pose_info_keyframe->points[i].qy,
                                        pose_info_keyframe->points[i].qz);

                Eigen::Vector3d t_po(pose_info_keyframe->points[i].x,
                                     pose_info_keyframe->points[i].y,
                                     pose_info_keyframe->points[i].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                *global_map += *transformCloud(full_clouds_ds[i], &Ttmp);
            }

            ds_filter_global_map.setInputCloud(global_map);
            ds_filter_global_map.filter(*global_map_ds);

            sensor_msgs::PointCloud2 msgs;
            pcl::toROSMsg(*global_map_ds, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_map.publish(msgs);
            global_map->clear();
            global_map_ds->clear();
        }
    }

    void mapVisualizationThread() {
        ros::Rate rate(0.038);
        while (ros::ok()) {
            return;
            rate.sleep();
            ROS_INFO("Publishing the map");
            publishCompleteMap();
        }

        if(!save_pcd)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;

        PointPoseInfo Tbl;
        Tbl.qw = q_bl.w();
        Tbl.qx = q_bl.x();
        Tbl.qy = q_bl.y();
        Tbl.qz = q_bl.z();
        Tbl.x = t_bl.x();
        Tbl.y = t_bl.y();
        Tbl.z = t_bl.z();

        for (int i = 0; i < pose_info_keyframe->points.size(); i = i + mapping_interval) {
            *global_map += *transformCloud(transformCloud(surf_frames[i], &Tbl), &pose_info_keyframe->points[i]);
        }
        ds_filter_global_map.setInputCloud(global_map);
        ds_filter_global_map.filter(*global_map_ds);
        std::string pcd_path = result_path + "global_map.pcd";
        pcl::io::savePCDFileASCII(pcd_path, *global_map_ds);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
        global_map->clear();
        global_map_ds->clear();
    }

    void backendFusionThread() {
        if (!enable_batch_fusion) return;
        ros::Rate rate(10);
        while (ros::ok()) {
            rate.sleep();
            optimizeBatchWithLandMark();
        }
    }

    void run()
    {
        if (new_surf && new_odom && new_each_odom && new_full_cloud)
        {
            frame_count++;

            new_surf = false;
            new_odom = false;
            new_each_odom = false;
            new_full_cloud = false;

            std::lock_guard<std::mutex> lock(mutual_exclusion);

            buildLocalMapWithLandMark();
            downSampleCloud();
            saveKeyFramesAndFactors();
            publishOdometry();
            clearCloud();
        }
    }
};


int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    ros::init(argc, argv, "RTTLIO");
    ros::NodeHandle nh("~");

    std::string result_path;
    nh.param<std::string>("result_path", result_path, "../result/");

    // 自动创建目录（如果不存在）
    struct stat info;
    if (stat(result_path.c_str(), &info) != 0)
    {
        std::string cmd = "mkdir -p '" + result_path + "'";
        system(cmd.c_str());
    }

    // std::string output_file = result_path + "/output.csv";
    // std::ofstream out(output_file);
    // std::streambuf *coutbuf = std::cout.rdbuf(); // 保存原始缓冲区
    // std::cout.rdbuf(out.rdbuf());                // 重定向到文件

    ROS_INFO("\033[1;32m---->\033[0m RTTLIO Started.");

    /* define the estimator */
    Estimator Estimator_;

    // 记录系统开始时间
    auto system_start = std::chrono::high_resolution_clock::now();

    /* loop closure detection thread */
    std::thread threadLoopClosure(&Estimator::loopClosureThread, &Estimator_);

    /* map visualization thread, typically the map visualization can be time consuming */
    std::thread threadMapVisualization(&Estimator::mapVisualizationThread, &Estimator_);

    /* backend fusion thread */
    // std::thread threadBackendFusion(&Estimator::backendFusionThread, &Estimator_);

    ros::Rate rate(200);
    bool publish_gt=false;
    while (ros::ok()) {
        ros::spinOnce();
        Estimator_.run();

        rate.sleep();
    }

    auto system_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(system_end - system_start).count();

    

    threadLoopClosure.join();
    threadMapVisualization.join();
    // threadBackendFusion.join();
    // std::cout.rdbuf(coutbuf); // 恢复原始缓冲区

    return 0;
}
