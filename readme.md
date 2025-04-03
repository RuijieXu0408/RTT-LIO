# RTT-LIO: A Wi-Fi RTT-aided LiDAR-Inertial Odometry via Tightly-Coupled Factor Graph Optimization for UAS

RTT-LIO is a novel tightly-coupled factor graph optimization framework that integrates Wi-Fi Round-Trip Time (RTT) measurements with LiDAR and inertial data for robust state estimation in Unmanned Aerial Systems (UAS). This system addresses the critical challenges of accurate localization in GPS-denied environments by using the complementary characteristics of multiple sensing modalities. The package is based on C++ which is compatible with the robot operation system (ROS) platform.

[demo]

**Authors**: [Ruijie Xu](https://polyu-taslab.github.io/members/Xu_Ruijie.html) [Xikun Liu](https://polyu-taslab.github.io/members/liu_xikun.html), [Xin Wang](https://polyu-taslab.github.io/members/wang_xin.html), [Weisong Wen](https://weisongwen.github.io/), from the [TASLAB](https://polyu-taslab.github.io/), The Hong Kong Polytechnic University.

**Contact**: Technical issue: [ruijie.xu@connect.polyu.hk](https://github.com/RuijieXu0408), commercial issue: [welson.wen@polyu.edu.hk](https://github.com/weisongwen).

## System Pipeline

RTT-LIO implements a comprehensive multi-stage processing pipeline:

![flowchart](https://github.com/RuijieXu0408/RTT-LIO/blob/main/fig/flowchart.png)

1. **Frontend Processing**:
   - **Keyframe Selection and Initialization**: Selects informative LiDAR frames for processing and establishes system initialization parameters
   - **IMU Pre-integration**: Performs integration of inertial measurements between LiDAR frames to capture high-frequency motion dynamics
   - **RTT Range from Multi-APs Modeling**: Models the ranging measurements from multiple Wi-Fi access points, handling outliers and non-line-of-sight conditions
2. **Factor Graph Construction**:
   - **Scan-to-multiscan LiDAR Factor**: Establishes geometric constraints between consecutive LiDAR scans
   - **IMU Factor**: Provides motion constraints from inertial measurements
3. **Two-Stage Optimization**:
   - **Global Pose from First-stage FGO**: Performs initial optimization to estimate UAS trajectory
   - **RTT Range Factor**: Adds distance constraints between the UAS and Wi-Fi access points
   - **Batch-based Second-stage FGO with FDE**: Implements Fault Detection and Exclusion (FDE) to identify and reject erroneous measurements, enhancing trajectory estimation robustness
4. **AP Position Estimation (Optional)**:
   - **RANSAC-based KDE and GDC**: Uses Random Sample Consensus with Kernel Density Estimation and Geometric Diversity Constraints for accurate access point position estimation, leveraging the estimated UAS trajectory

This tightly-coupled architecture enables RTT-LIO to maintain submeter-level accuracy even in challenging environments by dynamically weighting measurements based on their reliability and ensuring consistent constraints across the entire trajectory.

## Code and Datasets

The complete source code and datasets will be released soon, enabling researchers and developers to build upon this framework for various UAS applications. Stay tuned for this repository link and comprehensive documentation.
