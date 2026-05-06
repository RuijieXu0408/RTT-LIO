# RTT-LIO: A Wi-Fi RTT-aided LiDAR-Inertial Odometry via Tightly-Coupled Factor Graph Optimization for UAS

RTT-LIO is a novel tightly-coupled factor graph optimization framework that integrates Wi-Fi Round-Trip Time (RTT) measurements with LiDAR and inertial data for robust state estimation in Unmanned Aerial Systems (UAS). This system addresses the critical challenges of accurate localization in GPS-denied environments by using the complementary characteristics of multiple sensing modalities. The package is based on C++ which is compatible with the robot operation system (ROS) platform.

**Authors**: [Ruijie Xu](https://polyu-taslab.github.io/members/Xu_Ruijie.html) [Xikun Liu*](https://polyu-taslab.github.io/members/liu_xikun.html), [Xin Wang](https://polyu-taslab.github.io/members/wang_xin.html), [Weisong Wen](https://weisongwen.github.io/), from the [TASLAB](https://polyu-taslab.github.io/), The Hong Kong Polytechnic University. And [Yulong Huang](https://faculty.hrbeu.edu.cn/huangyulong/zh_CN/index.htm), from [HEU](www.hrbeu.edu.cn)

**Contact**: For questions, bug reports, or collaboration inquiries, please contact ruijie.xu@connect.polyu.hk.

**Citation**: If you use this repository, please cite our paper:
```
@article{xu2026rtt,
  title={RTT-LIO: A Wi-Fi RTT-aided LiDAR-Inertial Odometry via Tightly-Coupled Factor Graph Optimization in Complex Scenes},
  author={Xu, Ruijie and Liu, Xikun and Wang, Xin and Wen, Weisong and Huang, Yulong},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```


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
4. **AP Position Calibration (Optional)**:
   - **RANSAC-based KDE and GDC**: Uses Random Sample Consensus with Kernel Density Estimation and Geometric Diversity Constraints for accurate access point position estimation, leveraging the estimated UAS trajectory

This tightly-coupled architecture enables RTT-LIO to maintain submeter-level accuracy even in challenging environments by dynamically weighting measurements based on their reliability and ensuring consistent constraints across the entire trajectory.

## Code and Datasets

Three ROS nodes run in sequence:

| Node | Role |
|---|---|
| `Preprocessing` | Subscribes to raw LiDAR scans, extracts edge/surface features |
| `LidarOdometry` | Scan-to-scan feature matching to produce a high-rate odometry prior |
| `Estimator` | Sliding-window + batch factor graph (IMU pre-integration, LiDAR pose, Wi-Fi RTT range) |

---

## Key Features

- **Tightly-coupled** Wi-Fi RTT / LiDAR / IMU fusion in a single factor graph
- **Sliding-window** optimization (real-time) + **batch** re-optimization for global consistency
- **Marginalization** for bounded computational cost
- **Loop closure** via ICP + GTSAM iSAM2 (optional, toggle `loop_closure_on` in config)
- **AP position agnostic**: works with pre-surveyed AP coordinates
- Supports 16/32/64-beam Velodyne LiDARs

---

## Prerequisites

| Dependency | Version tested | Notes |
|---|---|---|
| Ubuntu | 20.04 | |
| ROS | Noetic | full desktop |
| C++ | 14 | |
| CMake | ≥ 3.10 | |
| Eigen | 3.3.7 (system) | GTSAM bundles its own 3.4; handled automatically |
| PCL | 1.10 | via `ros-noetic-pcl-ros` |
| Ceres Solver | **2.2.0** | build from source |
| GTSAM | **4.3** | build from source (`-DGTSAM_USE_SYSTEM_EIGEN=OFF`) |


---

## Installation

### Option A — Docker (recommended)

The provided `Dockerfile` installs all dependencies and builds the workspace automatically.

```bash
# Clone the repo
git clone https://github.com/RuijieXu0408/RTT-LIO.git -b rttlio-release
cd RTT-LIO

# Build the image (takes ~10–15 min on first run)
docker build -f src/RTTLIO/Dockerfile -t rttlio:latest .

# Run with display forwarding (for RViz)
xhost +local:docker
docker run -it --rm \
    --net=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/your/bag:/data \
    rttlio:latest bash
```

Inside the container, the workspace is at `/root/rttlio_ws` and the environment is already sourced.

---

### Option B — Native build

#### 1. Install ROS Noetic

Follow the [official guide](http://wiki.ros.org/noetic/Installation/Ubuntu).

```bash
sudo apt install ros-noetic-desktop-full
```

#### 2. Install system dependencies

```bash
sudo apt install -y \
    cmake build-essential git wget \
    libatlas-base-dev libgoogle-glog-dev libsuitesparse-dev \
    libpcl-dev \
    ros-noetic-pcl-ros ros-noetic-pcl-conversions \
    ros-noetic-velodyne-msgs
```

#### 3. Build Ceres Solver 2.2.0

```bash
git clone --depth 1 --branch 2.2.0 \
    https://ceres-solver.googlesource.com/ceres-solver
mkdir ceres-bin && cd ceres-bin
cmake ../ceres-solver \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_EXAMPLES=OFF
make -j$(nproc) && sudo make install
```

#### 4. Build GTSAM 4.3

Build with the **bundled Eigen** (`-DGTSAM_USE_SYSTEM_EIGEN=OFF`) to avoid the Eigen version assertion failure.

```bash
git clone --depth 1 --branch 4.3a0 https://github.com/borglab/gtsam.git
mkdir gtsam-build && cd gtsam-build
cmake ../gtsam \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_USE_SYSTEM_EIGEN=OFF
make -j$(nproc) && sudo make install
```

#### 5. Create workspace and build RTT-LIO

```bash
mkdir -p ~/rttlio_ws/src && cd ~/rttlio_ws/src
git clone https://github.com/RuijieXu0408/RTT-LIO.git \
    --branch rttlio-release --single-branch rttlio_pkg

cd ~/rttlio_ws
source /opt/ros/noetic/setup.bash
catkin_make -DCMAKE_BUILD_TYPE=Release
```

---

## Configuration

Edit `src/RTTLIO/config/config_urban_hk.yaml` before launching:

```yaml
IMU:
  imu_topic: "/imu/data"        # change to match your bag
  acc_n: 3.99e-03               # accelerometer noise density
  gyr_n: 1.56e-03               # gyroscope noise density

lidar_odometry:
  lidar_topic: "/velodyne_points"
  line_num: 32                  # 16 / 32 / 64

initialization:
  Euler_y: -155                 # initial yaw (degrees) — tune per dataset
  gt_path: /path/to/gt.csv      # ground-truth CSV for evaluation (optional)

Estimator:
  enable_batch_fusion: true     # enable batch RTT re-optimization
  loop_closure_on: false        # enable loop closure (requires more CPU)
  slide_window_width: 5
  surf_dist_thres: 0.18
```

The Wi-Fi RTT data file is set in the launch file:

```xml
<!-- run_urban_hk.launch -->
<param name="sol_folder" type="string" value="/path/to/your/rtt_data.csv" />
```

---

## Running

```bash
# Terminal 1 — source and launch
source ~/rttlio_ws/devel/setup.bash
roslaunch RTTLIO run_urban_hk.launch

# Terminal 2 — play your bag
rosbag play /path/to/your.bag
```

Expected topics the bag should contain:

| Topic | Type | Description |
|---|---|---|
| `/velodyne_points` | `sensor_msgs/PointCloud2` | 3D LiDAR scan |
| `/imu/data` | `sensor_msgs/Imu` | IMU (200 Hz recommended) |

Wi-Fi RTT measurements are loaded from the CSV file specified by `sol_folder` (not from a ROS topic).

---

## Dataset Format

The RTT data CSV (`sol_folder`) has one measurement per line:

```
unixtime, ap_index, ap_pos_x, ap_pos_y, ap_pos_z, range
```

| Field | Unit | Description |
|---|---|---|
| `unixtime` | s | Unix timestamp of the RTT measurement |
| `ap_index` | — | Access point identifier (integer) |
| `ap_pos_x/y/z` | m | AP position in the local ENU frame |
| `range` | m | RTT-derived range measurement |

Example datasets can be found in the `WiFiRTT_Dataset/` folder (see the repository).
