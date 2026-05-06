FROM osrf/ros:noetic-desktop-full

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    libatlas-base-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    libeigen3-dev \
    libpcl-dev \
    ros-noetic-pcl-ros \
    ros-noetic-pcl-conversions \
    ros-noetic-velodyne-msgs \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]

# Build Ceres Solver 2.2.0 from source
RUN mkdir -p /opt/software && cd /opt/software \
    && git clone --depth 1 --branch 2.2.0 https://ceres-solver.googlesource.com/ceres-solver \
    && mkdir ceres-bin && cd ceres-bin \
    && cmake ../ceres-solver \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DBUILD_EXAMPLES=OFF \
    && make -j$(nproc) install \
    && rm -rf /opt/software/ceres-solver /opt/software/ceres-bin

# Build GTSAM 4.3 from source (bundles Eigen 3.4, required to match GTSAM's internal Eigen)
RUN cd /opt/software \
    && git clone --depth 1 --branch 4.3a0 https://github.com/borglab/gtsam.git \
    && mkdir gtsam-build && cd gtsam-build \
    && cmake ../gtsam \
        -DCMAKE_BUILD_TYPE=Release \
        -DGTSAM_BUILD_TESTS=OFF \
        -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
        -DGTSAM_USE_SYSTEM_EIGEN=OFF \
    && make -j$(nproc) install \
    && rm -rf /opt/software/gtsam /opt/software/gtsam-build

# Set up catkin workspace and clone RTTLIO
RUN mkdir -p /root/rttlio_ws/src \
    && cd /root/rttlio_ws/src \
    && git clone https://github.com/RuijieXu0408/RTT-LIO.git --branch rttlio-release --single-branch rttlio_pkg \
    && cd /root/rttlio_ws \
    && source /opt/ros/noetic/setup.bash \
    && catkin_make -DCMAKE_BUILD_TYPE=Release

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc \
    && echo "source /root/rttlio_ws/devel/setup.bash" >> /root/.bashrc

WORKDIR /root/rttlio_ws
