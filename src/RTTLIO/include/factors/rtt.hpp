#ifndef RTTLIO_RTT_FACTOR_H
#define RTTLIO_RTT_FACTOR_H

#include <vector>
#include <Eigen/Dense>
#include <ceres/ceres.h>

/* height constraint factor */
struct heightFactor
{
    heightFactor(double height) : height_(height) {}

    template <typename T>
    bool operator()(const T *statePi, T *residuals) const
    {
        Eigen::Matrix<T, 3, 1> Pi(statePi[0], statePi[1], statePi[2]);
        residuals[0] = Pi[2] - T(height_);
        return true;
    }
    double height_;
};

/* tightly coupled RTT range factor with per-AP bias estimation */
struct rttFactor
{
    rttFactor(int ap_index, double pos_x, double pos_y, double pos_z, double range, double ts_ratio)
        : ap_index(ap_index), pos_x(pos_x), pos_y(pos_y), pos_z(pos_z), range(range), ts_ratio(ts_ratio) {}

    template <typename T>
    bool operator()(const T *statePi, const T *statePj, const T *Delay, T *residuals) const
    {
        Eigen::Vector3d ap_pos_ = Eigen::Vector3d(pos_x, pos_y, pos_z);
        T rtt_range = T(range);
        Eigen::Matrix<T, 3, 1> ap_pos = ap_pos_.cast<T>();
        Eigen::Matrix<T, 3, 1> Pi(statePi[0], statePi[1], statePi[2]);
        Eigen::Matrix<T, 3, 1> Pj(statePj[0], statePj[1], statePj[2]);
        Eigen::Matrix<T, 4, 1> Delay_(Delay[0], Delay[1], Delay[2], Delay[3]);
        Eigen::Matrix<T, 3, 1> P_ = T(ts_ratio) * Pi + (T(1.0) - T(ts_ratio)) * Pj;

        Eigen::Matrix<T, 3, 1> p2ap = ap_pos - P_;
        T p2ap_dist = T(sqrt(p2ap[0] * p2ap[0] + p2ap[1] * p2ap[1] + p2ap[2] * p2ap[2]));

        switch (int(ap_index))
        {
        case 1:
            rtt_range = rtt_range - Delay_[0];
        case 2:
            rtt_range = rtt_range - Delay_[1];
        case 3:
            rtt_range = rtt_range - Delay_[2];
        case 4:
            rtt_range = rtt_range - Delay_[3];
        }
        residuals[0] = rtt_range - p2ap_dist;
        return true;
    }
    double ap_index, pos_x, pos_y, pos_z, range, ts_ratio;
};

/* tightly coupled RTT range factor without bias */
struct rttFactor_noD
{
    rttFactor_noD(int ap_index, double pos_x, double pos_y, double pos_z, double range, double ts_ratio)
        : ap_index(ap_index), pos_x(pos_x), pos_y(pos_y), pos_z(pos_z), range(range), ts_ratio(ts_ratio) {}

    template <typename T>
    bool operator()(const T *statePi, const T *statePj, T *residuals) const
    {
        Eigen::Vector3d ap_pos_ = Eigen::Vector3d(pos_x, pos_y, pos_z);
        T rtt_range = T(range);
        Eigen::Matrix<T, 3, 1> ap_pos = ap_pos_.cast<T>();
        Eigen::Matrix<T, 3, 1> Pi(statePi[0], statePi[1], statePi[2]);
        Eigen::Matrix<T, 3, 1> Pj(statePj[0], statePj[1], statePj[2]);
        Eigen::Matrix<T, 3, 1> P_ = T(ts_ratio) * Pi + (T(1.0) - T(ts_ratio)) * Pj;

        Eigen::Matrix<T, 3, 1> p2ap = ap_pos - P_;
        T p2ap_dist = T(sqrt(p2ap[0] * p2ap[0] + p2ap[1] * p2ap[1] + p2ap[2] * p2ap[2]));

        residuals[0] = p2ap_dist - rtt_range;
        return true;
    }
    double ap_index, pos_x, pos_y, pos_z, range, ts_ratio;
};

/* RTT range factor with simultaneous AP position estimation */
struct rttFactorWithAPEstimation
{
    rttFactorWithAPEstimation(int ap_index, double range, double ts_ratio)
        : ap_index(ap_index), range(range), ts_ratio(ts_ratio) {}

    template <typename T>
    bool operator()(const T *statePi, const T *statePj, const T *ap_pos, T *residuals) const
    {
        Eigen::Matrix<T, 3, 1> Pi(statePi[0], statePi[1], statePi[2]);
        Eigen::Matrix<T, 3, 1> Pj(statePj[0], statePj[1], statePj[2]);
        Eigen::Matrix<T, 3, 1> P_ = T(ts_ratio) * Pi + (T(1.0) - T(ts_ratio)) * Pj;

        Eigen::Matrix<T, 3, 1> ap_position(ap_pos[0], ap_pos[1], ap_pos[2]);
        Eigen::Matrix<T, 3, 1> p2ap = ap_position - P_;
        T estimated_range = sqrt(p2ap[0] * p2ap[0] + p2ap[1] * p2ap[1] + p2ap[2] * p2ap[2]);

        residuals[0] = T(range) - estimated_range;
        return true;
    }

    double ap_index, range, ts_ratio;
};

/* AP position prior factor */
struct APPriorFactor
{
    APPriorFactor(const Eigen::Vector3d &prior_pos, double weight)
        : prior_pos(prior_pos), weight(weight) {}

    template <typename T>
    bool operator()(const T *ap_pos, T *residuals) const
    {
        residuals[0] = T(weight) * (T(prior_pos.x()) - ap_pos[0]);
        residuals[1] = T(weight) * (T(prior_pos.y()) - ap_pos[1]);
        residuals[2] = T(weight) * (T(prior_pos.z()) - ap_pos[2]);
        return true;
    }

    Eigen::Vector3d prior_pos;
    double weight;
};

/* motion model factor */
struct MotionModelFactorAutoDiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MotionModelFactorAutoDiff(Eigen::Vector3d delta_p) : delta_p(delta_p){}

    template <typename T>
    bool operator()(const T *statePi, const T *statePj, T *residuals) const
    {
        Eigen::Matrix<T, 3, 1> Pi(statePi[0], statePi[1], statePi[2]);
        Eigen::Matrix<T, 3, 1> Pj(statePj[0], statePj[1], statePj[2]);
        Eigen::Matrix<T, 3, 1> tmp_delta_p(T(delta_p.x()), T(delta_p.y()), T(delta_p.z()));
        Eigen::Matrix<T, 3, 1> residual1 = (Pi - Pj) - tmp_delta_p;

        residuals[0] = T(10) * residual1[0];
        residuals[1] = T(10) * residual1[1];
        residuals[2] = T(10) * residual1[2];
        return true;
    }
    static ceres::CostFunction *Create(Eigen::Vector3d delta_p)
    {
        return (new ceres::AutoDiffCostFunction<MotionModelFactorAutoDiff, 3, 3, 3>(
            new MotionModelFactorAutoDiff(delta_p)));
    }
    Eigen::Vector3d delta_p;
};

#endif // RTTLIO_RTT_FACTOR_H
