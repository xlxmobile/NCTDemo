#include "quadrule/quadrule.h"
#include <iostream>

QuadratureRule::QuadratureRule(int order) {

    // 3D四面体积分规则
    if (order == 1) {
        weights.resize(1);
        points.resize(1, 3);
        
        weights(0) = 1.0/6.0;  // 四面体体积为1/6
        points(0, 0) = 0.25;
        points(0, 1) = 0.25;
        points(0, 2) = 0.25;
    } else {
        // 4点积分规则
        weights.resize(4);
        points.resize(4, 3);
        
        double a = (5.0 - std::sqrt(5.0)) / 20.0;
        double b = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
        
        weights.fill(1.0/24.0);
        
        points << a, a, a,
                  b, a, a,
                  a, b, a,
                  a, a, b;
    }
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> QuadratureRule::getPointsAndWeights() const {
    return {points, weights};
}