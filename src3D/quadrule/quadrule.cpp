#include "quadrule.h"
#include <iostream>

QuadratureRule::QuadratureRule(int order, bool is3D) : is3D(is3D) {
    if (is3D) {
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
            
            if (order > 2) {
                std::cout << "Requested order " << order 
                          << " is not implemented for 3D. Returning order 2 quadrature." << std::endl;
            }
        }
    } else {
        // 2D三角形积分规则
        if (order == 1) {
            weights.resize(1);
            points.resize(1, 2);
            
            weights(0) = 0.5;
            points(0, 0) = 1.0/3.0;
            points(0, 1) = 1.0/3.0;
        } else {
            weights.resize(3);
            points.resize(3, 2);
            
            weights << 1.0/6.0, 1.0/6.0, 1.0/6.0;
            points << 1.0/6.0, 1.0/6.0,
                      2.0/3.0, 1.0/6.0,
                      1.0/6.0, 2.0/3.0;
            
            if (order > 2) {
                std::cout << "Requested order " << order 
                          << " is not implemented for 2D. Returning order 2 quadrature." << std::endl;
            }
        }
    }
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> QuadratureRule::getPointsAndWeights() const {
    return {points, weights};
}