#include "quadrule.h"
#include <iostream>

QuadratureRule::QuadratureRule(int order) {
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
                      << " is not implemented. Returning order 2 quadrature." << std::endl;
        }
    }
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> QuadratureRule::getPointsAndWeights() const {
    return {points, weights};
}