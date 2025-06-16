#ifndef QUADRULE_H
#define QUADRULE_H

#include <Eigen/Dense>

class QuadratureRule {
public:
    QuadratureRule(int order = 1, bool is3D = false);
    
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> getPointsAndWeights() const;

private:
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    bool is3D;
};

#endif