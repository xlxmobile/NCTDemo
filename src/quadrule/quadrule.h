#ifndef QUADRULE_H
#define QUADRULE_H

#include <Eigen/Dense>

class QuadratureRule {
public:
    QuadratureRule(int order = 1);
    
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> getPointsAndWeights() const;

private:
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
};

#endif