#ifndef BASIS_H
#define BASIS_H

#include <Eigen/Dense>
#include "grid/grid.h"

class Basis {
public:
    
    Eigen::MatrixXd evalPhi(const Eigen::MatrixXd& xHat);
    Eigen::MatrixXd evalGradPhi(const Eigen::MatrixXd& xHat);

};

class GlobalBasis {
public:
    GlobalBasis(const Grid& grid, const Basis& basis);
    
    Eigen::VectorXd evalPhi(const Eigen::MatrixXd& xHat, int globalInd);
    Eigen::MatrixXd evalPhiOnSupport(const Eigen::MatrixXd& xHat, int globalInd);
    std::pair<std::vector<int>, std::vector<int>> evalDOFMap(int globalInd);
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> 
        evalSharedDOFMap(int globalIndI, int globalIndJ);

private:
    const Grid& grid;
    const Basis& basis;
};

#endif