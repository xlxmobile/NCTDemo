#ifndef BASIS_H
#define BASIS_H

#include <Eigen/Dense>
#include "grid.h"

class Basis {
public:
    Basis(bool is3D = false) : is3D(is3D) {}
    
    Eigen::MatrixXd evalPhi(const Eigen::MatrixXd& xHat);
    Eigen::MatrixXd evalGradPhi(const Eigen::MatrixXd& xHat);
    
    bool is3D;
};

class GlobalBasis {
public:
    GlobalBasis(const Grid& grid, const Basis& basis);
    
    Eigen::VectorXd evalPhi(const Eigen::MatrixXd& xHat, int globalInd);
    std::pair<std::vector<int>, std::vector<int>> evalDOFMap(int globalInd);
    std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> 
        evalSharedDOFMap(int globalIndI, int globalIndJ);

private:
    const Grid& grid;
    const Basis& basis;
};

#endif