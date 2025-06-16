#include "basis.h"

Eigen::MatrixXd Basis::evalPhi(const Eigen::MatrixXd& xHat) {
    int N = xHat.rows();
    Eigen::MatrixXd phi(N, 3);
    
    phi.col(0) = 1.0 - xHat.col(0).array() - xHat.col(1).array();
    phi.col(1) = xHat.col(0);
    phi.col(2) = xHat.col(1);
    
    return phi;
}

Eigen::MatrixXd Basis::evalGradPhi(const Eigen::MatrixXd& xHat) {
    int N = xHat.rows();
    Eigen::MatrixXd gradPhi(N * 3, 2);
    
    for (int i = 0; i < N; ++i) {
        gradPhi.block<3, 2>(i * 3, 0) << -1, -1,
                                           1,  0,
                                           0,  1;
    }
    
    return gradPhi;
}

GlobalBasis::GlobalBasis(const Grid& grid, const Basis& basis) 
    : grid(grid), basis(basis) {}

Eigen::VectorXd GlobalBasis::evalPhi(const Eigen::MatrixXd& xHat, int globalInd) {
    auto [supp, localInd] = evalDOFMap(globalInd);
    Eigen::MatrixXd valHat = basis.evalPhi(xHat);
    
    Eigen::VectorXd result(xHat.rows());
    for (int i = 0; i < xHat.rows(); ++i) {
        result(i) = valHat(i, localInd[0]); // 简化版本
    }
    
    return result;
}

std::pair<std::vector<int>, std::vector<int>> GlobalBasis::evalDOFMap(int globalInd) {
    std::vector<int> supp, localInd;
    
    for (int T = 0; T < grid.getCells().rows(); ++T) {
        for (int j = 0; j < 3; ++j) {
            if (grid.getCells()(T, j) == globalInd) {
                supp.push_back(T);
                localInd.push_back(j);
            }
        }
    }
    
    return {supp, localInd};
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> 
GlobalBasis::evalSharedDOFMap(int globalIndI, int globalIndJ) {
    auto [suppI, localIndI] = evalDOFMap(globalIndI);
    auto [suppJ, localIndJ] = evalDOFMap(globalIndJ);
    
    std::vector<int> sharedSupp, sharedLocalI, sharedLocalJ;
    
    for (size_t i = 0; i < suppI.size(); ++i) {
        for (size_t j = 0; j < suppJ.size(); ++j) {
            if (suppI[i] == suppJ[j]) {
                sharedSupp.push_back(suppI[i]);
                sharedLocalI.push_back(localIndI[i]);
                sharedLocalJ.push_back(localIndJ[j]);
            }
        }
    }
    
    return {sharedSupp, sharedLocalI, sharedLocalJ};
}