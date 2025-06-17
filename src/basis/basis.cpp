#include "basis/basis.h"

Eigen::MatrixXd Basis::evalPhi(const Eigen::MatrixXd& xHat) {
    int N = xHat.rows();
    
    Eigen::MatrixXd phi(N, 4);
    
    phi.col(0) = 1.0 - xHat.col(0).array() - xHat.col(1).array() - xHat.col(2).array();
    phi.col(1) = xHat.col(0);
    phi.col(2) = xHat.col(1);
    phi.col(3) = xHat.col(2);
        
    return phi;
}

Eigen::MatrixXd Basis::evalGradPhi(const Eigen::MatrixXd& xHat) {
    int N = xHat.rows();

    Eigen::MatrixXd gradPhi(N * 4, 3);
    
    for (int i = 0; i < N; ++i) {
        gradPhi.block<4, 3>(i * 4, 0) << -1, -1, -1,
                                           1,  0,  0,
                                           0,  1,  0,
                                           0,  0,  1;
    }
    
    return gradPhi;

}

GlobalBasis::GlobalBasis(const Grid& grid, const Basis& basis) 
    : grid(grid), basis(basis) {}

Eigen::VectorXd GlobalBasis::evalPhi(const Eigen::MatrixXd& xHat, int globalInd) {
    
    auto [supp, localInd] = evalDOFMap(globalInd);
    
    
    Eigen::MatrixXd valHat = basis.evalPhi(xHat);
    
    int numQuadPoints = xHat.rows();
    int numSuppCells = supp.size();
    
   
    Eigen::VectorXd result = Eigen::VectorXd::Zero(numQuadPoints * numSuppCells);
    
    
    for (size_t cellIdx = 0; cellIdx < numSuppCells; ++cellIdx) {
        int localNodeIndex = localInd[cellIdx];
        
        
        for (int quadIdx = 0; quadIdx < numQuadPoints; ++quadIdx) {
            
            result(cellIdx * numQuadPoints + quadIdx) = valHat(quadIdx, localNodeIndex);
        }
    }
    
    return result;
}

std::pair<std::vector<int>, std::vector<int>> GlobalBasis::evalDOFMap(int globalInd) {
    std::vector<int> supp, localInd;
    int nodesPerElement = grid.getNodesPerElement();
    
    for (int T = 0; T < grid.getCells().rows(); ++T) {
        for (int j = 0; j < nodesPerElement; ++j) {
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