#include "problem.h"
#include <Eigen/SparseLU>
#include <iostream>
#include <fstream>
#include <set>
#include <numeric>

StationaryProblem::StationaryProblem(const GlobalBasis& globalBasis, 
                                   const QuadratureRule& quadrature,
                                   std::function<bool(const Eigen::VectorXd&)> dirichletLocations,
                                   std::function<double(const Eigen::VectorXd&)> dirichletValues)
    : grid(globalBasis.grid), basis(globalBasis.basis), globalBasis(globalBasis), quadrature(quadrature) {
    
    // 获取边界条件DOF
    dirichletDOFs = grid.getBoundaryIndices(dirichletLocations);
    
    // 设置所有DOF和自由DOF
    int numDOFs = grid.getPoints().rows();
    allDOFs.resize(numDOFs);
    std::iota(allDOFs.begin(), allDOFs.end(), 0);
    
    std::set<int> dirichletSet(dirichletDOFs.begin(), dirichletDOFs.end());
    for (int i = 0; i < numDOFs; ++i) {
        if (dirichletSet.find(i) == dirichletSet.end()) {
            freeDOFs.push_back(i);
        }
    }
    
    // 预计算数据
    auto [points, weights] = quadrature.getPointsAndWeights();
    xkHat = points;
    wkHat = weights;
    
    xkTrafo = grid.evalReferenceMap(xkHat);
    dSh = grid.getDeterminants();
    invJacT = grid.getInverseJacobiansT();
    
    phi = basis.evalPhi(xkHat);
    gradPhi = basis.evalGradPhi(xkHat);
    
    // 初始化矩阵和向量
    systemMatrix = Eigen::MatrixXd::Zero(numDOFs, numDOFs);
    systemVector = Eigen::VectorXd::Zero(numDOFs);
    mass = Eigen::MatrixXd::Zero(numDOFs, numDOFs);
    diffusion = Eigen::MatrixXd::Zero(numDOFs, numDOFs);
    
    initialMass();
    initialDiffusion();
    
    a = 0.0;
    c = 0.0;
    
    assembleBoundaryConditions(dirichletValues);
}

void StationaryProblem::assembleBoundaryConditions(std::function<double(const Eigen::VectorXd&)> dirichletValues) {
    for (int dof : dirichletDOFs) {
        systemMatrix(dof, dof) = 1.0;
        systemVector(dof) = dirichletValues(grid.getPoints().row(dof));
    }
}

void StationaryProblem::addSource(std::function<double(const Eigen::VectorXd&)> f) {
    int numQuadPoints = xkHat.rows();
    int numCells = grid.getCells().rows();
    int nodesPerElement = grid.getNodesPerElement();
    
    for (int i : freeDOFs) {
        auto [supp, localInd] = globalBasis.evalDOFMap(i);
        
        for (size_t idx = 0; idx < supp.size(); ++idx) {
            int T = supp[idx];
            int loc_i = localInd[idx];
            
            double integral = 0.0;
            for (int k = 0; k < numQuadPoints; ++k) {
                Eigen::VectorXd x = xkTrafo.row(T * numQuadPoints + k);
                integral += phi(k, loc_i) * wkHat(k) * f(x);
            }
            systemVector(i) += dSh(T) * integral;
        }
    }
}

void StationaryProblem::addDiscreteSource(const Eigen::VectorXd& vec) {
    systemVector += mass * vec;
}

void StationaryProblem::initialDiffusion() {
    int numQuadPoints = xkHat.rows();
    int nodesPerElement = grid.getNodesPerElement();
    int dim = grid.getDimension();
    
    for (int i : freeDOFs) {
        for (int j : allDOFs) {
            auto [supp_IJ, localIndices_I, localIndices_J] = globalBasis.evalSharedDOFMap(i, j);
            
            for (size_t idx = 0; idx < supp_IJ.size(); ++idx) {
                int T = supp_IJ[idx];
                int loc_i = localIndices_I[idx];
                int loc_j = localIndices_J[idx];
                
                for (int k = 0; k < numQuadPoints; ++k) {
                    Eigen::VectorXd gradPhi_i = gradPhi.block(k * nodesPerElement + loc_i, 0, 1, dim);
                    Eigen::VectorXd gradPhi_j = gradPhi.block(k * nodesPerElement + loc_j, 0, 1, dim);
                    
                    Eigen::MatrixXd invJac = invJacT.block(T * dim, 0, dim, dim);
                    Eigen::VectorXd transformedGradI = invJac * gradPhi_i;
                    Eigen::VectorXd transformedGradJ = invJac * gradPhi_j;
                    
                    diffusion(i, j) += dSh(T) * wkHat(k) * transformedGradI.dot(transformedGradJ);
                }
            }
        }
    }
}

void StationaryProblem::initialMass() {
    int numQuadPoints = xkHat.rows();
    int nodesPerElement = grid.getNodesPerElement();
    
    for (int i : freeDOFs) {
        for (int j : allDOFs) {
            auto [supp_IJ, localIndices_I, localIndices_J] = globalBasis.evalSharedDOFMap(i, j);
            
            for (size_t idx = 0; idx < supp_IJ.size(); ++idx) {
                int T = supp_IJ[idx];
                int loc_i = localIndices_I[idx];
                int loc_j = localIndices_J[idx];
                
                for (int k = 0; k < numQuadPoints; ++k) {
                    mass(i, j) += dSh(T) * wkHat(k) * phi(k, loc_j) * phi(k, loc_i);
                }
            }
        }
    }
}

void StationaryProblem::setReaction(double c) {
    this->c = c;
}

void StationaryProblem::setDiffusion(double a) {
    this->a = a;
}

void StationaryProblem::setSolution(const Eigen::VectorXd& vec) {
    solution = vec;
}

Eigen::VectorXd StationaryProblem::getSolution() const {
    return solution;
}

void StationaryProblem::assemble() {
    systemMatrix += c * mass + a * diffusion;
}

void StationaryProblem::resetSystemVector() {
    systemVector.setZero();
}

void StationaryProblem::resetSystemMatrix() {
    systemMatrix.setZero();
}

void StationaryProblem::solve() {
    Eigen::FullPivLU<Eigen::MatrixXd> solver(systemMatrix);
    solution = solver.solve(systemVector);
}

void StationaryProblem::show() {
    // 输出解到文件用于可视化
    std::string filename = grid.is3D() ? "solution_3d.dat" : "solution_2d.dat";
    std::ofstream file(filename);
    const auto& points = grid.getPoints();
    
    if (grid.is3D()) {
        for (int i = 0; i < points.rows(); ++i) {
            file << points(i, 0) << " " << points(i, 1) << " " << points(i, 2) << " " << solution(i) << std::endl;
        }
    } else {
        for (int i = 0; i < points.rows(); ++i) {
            file << points(i, 0) << " " << points(i, 1) << " " << solution(i) << std::endl;
        }
    }
    file.close();
    
    std::cout << "Solution written to " << filename << std::endl;
}