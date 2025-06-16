#include "problem.h"
#include <Eigen/SparseLU>
#include <iostream>
#include <fstream>

StationaryProblem::StationaryProblem(const GlobalBasis& globalBasis, 
                                   const QuadratureRule& quadrature,
                                   std::function<bool(double, double)> dirichletLocations,
                                   std::function<double(double, double)> dirichletValues)
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

void StationaryProblem::assembleBoundaryConditions(std::function<double(double, double)> dirichletValues) {
    for (int dof : dirichletDOFs) {
        systemMatrix(dof, dof) = 1.0;
        systemVector(dof) = dirichletValues(grid.getPoints()(dof, 0), grid.getPoints()(dof, 1));
    }
}

void StationaryProblem::addSource(std::function<double(double, double)> f) {
    int numQuadPoints = xkHat.rows();
    int numCells = grid.getCells().rows();
    
    for (int i : freeDOFs) {
        auto [supp, localInd] = globalBasis.evalDOFMap(i);
        
        for (size_t idx = 0; idx < supp.size(); ++idx) {
            int T = supp[idx];
            int loc_i = localInd[idx];
            
            double integral = 0.0;
            for (int k = 0; k < numQuadPoints; ++k) {
                double x = xkTrafo(T * numQuadPoints + k, 0);
                double y = xkTrafo(T * numQuadPoints + k, 1);
                integral += phi(k, loc_i) * wkHat(k) * f(x, y);
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
    
    for (int i : freeDOFs) {
        for (int j : allDOFs) {
            auto [supp_IJ, localIndices_I, localIndices_J] = globalBasis.evalSharedDOFMap(i, j);
            
            for (size_t idx = 0; idx < supp_IJ.size(); ++idx) {
                int T = supp_IJ[idx];
                int loc_i = localIndices_I[idx];
                int loc_j = localIndices_J[idx];
                
                for (int k = 0; k < numQuadPoints; ++k) {
                    Eigen::Vector2d gradPhi_i = gradPhi.block<1, 2>(k * 3 + loc_i, 0);
                    Eigen::Vector2d gradPhi_j = gradPhi.block<1, 2>(k * 3 + loc_j, 0);
                    
                    Eigen::Matrix2d invJac = invJacT.block<2, 2>(T * 2, 0);
                    Eigen::Vector2d transformedGradI = invJac * gradPhi_i;
                    Eigen::Vector2d transformedGradJ = invJac * gradPhi_j;
                    
                    diffusion(i, j) += dSh(T) * wkHat(k) * transformedGradI.dot(transformedGradJ);
                }
            }
        }
    }
}

void StationaryProblem::initialMass() {
    int numQuadPoints = xkHat.rows();
    
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
    std::ofstream file("solution.dat");
    const auto& points = grid.getPoints();
    
    for (int i = 0; i < points.rows(); ++i) {
        file << points(i, 0) << " " << points(i, 1) << " " << solution(i) << std::endl;
    }
    file.close();
    
    std::cout << "Solution written to solution.dat" << std::endl;
}