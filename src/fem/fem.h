#ifndef PROBLEM_H
#define PROBLEM_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <functional>
#include "grid.h"
#include "basis.h"
#include "quadrule.h"

class StationaryProblem {
public:
    StationaryProblem(const GlobalBasis& globalBasis, 
                     const QuadratureRule& quadrature,
                     std::function<bool(double, double)> dirichletLocations = [](double, double){ return true; },
                     std::function<double(double, double)> dirichletValues = [](double, double){ return 0.0; });
    
    void assembleBoundaryConditions(std::function<double(double, double)> dirichletValues);
    void addSource(std::function<double(double, double)> f);
    void addDiscreteSource(const Eigen::VectorXd& vec);
    
    void setReaction(double c);
    void setDiffusion(double a);
    void setSolution(const Eigen::VectorXd& vec);
    Eigen::VectorXd getSolution() const;
    
    void assemble();
    void resetSystemVector();
    void resetSystemMatrix();
    void solve();
    void show();

private:
    void initialDiffusion();
    void initialMass();
    
    const Grid& grid;
    const Basis& basis;
    const GlobalBasis& globalBasis;
    const QuadratureRule& quadrature;
    
    std::vector<int> dirichletDOFs;
    std::vector<int> allDOFs;
    std::vector<int> freeDOFs;
    
    Eigen::MatrixXd xkHat;
    Eigen::VectorXd wkHat;
    Eigen::MatrixXd xkTrafo;
    Eigen::VectorXd dSh;
    Eigen::MatrixXd invJacT;
    Eigen::MatrixXd phi;
    Eigen::MatrixXd gradPhi;
    
    Eigen::MatrixXd systemMatrix;
    Eigen::VectorXd systemVector;
    Eigen::VectorXd solution;
    
    Eigen::MatrixXd mass;
    Eigen::MatrixXd diffusion;
    
    double a, c;
};

#endif