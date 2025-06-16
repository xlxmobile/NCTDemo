#ifndef GRID_H
#define GRID_H

#include <Eigen/Dense>
#include <vector>
#include <functional>

class Grid {
public:
    // 2D构造函数
    Grid(double xlow, double xhigh, double ylow, double yhigh, int Nx, int Ny);
    
    // 3D构造函数
    Grid(double xlow, double xhigh, double ylow, double yhigh, double zlow, double zhigh, 
         int Nx, int Ny, int Nz);
    
    void createGrid2D();
    void createGrid3D();
    void updateTrafoInformation();
    Eigen::MatrixXd evalReferenceMap(const Eigen::MatrixXd& xHat);
    
    bool isBoundaryPoint(const Eigen::VectorXd& p);
    std::vector<int> getInnerIndices();
    std::vector<int> getBoundaryIndices(std::function<bool(const Eigen::VectorXd&)> locator);
    
    // Getters
    std::tuple<int, int, int> getDivisions() const { return {Nx, Ny, Nz}; }
    const Eigen::VectorXd& getDeterminants() const { return adet; }
    const Eigen::MatrixXd& getInverseJacobiansT() const { return invJacT; }
    const Eigen::MatrixXd& getPoints() const { return points; }
    const Eigen::MatrixXi& getCells() const { return cells; }
    bool is3D() const { return is3DGrid; }
    int getDimension() const { return is3DGrid ? 3 : 2; }
    int getNodesPerElement() const { return is3DGrid ? 4 : 3; }

protected:
    double xlow, xhigh, ylow, yhigh, zlow, zhigh;
    int Nx, Ny, Nz;
    bool is3DGrid;
    
public:
    Eigen::MatrixXd points;
    Eigen::MatrixXi cells;
    Eigen::VectorXd adet;
    Eigen::MatrixXd invJacT;
};

#endif