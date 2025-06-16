#ifndef GRID_H
#define GRID_H

#include <Eigen/Dense>
#include <vector>
#include <functional>

class Grid {
public:
    Grid(double xlow, double xhigh, double ylow, double yhigh, int Nx, int Ny);
    
    void createGrid();
    void updateTrafoInformation();
    Eigen::MatrixXd evalReferenceMap(const Eigen::MatrixXd& xHat);
    
    bool isBoundaryPoint(const Eigen::Vector2d& p);
    std::vector<int> getInnerIndices();
    std::vector<int> getBoundaryIndices(std::function<bool(double, double)> locator);
    
    // Getters
    std::pair<int, int> getDivisions() const { return {Nx, Ny}; }
    const Eigen::VectorXd& getDeterminants() const { return adet; }
    const Eigen::MatrixXd& getInverseJacobiansT() const { return invJacT; }
    const Eigen::MatrixXd& getPoints() const { return points; }
    const Eigen::MatrixXi& getCells() const { return cells; }

protected:
    double xlow, xhigh, ylow, yhigh;
    int Nx, Ny;
    
public:
    Eigen::MatrixXd points;
    Eigen::MatrixXi cells;
    Eigen::VectorXd adet;
    Eigen::MatrixXd invJacT;
};

#endif