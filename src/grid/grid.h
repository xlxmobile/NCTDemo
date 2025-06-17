#ifndef GRID_H
#define GRID_H

#include <Eigen/Dense>
#include <vector>
#include <functional>

class Grid {
public:

    Grid(double xlow, double xhigh, double ylow, double yhigh, double zlow, double zhigh, int Nx, int Ny, int Nz);

    void create3DGrid();
    void updateTrafoInformation();
    Eigen::MatrixXd evalReferenceMap(const Eigen::MatrixXd& Xhat);

    bool isBoundaryPoint(const Eigen::VectorXd& p);
    std::vector<int> getInnerIndices();
    std::vector<int> getBoundaryIndices(std::function<bool(const Eigen::VectorXd&)> locator);

    std::tuple<int, int, int> getDivisions() const { return {Nx, Ny, Nz}; }
    const Eigen::VectorXd& getDeterminants() const { return adet; }
    const Eigen::MatrixXd& getInverseJacobiansT() const { return invJacT; }
    const Eigen::MatrixXd& getPoints() const { return points; }
    const Eigen::MatrixXi& getCells() const { return cells; }

protected:
    double xlow, xhigh, ylow, yhigh, zlow, zhigh;
    int Nx, Ny, Nz;
    
public:
    Eigen::MatrixXd points;
    Eigen::MatrixXi cells;
    Eigen::VectorXd adet;
    Eigen::MatrixXd invJacT;


};

#endif