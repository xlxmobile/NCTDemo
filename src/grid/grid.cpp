#include "grid.h"
#include <iostream>
#include <cmath>

Grid::Grid(double xlow, double xhigh, double ylow, double yhigh, int Nx, int Ny)
    : xlow(xlow), xhigh(xhigh), ylow(ylow), yhigh(yhigh), Nx(Nx), Ny(Ny) {
    createGrid();
    updateTrafoInformation();
}

void Grid::createGrid() {
    // 创建点
    points.resize(Nx * Ny, 2);
    
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nx, xlow, xhigh);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(Ny, ylow, yhigh);
    
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            points(j * Nx + i, 0) = x(i);
            points(j * Nx + i, 1) = y(j);
        }
    }
    
    // 创建单元
    cells.resize(2 * (Nx - 1) * (Ny - 1), 3);
    
    int cellIndex = 0;
    for (int j = 0; j < Ny - 1; ++j) {
        for (int i = 0; i < Nx - 1; ++i) {
            int p0 = j * Nx + i;
            int p1 = j * Nx + i + 1;
            int p2 = (j + 1) * Nx + i;
            int p3 = (j + 1) * Nx + i + 1;
            
            // 第一个三角形
            cells(cellIndex, 0) = p0;
            cells(cellIndex, 1) = p1;
            cells(cellIndex, 2) = p2;
            cellIndex++;
            
            // 第二个三角形
            cells(cellIndex, 0) = p3;
            cells(cellIndex, 1) = p2;
            cells(cellIndex, 2) = p1;
            cellIndex++;
        }
    }
}

void Grid::updateTrafoInformation() {
    int numCells = cells.rows();
    adet.resize(numCells);
    invJacT.resize(numCells * 2, 2);
    
    for (int T = 0; T < numCells; ++T) {
        Eigen::Vector2d v0 = points.row(cells(T, 0));
        Eigen::Vector2d v1 = points.row(cells(T, 1));
        Eigen::Vector2d v2 = points.row(cells(T, 2));
        
        double det = (v1(0) - v0(0)) * (v2(1) - v0(1)) - (v2(0) - v0(0)) * (v1(1) - v0(1));
        adet(T) = std::abs(det);
        
        // 逆雅可比矩阵的转置
        invJacT.block<2, 2>(T * 2, 0) << (v2(1) - v0(1)) / det, (v0(0) - v2(0)) / det,
                                          (v0(1) - v1(1)) / det, (v1(0) - v0(0)) / det;
    }
}

Eigen::MatrixXd Grid::evalReferenceMap(const Eigen::MatrixXd& xHat) {
    int numCells = cells.rows();
    int numQuadPoints = xHat.rows();
    
    Eigen::MatrixXd result(numCells * numQuadPoints, 2);
    
    for (int T = 0; T < numCells; ++T) {
        Eigen::Vector2d v0 = points.row(cells(T, 0));
        Eigen::Vector2d v1 = points.row(cells(T, 1));
        Eigen::Vector2d v2 = points.row(cells(T, 2));
        
        for (int k = 0; k < numQuadPoints; ++k) {
            double xi = xHat(k, 0);
            double eta = xHat(k, 1);
            
            result.row(T * numQuadPoints + k) = (1 - xi - eta) * v0 + xi * v1 + eta * v2;
        }
    }
    
    return result;
}

bool Grid::isBoundaryPoint(const Eigen::Vector2d& p) {
    const double eps = 1e-6;
    return (p(0) <= xlow + eps || p(0) >= xhigh - eps || 
            p(1) <= ylow + eps || p(1) >= yhigh - eps);
}

std::vector<int> Grid::getInnerIndices() {
    std::vector<int> innerIndices;
    const double eps = 1e-6;
    
    for (int i = 0; i < points.rows(); ++i) {
        if (!isBoundaryPoint(points.row(i))) {
            innerIndices.push_back(i);
        }
    }
    
    return innerIndices;
}

std::vector<int> Grid::getBoundaryIndices(std::function<bool(double, double)> locator) {
    std::vector<int> boundaryIndices;
    
    for (int i = 0; i < points.rows(); ++i) {
        if (isBoundaryPoint(points.row(i)) && 
            locator(points(i, 0), points(i, 1))) {
            boundaryIndices.push_back(i);
        }
    }
    
    return boundaryIndices;
}