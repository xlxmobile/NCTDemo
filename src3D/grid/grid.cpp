#include "grid.h"
#include <iostream>
#include <cmath>
#include <algorithm>

// 2D构造函数
Grid::Grid(double xlow, double xhigh, double ylow, double yhigh, int Nx, int Ny)
    : xlow(xlow), xhigh(xhigh), ylow(ylow), yhigh(yhigh), zlow(0), zhigh(0),
      Nx(Nx), Ny(Ny), Nz(0), is3DGrid(false) {
    createGrid2D();
    updateTrafoInformation();
}

// 3D构造函数
Grid::Grid(double xlow, double xhigh, double ylow, double yhigh, double zlow, double zhigh,
           int Nx, int Ny, int Nz)
    : xlow(xlow), xhigh(xhigh), ylow(ylow), yhigh(yhigh), zlow(zlow), zhigh(zhigh),
      Nx(Nx), Ny(Ny), Nz(Nz), is3DGrid(true) {
    createGrid3D();
    updateTrafoInformation();
}

void Grid::createGrid2D() {
    // 创建2D点
    points.resize(Nx * Ny, 2);
    
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nx, xlow, xhigh);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(Ny, ylow, yhigh);
    
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            points(j * Nx + i, 0) = x(i);
            points(j * Nx + i, 1) = y(j);
        }
    }
    
    // 创建2D三角形单元
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

void Grid::createGrid3D() {
    // 创建3D点
    points.resize(Nx * Ny * Nz, 3);
    
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(Nx, xlow, xhigh);
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(Ny, ylow, yhigh);
    Eigen::VectorXd z = Eigen::VectorXd::LinSpaced(Nz, zlow, zhigh);
    
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = k * Nx * Ny + j * Nx + i;
                points(idx, 0) = x(i);
                points(idx, 1) = y(j);
                points(idx, 2) = z(k);
            }
        }
    }
    
    // 创建3D四面体单元
    // 每个立方体分解为6个四面体
    cells.resize(6 * (Nx - 1) * (Ny - 1) * (Nz - 1), 4);
    
    int cellIndex = 0;
    for (int k = 0; k < Nz - 1; ++k) {
        for (int j = 0; j < Ny - 1; ++j) {
            for (int i = 0; i < Nx - 1; ++i) {
                // 立方体的8个顶点
                int p000 = k * Nx * Ny + j * Nx + i;
                int p100 = k * Nx * Ny + j * Nx + i + 1;
                int p010 = k * Nx * Ny + (j + 1) * Nx + i;
                int p110 = k * Nx * Ny + (j + 1) * Nx + i + 1;
                int p001 = (k + 1) * Nx * Ny + j * Nx + i;
                int p101 = (k + 1) * Nx * Ny + j * Nx + i + 1;
                int p011 = (k + 1) * Nx * Ny + (j + 1) * Nx + i;
                int p111 = (k + 1) * Nx * Ny + (j + 1) * Nx + i + 1;
                
                // 将立方体分解为6个四面体
                // 四面体1: p000, p100, p010, p001
                cells(cellIndex, 0) = p000; cells(cellIndex, 1) = p100; 
                cells(cellIndex, 2) = p010; cells(cellIndex, 3) = p001;
                cellIndex++;
                
                // 四面体2: p100, p110, p010, p101
                cells(cellIndex, 0) = p100; cells(cellIndex, 1) = p110; 
                cells(cellIndex, 2) = p010; cells(cellIndex, 3) = p101;
                cellIndex++;
                
                // 四面体3: p010, p110, p011, p101
                cells(cellIndex, 0) = p010; cells(cellIndex, 1) = p110; 
                cells(cellIndex, 2) = p011; cells(cellIndex, 3) = p101;
                cellIndex++;
                
                // 四面体4: p001, p101, p010, p011
                cells(cellIndex, 0) = p001; cells(cellIndex, 1) = p101; 
                cells(cellIndex, 2) = p010; cells(cellIndex, 3) = p011;
                cellIndex++;
                
                // 四面体5: p100, p101, p110, p111
                cells(cellIndex, 0) = p100; cells(cellIndex, 1) = p101; 
                cells(cellIndex, 2) = p110; cells(cellIndex, 3) = p111;
                cellIndex++;
                
                // 四面体6: p110, p101, p011, p111
                cells(cellIndex, 0) = p110; cells(cellIndex, 1) = p101; 
                cells(cellIndex, 2) = p011; cells(cellIndex, 3) = p111;
                cellIndex++;
            }
        }
    }
}

void Grid::updateTrafoInformation() {
    int numCells = cells.rows();
    adet.resize(numCells);
    
    if (is3DGrid) {
        invJacT.resize(numCells * 3, 3);
        
        for (int T = 0; T < numCells; ++T) {
            Eigen::Vector3d v0 = points.row(cells(T, 0));
            Eigen::Vector3d v1 = points.row(cells(T, 1));
            Eigen::Vector3d v2 = points.row(cells(T, 2));
            Eigen::Vector3d v3 = points.row(cells(T, 3));
            
            // 计算雅可比矩阵
            Eigen::Matrix3d J;
            J.col(0) = v1 - v0;
            J.col(1) = v2 - v0;
            J.col(2) = v3 - v0;
            
            double det = J.determinant();
            adet(T) = std::abs(det);
            
            // 逆雅可比矩阵的转置
            invJacT.block<3, 3>(T * 3, 0) = J.inverse().transpose();
        }
    } else {
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
}

Eigen::MatrixXd Grid::evalReferenceMap(const Eigen::MatrixXd& xHat) {
    int numCells = cells.rows();
    int numQuadPoints = xHat.rows();
    int dim = getDimension();
    
    Eigen::MatrixXd result(numCells * numQuadPoints, dim);
    
    if (is3DGrid) {
        for (int T = 0; T < numCells; ++T) {
            Eigen::Vector3d v0 = points.row(cells(T, 0));
            Eigen::Vector3d v1 = points.row(cells(T, 1));
            Eigen::Vector3d v2 = points.row(cells(T, 2));
            Eigen::Vector3d v3 = points.row(cells(T, 3));
            
            for (int k = 0; k < numQuadPoints; ++k) {
                double xi = xHat(k, 0);
                double eta = xHat(k, 1);
                double zeta = xHat(k, 2);
                
                result.row(T * numQuadPoints + k) = (1 - xi - eta - zeta) * v0 + xi * v1 + eta * v2 + zeta * v3;
            }
        }
    } else {
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
    }
    
    return result;
}

bool Grid::isBoundaryPoint(const Eigen::VectorXd& p) {
    const double eps = 1e-6;
    
    if (is3DGrid) {
        return (p(0) <= xlow + eps || p(0) >= xhigh - eps || 
                p(1) <= ylow + eps || p(1) >= yhigh - eps ||
                p(2) <= zlow + eps || p(2) >= zhigh - eps);
    } else {
        return (p(0) <= xlow + eps || p(0) >= xhigh - eps || 
                p(1) <= ylow + eps || p(1) >= yhigh - eps);
    }
}

std::vector<int> Grid::getInnerIndices() {
    std::vector<int> innerIndices;
    
    for (int i = 0; i < points.rows(); ++i) {
        if (!isBoundaryPoint(points.row(i))) {
            innerIndices.push_back(i);
        }
    }
    
    return innerIndices;
}

std::vector<int> Grid::getBoundaryIndices(std::function<bool(const Eigen::VectorXd&)> locator) {
    std::vector<int> boundaryIndices;
    
    for (int i = 0; i < points.rows(); ++i) {
        if (isBoundaryPoint(points.row(i)) && locator(points.row(i))) {
            boundaryIndices.push_back(i);
        }
    }
    
    return boundaryIndices;
}