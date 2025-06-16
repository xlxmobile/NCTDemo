#include <iostream>
#include <cmath>
#include "grid.h"
#include "basis.h"
#include "quadrule.h"
#include "problem.h"

void run2D() {
    std::cout << "Running 2D simulation..." << std::endl;
    
    // 2D参数设置
    double a = 1.0;
    double c = 0.2;
    int Nx = 10, Ny = 10;
    int N = 5;
    double tau = 0.5;
    double xlow = -1, xhigh = 1, ylow = -1, yhigh = 1;
    
    // 2D边界条件
    auto dirichlet_locations_2d = [](const Eigen::VectorXd& p) -> bool {
        return (std::abs(p(1) + 1) < 1e-6 || std::abs(p(1) - 1) < 1e-6);
    };
    
    auto dirichlet_values_2d = [](const Eigen::VectorXd& p) -> double {
        return 1.0;
    };
    
    // 创建2D网格
    Grid grid_instance(xlow, xhigh, ylow, yhigh, Nx, Ny);
    
    // 创建2D基函数
    Basis basis_instance(false);
    GlobalBasis global_basis(grid_instance, basis_instance);
    
    // 创建2D积分规则
    QuadratureRule quadrature_instance(2, false);
    
    // 创建2D问题
    StationaryProblem prob(global_basis, quadrature_instance, 
                          dirichlet_locations_2d, dirichlet_values_2d);
    
    // 设置扩散和反应系数
    prob.setDiffusion(a);
    prob.setReaction(c);
    
    // 2D初始条件
    auto initial_condition_2d = [](const Eigen::VectorXd& p) -> double {
        return 2 - p(1) * p(1);
    };
    
    Eigen::VectorXd initial_condition_vector(grid_instance.getPoints().rows());
    for (int i = 0; i < grid_instance.getPoints().rows(); ++i) {
        initial_condition_vector(i) = initial_condition_2d(grid_instance.getPoints().row(i));
    }
    
    prob.setSolution(initial_condition_vector);
    
    // 2D时间步进
    for (int n = 0; n < N; ++n) {
        // 2D源项
        auto f_2d = [n, tau](const Eigen::VectorXd& p) -> double {
            return std::sin(M_PI / 2 * (n + 1) * tau);
        };
        
        prob.resetSystemVector();
        prob.resetSystemMatrix();
        
        prob.addSource(f_2d);
        prob.assemble();
        prob.assembleBoundaryConditions(dirichlet_values_2d);
        prob.solve();
        prob.show();
        prob.setSolution(prob.getSolution());
        
        std::cout << "2D Time step " << n + 1 << " completed." << std::endl;
    }
}

void run3D() {
    std::cout << "Running 3D simulation..." << std::endl;
    
    // 3D参数设置
    double a = 1.0;
    double c = 0.2;
    int Nx = 6, Ny = 6, Nz = 6;  // 减少网格大小以提高计算速度
    int N = 3;
    double tau = 0.5;
    double xlow = -1, xhigh = 1, ylow = -1, yhigh = 1, zlow = -1, zhigh = 1;
    
    // 3D边界条件
    auto dirichlet_locations_3d = [](const Eigen::VectorXd& p) -> bool {
        return (std::abs(p(2) + 1) < 1e-6 || std::abs(p(2) - 1) < 1e-6);
    };
    
    auto dirichlet_values_3d = [](const Eigen::VectorXd& p) -> double {
        return 1.0;
    };
    
    // 创建3D网格
    Grid grid_instance(xlow, xhigh, ylow, yhigh, zlow, zhigh, Nx, Ny, Nz);
    
    // 创建3D基函数
    Basis basis_instance(true);
    GlobalBasis global_basis(grid_instance, basis_instance);
    
    // 创建3D积分规则
    QuadratureRule quadrature_instance(2, true);
    
    // 创建3D问题
    StationaryProblem prob(global_basis, quadrature_instance, 
                          dirichlet_locations_3d, dirichlet_values_3d);
    
    // 设置扩散和反应系数
    prob.setDiffusion(a);
    prob.setReaction(c);
    
    // 3D初始条件
    auto initial_condition_3d = [](const Eigen::VectorXd& p) -> double {
        return 2 - p(2) * p(2);
    };
    
    Eigen::VectorXd initial_condition_vector(grid_instance.getPoints().rows());
    for (int i = 0; i < grid_instance.getPoints().rows(); ++i) {
        initial_condition_vector(i) = initial_condition_3d(grid_instance.getPoints().row(i));
    }
    
    prob.setSolution(initial_condition_vector);
    
    // 3D时间步进
    for (int n = 0; n < N; ++n) {
        // 3D源项
        auto f_3d = [n, tau](const Eigen::VectorXd& p) -> double {
            return std::sin(M_PI / 2 * (n + 1) * tau);
        };
        
        prob.resetSystemVector();
        prob.resetSystemMatrix();
        
        prob.addSource(f_3d);
        prob.assemble();
        prob.assembleBoundaryConditions(dirichlet_values_3d);
        prob.solve();
        prob.show();
        prob.setSolution(prob.getSolution());
        
        std::cout << "3D Time step " << n + 1 << " completed." << std::endl;
    }
}

int main() {
    std::cout << "FEM Solver with 2D/3D Support" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // 运行2D模拟
    run2D();
    
    std::cout << std::endl;
    
    // 运行3D模拟
    run3D();
    
    return 0;
}