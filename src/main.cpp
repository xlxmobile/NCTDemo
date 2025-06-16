#include <iostream>
#include <cmath>
#include "grid.h"
#include "basis.h"
#include "quadrule.h"
#include "problem.h"

int main() {
    // 参数设置
    double a = 1.0;
    double c = 0.2;
    int Nx = 10, Ny = 10;
    int N = 10;
    double tau = 0.5;
    double xlow = -1, xhigh = 1, ylow = -1, yhigh = 1;
    
    // 边界条件
    auto dirichlet_locations = [](double x, double y) -> bool {
        return (std::abs(y + 1) < 1e-6 || std::abs(y - 1) < 1e-6);
    };
    
    auto dirichlet_values = [](double x, double y) -> double {
        return 1.0;
    };
    
    // 创建网格
    Grid grid_instance(xlow, xhigh, ylow, yhigh, Nx, Ny);
    
    // 创建基函数
    Basis basis_instance;
    GlobalBasis global_basis(grid_instance, basis_instance);
    
    // 创建积分规则
    QuadratureRule quadrature_instance(2);
    
    // 创建问题
    StationaryProblem prob(global_basis, quadrature_instance, 
                          dirichlet_locations, dirichlet_values);
    
    // 设置扩散和反应系数
    prob.setDiffusion(a);
    prob.setReaction(c);
    
    // 初始条件
    auto initial_condition = [](double x, double y) -> double {
        return 2 - y * y;
    };
    
    Eigen::VectorXd initial_condition_vector(grid_instance.getPoints().rows());
    for (int i = 0; i < grid_instance.getPoints().rows(); ++i) {
        double x = grid_instance.getPoints()(i, 0);
        double y = grid_instance.getPoints()(i, 1);
        initial_condition_vector(i) = initial_condition(x, y);
    }
    
    prob.setSolution(initial_condition_vector);
    
    // 时间步进
    for (int n = 0; n < N; ++n) {
        // 源项
        auto f = [n, tau](double x, double y) -> double {
            return std::sin(M_PI / 2 * (n + 1) * tau);
        };
        
        prob.resetSystemVector();
        prob.resetSystemMatrix();
        
        // 添加源项
        prob.addSource(f);
        
        // 组装系统
        prob.assemble();
        
        // 组装边界条件
        prob.assembleBoundaryConditions(dirichlet_values);
        
        // 求解
        prob.solve();
        
        // 显示结果
        prob.show();
        
        // 更新解
        prob.setSolution(prob.getSolution());
        
        std::cout << "Time step " << n + 1 << " completed." << std::endl;
    }
    
    return 0;
}