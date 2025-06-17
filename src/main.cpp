#include <iostream>
#include <cmath>
#include "grid/grid.h"
#include "basis/basis.h"
#include "quadrule/quadrule.h"
#include "problem/problem.h"

void run3D() {
    std::cout << "Running 3D simulation..." << std::endl;
    
    double a = 1.0;
    double c = 0.2;
    int Nx = 6, Ny = 6, Nz = 6; 
    int N = 3;
    double tau = 0.5;
    double xlow = -1, xhigh = 1, ylow = -1, yhigh = 1, zlow = -1, zhigh = 1;
    
    auto dirichlet_locations_3d = [](const Eigen::VectorXd& p) -> bool {
        return (std::abs(p(2) + 1) < 1e-6 || std::abs(p(2) - 1) < 1e-6);
    };
    
    auto dirichlet_values_3d = [](const Eigen::VectorXd& p) -> double {
        return 1.0;
    };
    
    Grid grid_instance(xlow, xhigh, ylow, yhigh, zlow, zhigh, Nx, Ny, Nz);
    
    Basis basis_instance(true);
    GlobalBasis global_basis(grid_instance, basis_instance);
    
    QuadratureRule quadrature_instance(2, true);
    
    StationaryProblem prob(global_basis, quadrature_instance, 
                          dirichlet_locations_3d, dirichlet_values_3d);
    
    prob.setDiffusion(a);
    prob.setReaction(c);
    
    auto initial_condition_3d = [](const Eigen::VectorXd& p) -> double {
        return 2 - p(2) * p(2);
    };
    
    Eigen::VectorXd initial_condition_vector(grid_instance.getPoints().rows());
    for (int i = 0; i < grid_instance.getPoints().rows(); ++i) {
        initial_condition_vector(i) = initial_condition_3d(grid_instance.getPoints().row(i));
    }
    
    prob.setSolution(initial_condition_vector);
    
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
    std::cout << "FEM Solver with 3D Support" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // 运行3D模拟
    run3D();
    
    return 0;
}