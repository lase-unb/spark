#pragma once

#include <functional>
#include <vector>

#include "spark/core/vec.h"
#include "spark/spatial/grid.h"

namespace spark::em {

class ThomasPoissonSolver1D {
public:
    ThomasPoissonSolver1D(size_t n, double dx);
    void solve(const std::vector<double>& density, std::vector<double>& out, double v0, double v1);

private:
    double m_dx;
    size_t m_n;
};

void charge_density(double particle_weight,
                    const spark::spatial::UniformGrid<1>& ion_density,
                    const spark::spatial::UniformGrid<1>& electron_density,
                    spark::spatial::UniformGrid<1>& out);

enum class CellType : uint8_t { Internal, External, BoundaryDirichlet, BoundaryNeumann };

class StructPoissonSolver2D {
public:
    struct Region {
        CellType region_type = CellType::Internal;
        core::IntVec<2> lower_left, upper_right;
        std::function<double()> input;
    };

    struct DomainProp {
        core::IntVec<2> extents;
        core::Vec<2> dx;
    };

    explicit StructPoissonSolver2D(const DomainProp& prop, const std::vector<Region>& regions);
    void solve(core::Matrix<2>& out, const core::Matrix<2>& rho);
    ~StructPoissonSolver2D();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace spark::em