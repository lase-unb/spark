#include "spark/interpolate/weight.h"

#include "spark/particle/species.h"
#include "spark/spatial/grid.h"

namespace {
template <unsigned NV>
void weight_to_grid(const spark::particle::ChargedSpecies<1, NV>& species,
                    spark::spatial::UniformGrid<1>& out) {
    const size_t n = species.n();
    auto* x = species.x();

    out.set(0.0);

    const double dx = out.dx().x;
    auto& g = out.data().data();
    const double mdx = 1.0 / dx;

    for (size_t i = 0; i < n; i++) {
        const double xp_dx = x[i].x * mdx;
        const double il = floor(xp_dx);
        const size_t ils = static_cast<size_t>(il);

        g[ils] += il + 1.0 - xp_dx;
        g[ils + 1] += xp_dx - il;
    }

    g.front() *= 2.0;
    g.back() *= 2.0;
}

// Template specialization for 2D - Based on Birdsall and Langdon (1991), Chapter 14 Section 14.2
template <unsigned NV>
void weight_to_grid(const spark::particle::ChargedSpecies<2, NV>& species,
                    spark::spatial::UniformGrid<2>& out) {
    const size_t n = species.n();
    auto* x = species.x();

    out.set(0.0);

    const double dx = out.dx().x;
    const double dy = out.dx().y;
    auto& grid_data = out.data();
    const double mdx = 1.0 / dx;
    const double mdy = 1.0 / dy;

    for (size_t i = 0; i < n; i++) {
        const double xp_dx = x[i].x * mdx;
        const double yp_dy = x[i].y * mdy;

        const size_t j = static_cast<size_t>(floor(xp_dx));
        const size_t k = static_cast<size_t>(floor(yp_dy));

        const double x_local = xp_dx - static_cast<double>(j); 
        const double y_local = yp_dy - static_cast<double>(k); 

        const double w_jk = (1.0 - x_local) * (1.0 - y_local);
        const double w_j1k = x_local * (1.0 - y_local);
        const double w_jk1 = (1.0 - x_local) * y_local;
        const double w_j1k1 = x_local * y_local;

        grid_data(j, k) += w_jk;
        grid_data(j + 1, k) += w_j1k;
        grid_data(j, k + 1) += w_jk1;
        grid_data(j + 1, k + 1) += w_j1k1;
    }
}
}  // namespace

template <class GridType, unsigned NX, unsigned NV>
void spark::interpolate::weight_to_grid(const spark::particle::ChargedSpecies<NX, NV>& species,
                                        GridType& out) {
    ::weight_to_grid(species, out);
}

template void spark::interpolate::weight_to_grid(
    const spark::particle::ChargedSpecies<1, 1>& species,
    spark::spatial::UniformGrid<1>& out);
template void spark::interpolate::weight_to_grid(
    const spark::particle::ChargedSpecies<1, 3>& species,
    spark::spatial::UniformGrid<1>& out);
template void spark::interpolate::weight_to_grid(
    const spark::particle::ChargedSpecies<2, 1>& species,
    spark::spatial::UniformGrid<2>& out);
template void spark::interpolate::weight_to_grid(
    const spark::particle::ChargedSpecies<2, 3>& species,
    spark::spatial::UniformGrid<2>& out);
