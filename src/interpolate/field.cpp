#include "spark/interpolate/field.h"

#include <spark/spatial/grid.h>

#include "spark/particle/species.h"

using namespace spark;
using namespace spark::core;

namespace {
    template <typename T>
    T clamp(T min, T max, T d) {
        const double t = d < min ? min : d;
        return t > max ? max : t;
    }
    }  // namespace

namespace spark::interpolate{

template <typename T, unsigned NV>
void field_at_particles(const spatial::TUniformGrid<T, 1>& field,
                        const particle::ChargedSpecies<1, NV>& species,
                        TMatrix<T, 1>& out) {
    const size_t n = species.n();
    out.resize({n});

    auto* x = species.x();

    const double dx = field.dx().x;
    const auto& f = field.data();
    const double mdx = 1.0 / dx;

    for (size_t i = 0; i < n; i++) {
        const double xp_dx = x[i].x * mdx;
        const double il = floor(xp_dx);
        const auto ils = static_cast<size_t>(il);

        out[i] = f[ils] * (il + 1.0 - xp_dx) + f[ils + 1] * (xp_dx - il);
    }
}

template <typename T, unsigned NV>
void field_at_particles(const spatial::TUniformGrid<T, 2>& field,
                        const particle::ChargedSpecies<2, NV>& species,
                        TMatrix<T, 1>& out) {
    const size_t n = species.n();
    out.resize({n});

    auto* x = species.x();

    const auto dx = field.dx();
    const auto& f = field.data();
    const auto mdx = 1.0 / dx;

    for (size_t i = 0; i < n; ++i) {
        const auto xp = x[i] * mdx;
        const auto xmesh_lower_left = xp.template apply<std::floor>();
        const auto xmesh_upper_right = xmesh_lower_left + 1.0;
        const auto idx = xmesh_lower_left.template to<size_t>();

        const double a1 = (xmesh_upper_right.x - xp.x) * (xmesh_upper_right.y - xp.y);
        const double a2 = (xp.x - xmesh_lower_left.x) * (xmesh_upper_right.y - xp.y);
        const double a3 = (xp.x - xmesh_lower_left.x) * (xp.y - xmesh_lower_left.y);
        const double a4 = (xmesh_upper_right.x - xp.x) * (xp.y - xmesh_lower_left.y);

        out[i] = a1 * f(idx.x, idx.y) + a2 * f(idx.x + 1, idx.y) + a3 * f(idx.x + 1, idx.y + 1) +
                 a4 * f(idx.x, idx.y + 1);
    }
}

template <typename T, unsigned NV>
void field_at_particles_cylindrical(const spatial::TUniformGrid<T, 2>& field,
                                    const particle::ChargedSpecies<2, NV>& species,
                                    core::TMatrix<T, 1>& out) {
    const size_t n_particles = species.n();
    out.resize({n_particles});

    auto* x = species.x();

    const auto dx = field.dx();
    const double dz = dx.x;
    const double dr = dx.y;
    const double mdz = 1.0 / dz;
    const double mdr = 1.0 / dr;

    const auto& f = field.data();
    const auto grid_dims = field.n();
    const int nz = grid_dims.x;
    const int nr = grid_dims.y;

    const double z_grid_start = field.l().x;

    const double r_grid_start = field.l().y;


    for (size_t p_idx = 0; p_idx < n_particles; ++p_idx) {
        const double zp = x[p_idx].x;
        const double rp = x[p_idx].y;

        const double z_idx_frac = (zp - z_grid_start) * mdz;
        const double r_idx_frac = (rp - r_grid_start) * mdr;

        const double iz_floor = std::floor(z_idx_frac);
        const double ir_floor = std::floor(r_idx_frac);

        const int iz = static_cast<int>(iz_floor);
        const int ir = static_cast<int>(ir_floor);

        const double z_local = z_idx_frac - iz_floor;
        const double r_local = r_idx_frac - ir_floor;

        const int iz_clamped = clamp(0, nz - 1, iz);
        const int ir_clamped = clamp(0, nr - 1, ir);

        const int iz_p1_clamped = clamp(0, nz - 1, iz + 1);
        const int ir_p1_clamped = clamp(0, nr - 1, ir + 1);

        const double r_node_ir = r_grid_start + ir_clamped * dr;

        T interpolated_field =
             f(iz_clamped, ir_clamped) * (1.0 - z_local) * (1.0 - r_local) +
             f(iz_p1_clamped, ir_clamped) * z_local * (1.0 - r_local) +
             f(iz_clamped, ir_p1_clamped) * (1.0 - z_local) * r_local +
             f(iz_p1_clamped, ir_p1_clamped) * z_local * r_local;

        if (rp < r_grid_start + 1e-15) {
             if (iz_clamped >= 0 && iz_clamped < nz) {
                 interpolated_field = f(iz_clamped, 0);
             } else if (nz > 0) {
                 size_t nearest_iz = clamp(0, nz - 1, iz_clamped);
                 interpolated_field = f(nearest_iz, 0);
             } else {
                 interpolated_field = {};
             }
        }


        out[p_idx] = interpolated_field;
    }
}

}  // namespace

template <typename T, unsigned NX, unsigned NV>
void spark::interpolate::field_at_particles(const spark::spatial::TUniformGrid<T, NX>& field,
                                            const spark::particle::ChargedSpecies<NX, NV>& species,
                                            core::TMatrix<T, 1>& out) {
    spark::interpolate::field_at_particles(field, species, out);
}
template <typename T, unsigned NX>
T interpolate::field_at_position(const spatial::TUniformGrid<T, NX>& field, const Vec<NX>& pos) {
    const auto& f = field.data();

    const auto xp = pos / field.dx();
    const auto xmesh_lower_left = xp.template apply<std::floor>();
    const auto xmesh_upper_right = xmesh_lower_left + 1.0;
    const auto idx = xmesh_lower_left.template to<size_t>();

    const double a1 = (xmesh_upper_right.x - xp.x) * (xmesh_upper_right.y - xp.y);
    const double a2 = (xp.x - xmesh_lower_left.x) * (xmesh_upper_right.y - xp.y);
    const double a3 = (xp.x - xmesh_lower_left.x) * (xp.y - xmesh_lower_left.y);
    const double a4 = (xmesh_upper_right.x - xp.x) * (xp.y - xmesh_lower_left.y);

    return a1 * f(idx.x, idx.y) + a2 * f(idx.x + 1, idx.y) + a3 * f(idx.x + 1, idx.y + 1) +
           a4 * f(idx.x, idx.y + 1);
}

template void interpolate::field_at_particles(const spatial::UniformGrid<1>& field,
                                              const particle::ChargedSpecies<1, 1>& species,
                                              Matrix<1>& out);
template void interpolate::field_at_particles(const spatial::UniformGrid<1>& field,
                                              const particle::ChargedSpecies<1, 3>& species,
                                              Matrix<1>& out);
template void interpolate::field_at_particles(const spatial::UniformGrid<2>& field,
                                              const particle::ChargedSpecies<2, 3>& species,
                                              Matrix<1>& out);
template void interpolate::field_at_particles(const spatial::TUniformGrid<Vec<1>, 1>& field,
                                              const particle::ChargedSpecies<1, 3>& species,
                                              TMatrix<Vec<1>, 1>& out);
template void interpolate::field_at_particles(const spatial::TUniformGrid<Vec<2>, 2>& field,
                                              const particle::ChargedSpecies<2, 3>& species,
                                              TMatrix<Vec<2>, 1>& out);

template double interpolate::field_at_position(const spatial::UniformGrid<2>& field,
                                               const core::Vec<2>& pos);
template void interpolate::field_at_particles_cylindrical<double, 3>(const spatial::TUniformGrid<double, 2>& field, const particle::ChargedSpecies<2, 3>& species, core::TMatrix<double, 1>& out);
template void interpolate::field_at_particles_cylindrical(const spatial::TUniformGrid<Vec<2>, 2>& field,
    const particle::ChargedSpecies<2, 3>& species,
    TMatrix<Vec<2>, 1>& out);
