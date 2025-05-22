#include "spark/em/electric_field.h"

using namespace spark;

namespace {
template <typename T>
T clamp(T min, T max, T d) {
    const double t = d < min ? min : d;
    return t > max ? max : t;
}
}  // namespace

template <>
void em::electric_field<1>(const spatial::UniformGrid<1>& phi,
                           core::TMatrix<core::Vec<1>, 1>& out) {
    out.resize(phi.n());

    const auto n = phi.n().x;
    const double k = -1.0 / (2.0 * phi.dx().x);

    const auto* p = phi.data_ptr();
    auto* ef = out.data_ptr();

    for (int i = 1; i < n - 1; ++i) {
        ef[i].x = k * (p[i + 1] - p[i - 1]);
    }

    ef[0].x = 2.0 * ef[1].x - ef[2].x;
    ef[n - 1].x = 2.0 * ef[n - 2].x - ef[n - 3].x;
    // eout[0]   = - (phi[1]   - phi[0])   / dx;
    // eout[n-1] = - (phi[n-1] - phi[n-2]) / dx;
}

template <>
void em::electric_field<2>(const spatial::UniformGrid<2>& phi,
                           core::TMatrix<core::Vec<2>, 2>& out) {
    out.resize(phi.n());
    const auto [nx, ny] = out.size().to<int>();
    const auto& phi_mat = phi.data();
    const auto [kx, ky] = -1.0 / phi.dx();

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int j1 = clamp(0, ny - 1, j + 1);
            const int j0 = clamp(0, ny - 1, j - 1);
            const int i1 = clamp(0, nx - 1, i + 1);
            const int i0 = clamp(0, nx - 1, i - 1);

            out(i, j) = {kx * (phi_mat(i1, j) - phi_mat(i0, j)) / static_cast<double>(i1 - i0),
                         ky * (phi_mat(i, j1) - phi_mat(i, j0)) / static_cast<double>(j1 - j0)};
        }
    }

    for (int i = 0; i < nx; ++i) {
        out(i, 0).y = 2.0 * out(i, 1).y - out(i, 2).y;
        out(i, ny - 1).y = 2.0 * out(i, ny - 2).y - out(i, ny - 3).y;
    }

    for (int j = 0; j < ny; ++j) {
        out(0, j).x = 2.0 * out(1, j).x - out(2, j).x;
        out(nx - 1, j).x = 2.0 * out(nx - 2, j).x - out(nx - 3, j).x;
    }
}