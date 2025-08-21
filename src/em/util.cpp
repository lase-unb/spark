#include "spark/constants/constants.h"
#include "spark/em/poisson.h"
#include "spark/core/vec.h"

void spark::em::charge_density(double particle_weight,
                               const spark::spatial::UniformGrid<1>& ion_density,
                               const spark::spatial::UniformGrid<1>& electron_density,
                               spark::spatial::UniformGrid<1>& out) {
    auto& out_data = out.data();
    auto& ne = electron_density.data();
    auto& ni = ion_density.data();
    double k = spark::constants::e * particle_weight / ion_density.dx().x;

    for (size_t i = 0; i < out.n().x; i++) {
        out_data(i) = k * (ni(i) - ne(i));
    }
}


inline spark::core::TVec<double, 2> cartesian_to_polar(const spark::core::TVec<double, 2>& v) {
    double R = std::sqrt(v.x * v.x + v.y * v.y);
    double theta = std::atan2(v.y, v.x);
    return spark::core::TVec<double, 2>{R, theta};
}

inline spark::core::TVec<double, 2> polar_to_cartesian(const spark::core::TVec<double, 2>& v) {
    double x = v.x * std::cos(v.y);
    double y = v.x * std::sin(v.y);
    return spark::core::TVec<double, 2>{x, y};
}