#include "kn/interpolate/field.h"
#include "kn/particle/species.h"

namespace {
    template <unsigned NV>
    void field_at_particles(const kn::spatial::UniformGrid &field, kn::particle::ChargedSpecies<1, NV> &species) {
        
        size_t n = species.n();
        auto* f = species.f();
        auto* x = species.x();

        const double dx = field.dx();
        double* e = field.data_ptr();

        for(size_t i = 0; i < n; i++) {
            double xp = x[i].x;
            size_t il = static_cast<size_t>(floor(xp / dx));
            double xl = static_cast<double>(il) * dx;
            double xr = static_cast<double>(il + 1) * dx;

            f[i] = {e[il]     * (xr - xp) / dx +
                    e[il + 1] * (xp - xl) / dx};
        }
    }
}

template<unsigned NX, unsigned NV> 
void kn::interpolate::field_at_particles(const kn::spatial::UniformGrid &field, kn::particle::ChargedSpecies<NX, NV> &species) {
    ::field_at_particles(field, species);
}

template void kn::interpolate::field_at_particles(const kn::spatial::UniformGrid &field, kn::particle::ChargedSpecies<1, 1> &species);
template void kn::interpolate::field_at_particles(const kn::spatial::UniformGrid &field, kn::particle::ChargedSpecies<1, 3> &species);
