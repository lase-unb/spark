#pragma once

#include "spark/core/matrix.h"
#include "spark/core/vec.h"
#include "spark/particle/species.h"

namespace spark::particle {

template <unsigned NX, unsigned NV>
void move_particles(ChargedSpecies<NX, NV>& species,
                    const core::TMatrix<core::Vec<NX>, 1>& force,
                    double dt);

template <unsigned NV>
void move_particles_cylindrical(ChargedSpecies<2, NV>& species,
                                const core::TMatrix<core::Vec<2>, 1>& force,
                                double dt);

} // namespace spark::particle