#pragma once

#include "spark/core/matrix.h"
#include "spark/core/vec.h"
#include "spark/particle/species.h"

namespace spark::particle {

template <unsigned NX, unsigned NV>
void move_particles(ChargedSpecies<NX, NV>& species,
                    const core::TMatrix<core::Vec<NX>, 1>& force,
                    double dt);
}