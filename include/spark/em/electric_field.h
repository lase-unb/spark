#ifndef ELECTRIC_FIELD_H
#define ELECTRIC_FIELD_H

#include "spark/core/matrix.h"
#include "spark/spatial/grid.h"
#include "spark/core/vec.h"

namespace spark::em {

template <unsigned N>
void electric_field(const spatial::UniformGrid<N>& phi, core::TMatrix<core::Vec<N>, N>& out);

void electric_field_cylindrical(const spatial::UniformGrid<2>& phi, core::TMatrix<core::Vec<2>, 2>& out);

}  // namespace spark::em

#endif  // ELECTRIC_FIELD_H