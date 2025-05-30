#pragma once

#include <cstdint>
#include <functional>

#include "spark/core/matrix.h"
#include "spark/core/vec.h"
#include "spark/particle/species.h"
#include "spark/spatial/grid.h"

namespace spark::particle {
template <unsigned NV>
void apply_symmetric_boundary(ChargedSpecies<1, NV>& species, double xmin, double xmax);

template <unsigned NV>
void apply_absorbing_boundary(ChargedSpecies<1, NV>& species, double xmin, double xmax);

enum class BoundaryType { Specular, Absorbing };

struct TiledBoundary {
    core::IntVec<2> lower_left, upper_right;
    BoundaryType boundary_type = BoundaryType::Absorbing;
};

class TiledBoundary2D {
public:
    TiledBoundary2D() = default;
    TiledBoundary2D(const spatial::GridProp<2>& grid_prop,
                    const std::vector<TiledBoundary>& boundaries,
                    double dt,
                    bool empty_box = false);

    // TODO(lui): remove the dependency on std::function by using templates and lambdas. However
    // this means that we need to move all the code to the header.
    using Callback = std::function<void(int, core::Vec<2>, core::Vec<3>)>;

    void apply(Species<2, 3>& species, const Callback& collision_callback = nullptr);
    uint8_t cell(int i, int j) const;
    uint8_t cell(const core::Vec<2>& pos) const;

    const std::vector<TiledBoundary>& boundaries() const { return boundaries_; }

private:
    void add_boundary(const TiledBoundary& boundary, uint8_t id);
    void set_distance_cells();
    int bfs_closest_boundary(int i, int j);
    int sx_ = 0, sy_ = 0;

    core::TMatrix<uint8_t, 2> distance_cells_;
    core::TMatrix<uint8_t, 2> cells_;
    spatial::GridProp<2> gprop_;
    std::vector<TiledBoundary> boundaries_;
    double dt_ = 0.0;
    bool empty_box_ = false;
};

}  // namespace spark::particle
