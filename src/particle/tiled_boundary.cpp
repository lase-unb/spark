#include <cstddef>

#include "spark/core/vec.h"
#include "spark/particle/boundary.h"
#include "spark/spatial/grid.h"

using namespace spark::core;

namespace {

#define CMOD(idx, size) ((size + (idx % size)) % size)
#define CMODV(idx, size_vec) CMOD(idx.x, size_vec.x), CMOD(idx.y, size_vec.y)

constexpr size_t padding_ = 4;

int cmod(int idx, int size) {
    return ((size + (idx % size)) % size);
}

struct CollisionHit {
    IntVec<2> loc;
    Vec<2> normal;
    Vec<2> pos;
    uint8_t val;
};

// Assuming cell size is 1x1
CollisionHit grid_raycast(const spark::core::TMatrix<uint8_t, 2>& grid,
                          const Vec<2>& a,
                          const Vec<2>& b) {
    auto current_index = a.apply<std::floor>().to<int>();
    const auto end_index = b.apply<std::floor>().to<int>();
    const auto sz = grid.size().to<int>();

    if (current_index != end_index) {
        auto dir = (b - a).normalized();
        Vec<2> t{0.0, 0.0};

        IntVec<2> step = {0, 0};
        Vec<2> k = {std::abs(1.0 / dir.x), std::abs(1.0 / dir.y)};

        if (dir.x < 0) {
            step.x = -1;
            t.x = (a.x - static_cast<double>(current_index.x)) * k.x;
        } else {
            step.x = 1;
            t.x = (static_cast<double>(current_index.x + 1) - a.x) * k.x;
        }

        if (dir.y < 0) {
            step.y = -1;
            t.y = (a.y - static_cast<double>(current_index.y)) * k.y;
        } else {
            step.y = 1;
            t.y = (static_cast<double>(current_index.y + 1) - a.y) * k.y;
        }

        double distance = 0.0;
        Vec<2> normal = {0.0, 0.0};

        while (current_index != end_index) {
            if (t.x < t.y) {
                current_index.x += step.x;
                distance = t.x;
                t.x += k.x;
                normal = {-(double)step.x, 0};
            } else {
                current_index.y += step.y;
                distance = t.y;
                t.y += k.y;
                normal = {0, -(double)step.y};
            }

            if (uint8_t val = grid(CMODV(current_index, sz)))
                return {
                    .loc = current_index, .normal = normal, .pos = a + dir * distance, .val = val};
        }
    }

    return {0};
}

inline Vec<2> reflect(const Vec<2>& x, const Vec<2>& n, const Vec<2>& contact) {
    // Omitting terms:
    // 1) 2 * n.x * n.y * (contact.y - x.y) in x
    // 2) 2 * n.x * n.y * (contact.x - x.x) in y
    // Since they will always be zero if n is axis aligned
    return {2.0 * (contact.x - x.x) * n.x * n.x + x.x, 2.0 * (contact.y - x.y) * n.y * n.y + x.y};
}

inline Vec<3> reflect(const Vec<3>& v, const Vec<2>& n) {
    return {v.x * (1.0 - 2.0 * n.x * n.x), v.y * (1.0 - 2.0 * n.y * n.y), v.z};
}

}  // namespace

namespace spark::particle {

TiledBoundary2D::TiledBoundary2D(const spatial::GridProp<2>& grid_prop,
                                 const std::vector<TiledBoundary>& boundaries,
                                 double dt)
    : gprop_(grid_prop), boundaries_(boundaries), dt_(dt) {
    cells_.resize(grid_prop.n + padding_);
    for (uint8_t i = 0; i < boundaries_.size(); ++i) {
        // Add circular_mod
        add_boundary(boundaries_[i], i + 1);
    }
}

uint8_t TiledBoundary2D::cell(int i, int j) const {
    const auto sz = cells_.size().to<int>();
    return cells_(CMOD(i, sz.x), CMOD(j, sz.y));
}

uint8_t TiledBoundary2D::cell(const Vec<2>& pos) const {
    auto idx = (pos / gprop_.dx).apply<std::floor>().to<int>();
    const auto sz = cells_.size().to<int>();
    return cells_(CMOD(idx.x, sz.x), CMOD(idx.y, sz.y));
}

void TiledBoundary2D::add_boundary(const TiledBoundary& b, uint8_t id) {
    const auto imin = std::min(b.lower_left.x, b.upper_right.x);
    const auto imax = std::max(b.lower_left.x, b.upper_right.x);
    const auto jmin = std::min(b.lower_left.y, b.upper_right.y);
    const auto jmax = std::max(b.lower_left.y, b.upper_right.y);
    const auto sz = cells_.size().to<int>();

    for (int i = imin; i <= imax; ++i)
        for (int j = jmin; j <= jmax; ++j)
            cells_(CMOD(i, sz.x), CMOD(j, sz.y)) = id;
}

void TiledBoundary2D::apply(Species<2, 3>* species) {
    if (!species)
        return;

    int n = species->n();
    auto* x = species->x();
    auto* v = species->v();

    for (int i = 0; i < n; ++i) {
        auto& x1 = x[i];
        auto& v1 = v[i];
        const auto x0 = core::Vec<2>{x1.x - v1.x * dt_, x1.y - v1.y * dt_};

        auto x0_tmp = x0 / gprop_.dx;
        auto x1_tmp = x1 / gprop_.dx;

        while (true) {
            const auto [loc, normal, coll_pos, val] = grid_raycast(cells_, x0_tmp, x1_tmp);

            // val == 0 means that no boundary was found
            if (!val)
                break;

            const auto btype = boundaries_[val - 1].boundary_type;
            if (btype == BoundaryType::Absorbing) {
                // TODO(lui): Check if this is OK
                species->remove(i);
                i--;  // check ith particle again since the particle is replaced during removal
                n--;  // decrease the number of particles
                break;
            } else if (btype == BoundaryType::Specular) {
                // Specular reflection
                x1 = reflect(x1, normal, coll_pos * gprop_.dx);
                v1 = reflect(v1, normal);

                x1_tmp = x1 / gprop_.dx;
                x0_tmp = coll_pos;
            }

            // TODO(lui): Implement diffuse (Lambertian) reflection
        }
    }
}

}  // namespace spark::particle
