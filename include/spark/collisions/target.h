#pragma once

#include <spark/interpolate/field.h>
#include <spark/spatial/grid.h>

#include <algorithm>

#include "spark/core/vec.h"

namespace spark::collisions {

template <unsigned NX, unsigned NV>
class Target {
public:
    virtual double dens_at(const core::Vec<NX>& pos) = 0;
    virtual double dens_max() = 0;
    virtual double temperature() = 0;
    virtual ~Target() = default;
};

template <unsigned NX, unsigned NV>
class StaticUniformTarget : public Target<NX, NV> {
public:
    StaticUniformTarget(double density, double temperature)
        : density_(density), temperature_(temperature) {}

    double dens_at(const core::Vec<NX>& pos) override { return density_; }
    double dens_max() override { return density_; }
    double temperature() override { return temperature_; }

private:
    double density_ = 0.0;
    double temperature_ = 0.0;
};

template <unsigned NX, unsigned NV>
class StaticFieldTarget final : public Target<NX, NV> {
public:
    StaticFieldTarget(const spatial::UniformGrid<NX>& density, const double temperature)
        : temperature_(temperature), field_(density) {
        field_max_ = *std::max_element(field_.data().data().begin(), field_.data().data().end());
    }

    double dens_at(const core::Vec<NX>& pos) override {
        return interpolate::field_at_position(field_, pos);
    }
    double dens_max() override { return field_max_; }
    double temperature() override { return temperature_; }

private:
    double temperature_ = 0.0;
    double field_max_ = 0.0;
    spatial::UniformGrid<NX> field_;
};

}  // namespace spark::collisions
