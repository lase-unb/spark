#pragma once

#include <vector>

#include "spark/core/matrix.h"
#include "spark/core/vec.h"
#include "spark/constants/constants.h"

namespace spark::spatial {

template <unsigned N>
struct GridProp {
    core::Vec<N> l;
    core::Vec<N> dx;
    core::ULongVec<N> n;
};

template <typename T, unsigned N>
class TUniformGrid {
public:
    TUniformGrid() = default;
    TUniformGrid(const core::Vec<N>& l, const core::ULongVec<N>& n) {
        data_.resize(n);
        prop_.l = l;
        prop_.dx = l / (n.template to<double>() - 1.0);
        prop_.n = n;

        set({0});
    }

    TUniformGrid(const GridProp<N>& prop) : prop_(prop) {
        data_.resize(prop_.n);
        set({0});
    }

    void set(T v) { data_.fill(v); }
    auto& data() { return data_; }
    const auto& data() const { return data_; }
    auto* data_ptr() const { return const_cast<T*>(data_.data_ptr()); }

    auto n_total() const { return data_.count(); }
    auto n() const { return data_.size(); }
    auto l() const { return prop_.l; }
    auto dx() const { return prop_.dx; }
    auto prop() const { return prop_; }

    void apply(T mul, T add) {
        const auto count = data_.count();
        for (size_t i = 0; i < count; i++) {
            data_[i] = data_[i] * mul + add;
        }
    }

private:
    GridProp<N> prop_;
    core::TMatrix<T, N> data_;
};

template <unsigned N>
using UniformGrid = TUniformGrid<double, N>;

template <typename T>
class TCylindricalGrid {
public:
    TCylindricalGrid() = default;

    TCylindricalGrid(const core::Vec<2>& l, const core::ULongVec<2>& n) {
        prop_.l = l;
        prop_.n = n;
        prop_.dx = l / (n.template to<double>() - 1.0);
        data_.resize(n);
        set({0});
    }

    TCylindricalGrid(const GridProp<2>& prop) : prop_(prop) {
        data_.resize(prop_.n);
        set({0});
    }

    void set(T v) { data_.fill(v); }
    auto& data() { return data_; }
    const auto& data() const { return data_; }
    auto* data_ptr() const { return const_cast<T*>(data_.data_ptr()); }
    auto n_total() const { return data_.count(); }
    auto n() const { return data_.size(); }
    auto l() const { return prop_.l; }
    auto dx() const { return prop_.dx; }
    auto prop() const { return prop_; }

    T getRadius(size_t i_r) const {
        return static_cast<T>(i_r) * prop_.dx.x;
    }

    T getMidRadius(size_t i_r) const {
        return (static_cast<T>(i_r) + static_cast<T>(0.5)) * prop_.dx.x;
    }

    T getCellVolume(size_t i_r, size_t i_z) const {
        T dz = prop_.dx.y;
        T r_outer = getMidRadius(i_r); 
        T r_inner;

        if (i_r == 0) {
            r_inner = static_cast<T>(0.0);
        } else {
            r_inner = getMidRadius(i_r - 1);
        }

        return constants::pi * (r_outer * r_outer - r_inner * r_inner) * dz;
    }

    T cellArea(size_t i_r, size_t i_z) const {
        T dr = prop_.dx.x;
        T dz = prop_.dx.y;
        T r_center = (static_cast<T>(i_r) + static_cast<T>(0.5)) * dr;
        return r_center * dr * dz;
    }

private:
    GridProp<2> prop_;
    core::TMatrix<T, 2> data_;
};

using CylindricalGrid = TCylindricalGrid<double>;

template <unsigned N>
class AverageGrid {
public:
    AverageGrid() = default;
    AverageGrid(const UniformGrid<N>& grid) : m_average_grid(grid) { m_average_grid.set(0); }

    void add(const UniformGrid<N>& grid) {
        auto* av = m_average_grid.data_ptr();
        const auto* gr = grid.data_ptr();

        const auto count = m_average_grid.n_total();
        for (size_t i = 0; i < count; i++) {
            av[i] = (av[i] * static_cast<double>(m_count) / static_cast<double>(m_count + 1)) +
                    (gr[i] / static_cast<double>(m_count + 1));
        }

        m_count++;
    }

    auto& get() { return m_average_grid.data(); }

private:
    UniformGrid<N> m_average_grid;
    size_t m_count = 0;
};

}  // namespace spark::spatial