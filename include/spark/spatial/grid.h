#pragma once

#include <vector>

#include "spark/core/matrix.h"
#include "spark/core/vec.h"

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

template <unsigned N>
class AverageGrid {
public:
    AverageGrid() = default;
    AverageGrid(const UniformGrid<N>& grid) : m_average_grid(grid) { m_average_grid.set(0); }

    void add(const UniformGrid<1>& grid) {
        auto& av = m_average_grid.data();
        auto& gr = grid.data();

        const auto count = m_average_grid.n_total();
        for (size_t i = 0; i < count; i++) {
            av(i) = (av(i) * static_cast<double>(m_count) / static_cast<double>(m_count + 1)) +
                    (gr(i) / static_cast<double>(m_count + 1));
        }

        m_count++;
    }

    auto& get() { return m_average_grid.data(); }

private:
    UniformGrid<N> m_average_grid;
    size_t m_count = 0;
};

}  // namespace spark::spatial