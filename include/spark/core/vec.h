#pragma once

#include <array>
#include <cmath>

namespace spark::core {
template <typename T, unsigned N>
struct TVec {
    T norm() const;
    TVec normalized() const;
};

template <typename T>
struct TVec<T, 1> {
    T x{};
    T norm() const { return x; }
    T sum() const { return x; }
    T mul() const { return x; }
    TVec normalized() const { return TVec{x / norm()}; }

    template <typename G>
    TVec<G, 1> to() const {
        return TVec<G, 1>{static_cast<G>(x)};
    }

    std::array<T, 1> arr() { return {x}; }

    template <auto Func(T)->T>
    TVec apply() const {
        return {Func(x)};
    }
};

template <typename T>
struct TVec<T, 2> {
    T x{}, y{};
    T norm() const { return std::sqrt(x * x + y * y); }
    T sum() const { return x + y; }
    T mul() const { return x * y; }
    TVec normalized() const {
        const T n = norm();
        return TVec{x / n, y / n};
    }

    template <typename G>
    TVec<G, 2> to() const {
        return TVec<G, 2>{static_cast<G>(x), static_cast<G>(y)};
    }

    std::array<T, 2> arr() { return {x, y}; }

    template <auto Func(T)->T>
    TVec apply() const {
        return {Func(x), Func(y)};
    }
};

template <typename T>
struct TVec<T, 3> {
    T x{}, y{}, z{};
    T norm() const { return std::sqrt(x * x + y * y + z * z); }
    T sum() const { return x + y + z; }
    T mul() const { return x * y * z; }
    TVec normalized() const {
        const T n = norm();
        return TVec{x / n, y / n, z / n};
    }

    template <typename G>
    TVec<G, 3> to() const {
        return TVec<G, 3>{static_cast<G>(x), static_cast<G>(y), static_cast<G>(z)};
    }

    std::array<T, 3> arr() { return {x, y, z}; }

    template <auto Func(T)->T>
    TVec apply() const {
        return {Func(x), Func(y), Func(z)};
    }
};

template <typename T, unsigned N>
inline bool operator==(const TVec<T, N>& a, const TVec<T, N>& b) {
    if constexpr (N == 1) {
        return a.x == b.x;
    } else if constexpr (N == 2) {
        return (a.x == b.x) && (a.y == b.y);
    } else if constexpr (N == 3) {
        return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
    }
}

template <typename T, unsigned N>
inline bool operator!=(const TVec<T, N>& a, const TVec<T, N>& b) {
    return !(a == b);
}

#define DECLARE_VEC_OPS(OP)                                                   \
    template <typename T, unsigned N>                                         \
    inline TVec<T, N> operator OP(const T & a, const TVec<T, N>& b) {         \
        if constexpr (N == 1) {                                               \
            return {a OP b.x};                                                \
        } else if constexpr (N == 2) {                                        \
            return {a OP b.x, a OP b.y};                                      \
        } else if constexpr (N == 3) {                                        \
            return {a OP b.x, a OP b.y, a OP b.z};                            \
        } else {                                                              \
            return {};                                                        \
        }                                                                     \
    }                                                                         \
    template <typename T, unsigned N>                                         \
    inline TVec<T, N> operator OP(const TVec<T, N>& b, const T & a) {         \
        if constexpr (N == 1) {                                               \
            return {b.x OP a};                                                \
        } else if constexpr (N == 2) {                                        \
            return {b.x OP a, b.y OP a};                                      \
        } else if constexpr (N == 3) {                                        \
            return {b.x OP a, b.y OP a, b.z OP a};                            \
        } else {                                                              \
            return {};                                                        \
        }                                                                     \
    }                                                                         \
    template <typename T, unsigned N>                                         \
    inline TVec<T, N> operator OP(const TVec<T, N>& a, const TVec<T, N>& b) { \
        if constexpr (N == 1) {                                               \
            return {a.x OP b.x};                                              \
        } else if constexpr (N == 2) {                                        \
            return {a.x OP b.x, a.y OP b.y};                                  \
        } else if constexpr (N == 3) {                                        \
            return {a.x OP b.x, a.y OP b.y, a.z OP b.z};                      \
        } else {                                                              \
            return {};                                                        \
        }                                                                     \
    }

DECLARE_VEC_OPS(+)
DECLARE_VEC_OPS(-)
DECLARE_VEC_OPS(/)
DECLARE_VEC_OPS(*)

#undef DECLARE_VEC_OPS

template <unsigned N>
using Vec = TVec<double, N>;

template <unsigned N>
using IntVec = TVec<int, N>;

template <unsigned N>
using ULongVec = TVec<size_t, N>;

}  // namespace spark::core