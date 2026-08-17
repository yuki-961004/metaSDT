#ifndef QUADRATURE_HPP
#define QUADRATURE_HPP

#include <cstddef>
#include <vector>

// 32-point Gauss-Legendre quadrature for deterministic 1D numerical integration
namespace Quadrature {

// 16 positive roots and weights on [-1, 1]
constexpr double NODES_32[16] = {
    0.048307665687738338,
    0.14447196158279643,
    0.23928736225213709,
    0.33186860228212767,
    0.42135127613063539,
    0.50689990893222947,
    0.5877157572407623,
    0.66304426693021523,
    0.73218211874028971,
    0.79448379596794227,
    0.84936761373256986,
    0.89632115576605209,
    0.93490607593773967,
    0.96476225558750639,
    0.98561151154526838,
    0.99726386184948157
};

constexpr double WEIGHTS_32[16] = {
    0.09654008851472759,
    0.095638720079274653,
    0.093844399080804414,
    0.091173878695763641,
    0.0876520930044037,
    0.083311924226946471,
    0.078193895787070158,
    0.072345794108848491,
    0.065822222776361336,
    0.058684093478535787,
    0.050998059262376251,
    0.042835898022227203,
    0.034273862913020543,
    0.025392065309261264,
    0.016274394730904029,
    0.0070186100094744202
};

template <typename T, typename Func>
T integrate_32(Func&& f, const T& a, const T& b) {
    if (a >= b) {
        return static_cast<T>(0.0);
    }
    const T half_len = (b - a) * static_cast<T>(0.5);
    const T mid = (b + a) * static_cast<T>(0.5);

    T sum = static_cast<T>(0.0);
    for (std::size_t i = 0; i < 16; ++i) {
        const T node = static_cast<T>(NODES_32[i]);
        const T weight = static_cast<T>(WEIGHTS_32[i]);
        const T x_pos = mid + half_len * node;
        const T x_neg = mid - half_len * node;
        sum += weight * (f(x_pos) + f(x_neg));
    }
    return sum * half_len;
}

} // namespace Quadrature

#endif // QUADRATURE_HPP
