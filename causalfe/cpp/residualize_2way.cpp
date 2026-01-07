#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> residualize_2way(
    py::array_t<double> y,
    py::array_t<int> unit,
    py::array_t<int> time,
    int n_unit,
    int n_time,
    int iters
) {
    auto ybuf = y.request();
    auto ubuf = unit.request();
    auto tbuf = time.request();

    int n = ybuf.size;
    std::vector<double> yt((double*)ybuf.ptr,
                            (double*)ybuf.ptr + n);

    for (int it = 0; it < iters; it++) {
        // unit FE
        std::vector<double> usum(n_unit, 0.0);
        std::vector<int> ucnt(n_unit, 0);

        for (int i = 0; i < n; i++) {
            usum[((int*)ubuf.ptr)[i]] += yt[i];
            ucnt[((int*)ubuf.ptr)[i]]++;
        }
        for (int i = 0; i < n; i++) {
            yt[i] -= usum[((int*)ubuf.ptr)[i]] /
                     ucnt[((int*)ubuf.ptr)[i]];
        }

        // time FE
        std::vector<double> tsum(n_time, 0.0);
        std::vector<int> tcnt(n_time, 0);

        for (int i = 0; i < n; i++) {
            tsum[((int*)tbuf.ptr)[i]] += yt[i];
            tcnt[((int*)tbuf.ptr)[i]]++;
        }
        for (int i = 0; i < n; i++) {
            yt[i] -= tsum[((int*)tbuf.ptr)[i]] /
                     tcnt[((int*)tbuf.ptr)[i]];
        }
    }

    return py::array_t<double>(yt.size(), yt.data());
}
