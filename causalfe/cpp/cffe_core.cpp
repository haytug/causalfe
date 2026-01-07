/**
 * CFFE Core C++ Implementation
 * 
 * Three core functions:
 * 1. residualize_2way - Two-way FE residualization via iterative demeaning
 * 2. estimate_tau - IV-style CATE estimator: Σ D̃Ỹ / Σ D̃²
 * 3. split_score - τ-heterogeneity score for splitting
 */

#include <vector>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

/**
 * Two-way fixed effects residualization via iterative demeaning.
 * Converges to within-transformation after ~3-5 iterations.
 */
py::array_t<double> residualize_2way(
    py::array_t<double> y,
    py::array_t<int> unit,
    py::array_t<int> time,
    int n_unit,
    int n_time,
    int iters = 5
) {
    auto ybuf = y.request();
    auto ubuf = unit.request();
    auto tbuf = time.request();

    size_t n = ybuf.size;
    int* unit_ptr = static_cast<int*>(ubuf.ptr);
    int* time_ptr = static_cast<int*>(tbuf.ptr);

    // Copy input to output
    std::vector<double> yt(static_cast<double*>(ybuf.ptr),
                           static_cast<double*>(ybuf.ptr) + n);

    for (int it = 0; it < iters; it++) {
        // Unit FE demeaning
        std::vector<double> usum(n_unit, 0.0);
        std::vector<int> ucnt(n_unit, 0);

        for (size_t i = 0; i < n; i++) {
            usum[unit_ptr[i]] += yt[i];
            ucnt[unit_ptr[i]]++;
        }
        for (size_t i = 0; i < n; i++) {
            if (ucnt[unit_ptr[i]] > 0) {
                yt[i] -= usum[unit_ptr[i]] / ucnt[unit_ptr[i]];
            }
        }

        // Time FE demeaning
        std::vector<double> tsum(n_time, 0.0);
        std::vector<int> tcnt(n_time, 0);

        for (size_t i = 0; i < n; i++) {
            tsum[time_ptr[i]] += yt[i];
            tcnt[time_ptr[i]]++;
        }
        for (size_t i = 0; i < n; i++) {
            if (tcnt[time_ptr[i]] > 0) {
                yt[i] -= tsum[time_ptr[i]] / tcnt[time_ptr[i]];
            }
        }
    }

    // Return as numpy array
    auto result = py::array_t<double>(n);
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);
    std::copy(yt.begin(), yt.end(), result_ptr);

    return result;
}

/**
 * IV-style CATE estimator: τ̂ = Σ D̃Ỹ / Σ D̃²
 * Uses pre-residualized Y and D.
 */
double estimate_tau_from_residuals(
    py::array_t<double> y_tilde,
    py::array_t<double> d_tilde
) {
    auto ybuf = y_tilde.request();
    auto dbuf = d_tilde.request();

    double* y_ptr = static_cast<double*>(ybuf.ptr);
    double* d_ptr = static_cast<double*>(dbuf.ptr);
    size_t n = ybuf.size;

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        num += d_ptr[i] * y_ptr[i];
        den += d_ptr[i] * d_ptr[i];
    }

    if (den < 1e-10) return 0.0;
    return num / den;
}

/**
 * Full τ estimator with FE residualization.
 */
double estimate_tau(
    py::array_t<double> y,
    py::array_t<double> d,
    py::array_t<int> unit,
    py::array_t<int> time,
    int n_unit,
    int n_time,
    int iters = 5
) {
    auto y_tilde = residualize_2way(y, unit, time, n_unit, n_time, iters);
    auto d_tilde = residualize_2way(d, unit, time, n_unit, n_time, iters);
    return estimate_tau_from_residuals(y_tilde, d_tilde);
}


/**
 * Split score based on τ-heterogeneity.
 * Score = (nL * nR / n²) * (τL - τR)²
 * 
 * This maximizes heterogeneity in treatment effects across the split.
 */
double split_score(
    py::array_t<double> y_tilde,
    py::array_t<double> d_tilde,
    py::array_t<bool> left_mask
) {
    auto ybuf = y_tilde.request();
    auto dbuf = d_tilde.request();
    auto mbuf = left_mask.request();

    double* y_ptr = static_cast<double*>(ybuf.ptr);
    double* d_ptr = static_cast<double*>(dbuf.ptr);
    bool* mask_ptr = static_cast<bool*>(mbuf.ptr);
    size_t n = ybuf.size;

    // Compute τL and τR
    double numL = 0.0, denL = 0.0;
    double numR = 0.0, denR = 0.0;
    int nL = 0, nR = 0;

    for (size_t i = 0; i < n; i++) {
        if (mask_ptr[i]) {
            numL += d_ptr[i] * y_ptr[i];
            denL += d_ptr[i] * d_ptr[i];
            nL++;
        } else {
            numR += d_ptr[i] * y_ptr[i];
            denR += d_ptr[i] * d_ptr[i];
            nR++;
        }
    }

    if (nL == 0 || nR == 0) return 0.0;
    if (denL < 1e-10 || denR < 1e-10) return 0.0;

    double tauL = numL / denL;
    double tauR = numR / denR;
    double diff = tauL - tauR;

    return (static_cast<double>(nL) * nR / (n * n)) * diff * diff;
}

/**
 * Find best split for a single feature.
 * Returns (best_threshold, best_score).
 */
std::pair<double, double> find_best_split(
    py::array_t<double> x_col,
    py::array_t<double> y_tilde,
    py::array_t<double> d_tilde,
    int min_leaf
) {
    auto xbuf = x_col.request();
    auto ybuf = y_tilde.request();
    auto dbuf = d_tilde.request();

    double* x_ptr = static_cast<double*>(xbuf.ptr);
    double* y_ptr = static_cast<double*>(ybuf.ptr);
    double* d_ptr = static_cast<double*>(dbuf.ptr);
    size_t n = xbuf.size;

    // Get sorted unique thresholds
    std::vector<double> x_sorted(x_ptr, x_ptr + n);
    std::sort(x_sorted.begin(), x_sorted.end());

    double best_score = -1.0;
    double best_thresh = 0.0;

    // Try midpoints between sorted values
    for (size_t i = 0; i < n - 1; i++) {
        if (x_sorted[i] == x_sorted[i + 1]) continue;

        double thresh = (x_sorted[i] + x_sorted[i + 1]) / 2.0;

        // Count left/right
        int nL = 0, nR = 0;
        for (size_t j = 0; j < n; j++) {
            if (x_ptr[j] <= thresh) nL++;
            else nR++;
        }

        // Check min_leaf constraint
        if (nL < min_leaf || nR < min_leaf) continue;

        // Compute split score
        double numL = 0.0, denL = 0.0;
        double numR = 0.0, denR = 0.0;

        for (size_t j = 0; j < n; j++) {
            if (x_ptr[j] <= thresh) {
                numL += d_ptr[j] * y_ptr[j];
                denL += d_ptr[j] * d_ptr[j];
            } else {
                numR += d_ptr[j] * y_ptr[j];
                denR += d_ptr[j] * d_ptr[j];
            }
        }

        if (denL < 1e-10 || denR < 1e-10) continue;

        double tauL = numL / denL;
        double tauR = numR / denR;
        double diff = tauL - tauR;
        double score = (static_cast<double>(nL) * nR / (n * n)) * diff * diff;

        if (score > best_score) {
            best_score = score;
            best_thresh = thresh;
        }
    }

    return std::make_pair(best_thresh, best_score);
}

// ============== PYBIND11 MODULE ==============

PYBIND11_MODULE(cffe_core, m) {
    m.doc() = "CFFE Core C++ functions for Causal Forests with Fixed Effects";

    m.def("residualize_2way", &residualize_2way,
          "Two-way FE residualization via iterative demeaning",
          py::arg("y"), py::arg("unit"), py::arg("time"),
          py::arg("n_unit"), py::arg("n_time"), py::arg("iters") = 5);

    m.def("estimate_tau", &estimate_tau,
          "IV-style CATE estimator with FE residualization",
          py::arg("y"), py::arg("d"), py::arg("unit"), py::arg("time"),
          py::arg("n_unit"), py::arg("n_time"), py::arg("iters") = 5);

    m.def("estimate_tau_from_residuals", &estimate_tau_from_residuals,
          "IV-style CATE estimator from pre-residualized data",
          py::arg("y_tilde"), py::arg("d_tilde"));

    m.def("split_score", &split_score,
          "τ-heterogeneity split score",
          py::arg("y_tilde"), py::arg("d_tilde"), py::arg("left_mask"));

    m.def("find_best_split", &find_best_split,
          "Find best split threshold for a feature",
          py::arg("x_col"), py::arg("y_tilde"), py::arg("d_tilde"), py::arg("min_leaf"));
}
