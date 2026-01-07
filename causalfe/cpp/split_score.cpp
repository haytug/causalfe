double split_score(
    py::array_t<double> y,
    py::array_t<double> d,
    py::array_t<int> unit,
    py::array_t<int> time,
    py::array_t<int> left,
    py::array_t<int> right
) {
    int nL = left.request().size;
    int nR = right.request().size;

    double tauL = estimate_tau(
        y[left], d[left], unit[left], time[left],
        /*reindexed*/ nL, nL
    );
    double tauR = estimate_tau(
        y[right], d[right], unit[right], time[right],
        /*reindexed*/ nR, nR
    );

    double n = nL + nR;
    return (nL * nR / (n * n)) * (tauL - tauR) * (tauL - tauR);
}
