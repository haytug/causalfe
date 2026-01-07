double estimate_tau(
    py::array_t<double> y,
    py::array_t<double> d,
    py::array_t<int> unit,
    py::array_t<int> time,
    int n_unit,
    int n_time
) {
    auto yt = residualize_2way(y, unit, time, n_unit, n_time, 5);
    auto dt = residualize_2way(d, unit, time, n_unit, n_time, 5);

    auto yb = yt.request();
    auto db = dt.request();

    double num = 0.0, den = 0.0;
    for (int i = 0; i < yb.size; i++) {
        num += ((double*)db.ptr)[i] * ((double*)yb.ptr)[i];
        den += ((double*)db.ptr)[i] * ((double*)db.ptr)[i];
    }
    return num / den;
}
