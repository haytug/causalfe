PYBIND11_MODULE(cffe_core, m) {
    m.def("residualize_2way", &residualize_2way);
    m.def("estimate_tau", &estimate_tau);
    m.def("split_score", &split_score);
}
