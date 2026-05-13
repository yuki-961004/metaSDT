#include <pybind11/pybind11.h>
#include "../../Cpp/include/progress_bar.hpp"

// This module exposes progress-state APIs to Python. It intentionally keeps
// logic thin so rendering strategy can stay in Python-side caller code.

PYBIND11_MODULE(_ui, m) {
    m.doc() = "metaSDT UI helpers";

    m.def("progress_start",
          [](std::size_t total, const std::string& title, int refresh_ms,
             const std::string& mode, int line_interval_sec, double line_interval_pct) {
              ui::ProgressOptions opts;
              opts.mode = mode;
              opts.refresh_ms = refresh_ms;
              opts.line_interval_sec = static_cast<double>(line_interval_sec);
              opts.line_interval_pct = line_interval_pct;
              ui::progress_start(total, title, refresh_ms, opts);
          },
          pybind11::arg("total"),
          pybind11::arg("title") = "Progress",
          pybind11::arg("refresh_ms") = 100,
          pybind11::arg("mode") = "auto",
          pybind11::arg("line_interval_sec") = 2,
          pybind11::arg("line_interval_pct") = 5.0);

    m.def("progress_bar", &ui::progress_set, pybind11::arg("current"));
    m.def("progress_advance", &ui::progress_advance, pybind11::arg("step") = 1);
    m.def("progress_finish", &ui::progress_finish);

    m.def("progress_snapshot", []() {
        ui::ProgressSnapshot s = ui::progress_last_snapshot();
        pybind11::dict out;
        out["title"] = s.title;
        out["requested_mode"] = s.requested_mode;
        out["resolved_mode"] = s.resolved_mode;
        out["elapsed_sec"] = s.elapsed_sec;
        out["percent"] = s.percent;
        out["eta_sec"] = s.eta_sec;
        out["speed"] = s.speed;
        out["started"] = s.started;
        out["running"] = s.running;
        out["finished"] = s.finished;
        out["total_iterations"] = s.total;
        out["completed_iterations"] = s.completed;
        return out;
    });
}
