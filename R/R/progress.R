progress_start <- function(
    total,
    title = "Progress",
    refresh_ms = 100L,
    mode = "auto",
    line_interval_sec = 2.0,
    line_interval_pct = 5.0
) {
    .ui_progress_start(
        total,
        title,
        refresh_ms,
        mode,
        line_interval_sec,
        line_interval_pct
    )
}

progress_set <- function(current) {
    .ui_progress_set(current)
}

progress_advance <- function(step = 1.0) {
    .ui_progress_advance(step)
}

progress_finish <- function() {
    .ui_progress_finish()
}

progress_snapshot <- function() {
    .ui_progress_snapshot()
}
