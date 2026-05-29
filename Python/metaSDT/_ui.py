"""Progress UI helpers for the Python frontend."""

from . import _progress_bar


class _UI:
    @staticmethod
    def progress_start(
        total,
        title="Progress",
        refresh_ms=100,
        mode="auto",
        line_interval_sec=2,
        line_interval_pct=5.0,
    ):
        _progress_bar.progress_start(
            total=total,
            title=title,
            refresh_ms=refresh_ms,
            mode=mode,
            line_interval_sec=line_interval_sec,
            line_interval_pct=line_interval_pct,
        )

    @staticmethod
    def progress_bar(current):
        _progress_bar.progress_bar(current=current)

    @staticmethod
    def progress_advance(step=1):
        _progress_bar.progress_advance(step=step)

    @staticmethod
    def progress_finish():
        _progress_bar.progress_finish()

    @staticmethod
    def progress_snapshot():
        return _progress_bar.progress_snapshot()


ui = _UI()
