"""What long-running engine work tells the UI about itself.

One call is one line in the activity log, with two optional extras:

    progress   (done, total) for work that advances in place. The UI draws a
               bar and the activity log keeps a single entry for the whole
               sequence, instead of one line per tuning trial.
    device     the compute this work actually runs on ("CPU" / "GPU").
"""

from __future__ import annotations

from typing import Protocol


class ProgressCallback(Protocol):
    def __call__(
        self,
        task: str,
        *,
        progress: tuple[int, int] | None = None,
        device: str | None = None,
    ) -> None: ...
