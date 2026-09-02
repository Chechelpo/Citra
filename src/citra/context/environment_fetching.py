from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import platform
import os

@dataclass(frozen=True)
class EnvironmentInfo:
    """
    Dynamic environment information exposed to the model.
    """

    os: str
    os_release: str
    architecture: str
    datetime: str
    timezone: str

    @staticmethod
    def collect_environment() -> EnvironmentInfo:
        now = datetime.now().astimezone()

        return EnvironmentInfo(
            os=platform.system(),
            os_release=platform.release() or "unknown",
            architecture=platform.machine() or "unknown",
            datetime=now.isoformat(
                timespec="seconds"
            ),
            timezone=_timezone_name(now),
        )

    def as_prompt_section(self) -> str:
        return "\n".join(
            (
                "## Environment",
                "",
                f"- OS: {self.os} - {self.os_release}",
                f"- Architecture: {self.architecture}",
                f"- Date/time: {self.datetime}",
                f"- Timezone: {self.timezone}",
            )
        )

def _timezone_name(
    value: datetime,
) -> str:
    name = value.tzname()

    if name:
        return name

    offset = value.strftime("%z")

    if offset:
        return offset

    return "unknown"
