import json
import os
import time
from typing import Any

SESSION_ID = "b6799e"
LOG_PATHS = [
    os.getenv(
        "DEBUG_LOG_PATH",
        "/Users/andhrelja/Projects/refinery-ui/.cursor/debug-b6799e.log",
    ),
    "/program/.cursor/debug-b6799e.log",
]


def debug_log(
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    hypothesis_id: str = "",
) -> None:
    payload = {
        "sessionId": SESSION_ID,
        "timestamp": int(time.time() * 1000),
        "location": location,
        "message": message,
        "data": data or {},
        "hypothesisId": hypothesis_id,
    }
    line = json.dumps(payload, default=str)
    print(f"DEBUG_B6799E {line}", flush=True)
    for path in LOG_PATHS:
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8") as log_file:
                log_file.write(line + "\n")
            break
        except OSError:
            continue
