"""Headless launch smoke test: builds the UI, starts the server, GETs the page.

    uv run --group ui python webui/serve_test.py
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webui import app as app_module  # noqa: E402


def main() -> None:
    demo = app_module.build_ui()
    demo.queue(max_size=2)
    import contextlib

    with contextlib.suppress(Exception):
        demo.launch(
            server_name="127.0.0.1",
            server_port=7865,
            prevent_thread_lock=True,
            inbrowser=False,
            share=False,
            show_error=True,
        )
    url = str(demo.local_url)
    html = ""
    for attempt in range(40):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                html = response.read().decode("utf-8", "replace")
            break
        except Exception as exc:
            if attempt == 39:
                print(f"server unreachable at {url}: {exc}")
                raise SystemExit(1)
            time.sleep(1)
    assert "MSFlow" in html or "gradio" in html, "unexpected landing page"
    with urllib.request.urlopen(url, timeout=5) as config_page:
        assert config_page.status == 200
    demo.close()
    print(f"serve_test OK: {url} answered ({len(html)} bytes)")


if __name__ == "__main__":
    main()
