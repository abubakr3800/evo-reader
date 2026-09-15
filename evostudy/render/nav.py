"""
Tiny shared nav bar so the two standalone HTML exports (dashboard.html and
calculations.html) can link to each other. Each page is still a fully
self-contained file (no shared assets, no server needed) — this just keeps
the markup/CSS identical between them.
"""

from __future__ import annotations

PAGE_NAV_CSS = """
nav.pagenav{display:flex;gap:4px;padding:0 28px;background:#fff;
  border-bottom:1px solid var(--border, #e3e5e9)}
nav.pagenav a{color:#6b7280;text-decoration:none;font-size:13px;font-weight:600;
  padding:11px 14px;border-bottom:2px solid transparent}
nav.pagenav a:hover{color:#1b1f24}
nav.pagenav a.active{color:#c8102e;border-bottom-color:#c8102e}
"""

PAGES = [
    ("dashboard.html", "Dashboard"),
    ("calculations.html", "Calculations"),
]


def page_nav_html(active: str) -> str:
    links = "".join(
        f'<a href="{href}" class="{"active" if href == active else ""}">{label}</a>'
        for href, label in PAGES
    )
    return f'<nav class="pagenav">{links}</nav>'
