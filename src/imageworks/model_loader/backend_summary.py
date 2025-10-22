from __future__ import annotations

import json
from typing import List, Dict, Any

from .registry import load_registry
from .service import default_backend_port


def gather_backend_rows() -> List[Dict[str, Any]]:
    reg = load_registry()
    rows: List[Dict[str, Any]] = []
    for entry in sorted(reg.values(), key=lambda e: e.name.lower()):
        cfg = entry.backend_config
        base_url = getattr(cfg, "base_url", None)
        host = getattr(cfg, "host", None)
        port_value = getattr(cfg, "port", None)
        if base_url:
            host_display = base_url
            port_display = "-"
        else:
            host_display = host or "localhost"
            resolved_port = (
                port_value
                if isinstance(port_value, int) and port_value > 0
                else default_backend_port(entry.backend)
            )
            port_display = resolved_port
        rows.append(
            {
                "name": entry.name,
                "backend": entry.backend,
                "host_or_base": host_display,
                "port": port_display,
                "served_model_id": entry.served_model_id or "-",
                "model_path": getattr(cfg, "model_path", "") or "-",
                "extra_args": getattr(cfg, "extra_args", []) or [],
            }
        )
    return rows


def render_backend_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No registry entries loaded."
    name_width = max(20, max(len(r["name"]) for r in rows))
    host_width = max(20, max(len(str(r["host_or_base"])) for r in rows))
    header = (
        f"{'name':{name_width}}  {'backend':7}  {'host/base':{host_width}}  {'port':5}  "
        f"{'served_id':20}  model_path"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        extra = " ".join(r["extra_args"]) if r["extra_args"] else "-"
        line = (
            f"{r['name']:{name_width}}  {r['backend']:7}  "
            f"{str(r['host_or_base']):{host_width}}  {str(r['port']):5}  "
            f"{r['served_model_id'][:20]:20}  {r['model_path']}  ({extra})"
        )
        lines.append(line)
    return "\n".join(lines)


def emit_backend_summary(json_output: bool) -> None:
    rows = gather_backend_rows()
    if json_output:
        print(json.dumps(rows, indent=2))
        return
    print(render_backend_rows(rows))


__all__ = ["gather_backend_rows", "render_backend_rows", "emit_backend_summary"]
