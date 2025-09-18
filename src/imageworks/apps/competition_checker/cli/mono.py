from pathlib import Path
import json
import typer
from rich.progress import track
from imageworks.libs.vision.mono import check_monochrome

app = typer.Typer(help="Competition Checker - Monochrome validation")


def _iter_files(root: Path, exts_csv: str):
    exts = [e.strip().lstrip(".").lower() for e in exts_csv.split(",") if e.strip()]
    for p in root.rglob("*"):
        if p.is_file():
            suf = p.suffix.lstrip(".").lower()
            if suf in exts:
                yield p


@app.command()
def check(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    exts: str = typer.Option(
        "jpg,jpeg", help="Comma-separated extensions (case-insensitive)"
    ),
    neutral_tol: int = typer.Option(2, help="Max channel diff (8-bit) for 'neutral'"),
    toned_pass: float = typer.Option(6.0, help="Hue σ (deg) threshold for PASS toned"),
    toned_query: float = typer.Option(
        10.0, help="Hue σ (deg) threshold for PASS-WITH-QUERY"
    ),
    jsonl_out: Path | None = typer.Option(
        None, help="Write results as JSONL to this path"
    ),
    summary_only: bool = typer.Option(False, help="Only print summary counts"),
):
    paths = list(_iter_files(folder, exts))
    if not paths:
        typer.echo(f"No files matched in {folder} for extensions: {exts}")
        raise typer.Exit(1)

    out_f = open(jsonl_out, "w") if jsonl_out else None
    counts = {"pass": 0, "pass_with_query": 0, "fail": 0}

    for p in track(paths, description="Checking"):
        res = check_monochrome(str(p), neutral_tol, toned_pass, toned_query)
        counts[res.verdict] += 1
        if not summary_only:
            typer.echo(
                f"[{res.verdict}] {p.name:40s}  mode={res.mode:9s}  "
                f"maxΔ={res.channel_max_diff:.1f}  hueσ={res.hue_std_deg:.2f}"
            )
        if out_f:
            out_f.write(
                json.dumps(
                    {
                        "path": str(p),
                        "verdict": res.verdict,
                        "mode": res.mode,
                        "channel_max_diff": res.channel_max_diff,
                        "hue_std_deg": res.hue_std_deg,
                    }
                )
                + "\n"
            )

    if out_f:
        out_f.close()
    typer.echo(
        f"\nSummary: PASS={counts['pass']}  QUERY={counts['pass_with_query']}  FAIL={counts['fail']}"
    )
