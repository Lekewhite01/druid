"""
druid.utils.display
~~~~~~~~~~~~~~~~~~~~

Rich console output and formatting helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def _get_console():
    """Lazy-import rich Console."""
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


def print_header(text: str) -> None:
    """Print a styled section header."""
    console = _get_console()
    if console:
        console.print(f"\n[bold cyan]{'─' * 60}[/]")
        console.print(f"[bold cyan]  {text}[/]")
        console.print(f"[bold cyan]{'─' * 60}[/]\n")
    else:
        print(f"\n{'─' * 60}")
        print(f"  {text}")
        print(f"{'─' * 60}\n")


def print_leaderboard(leaderboard: List[Dict[str, Any]]) -> None:
    """Pretty-print a model leaderboard."""
    console = _get_console()

    if console:
        from rich.table import Table

        table = Table(title="Model Leaderboard", show_lines=True)
        if not leaderboard:
            console.print("[yellow]No results to display[/]")
            return

        # Add columns from the first result
        for key in leaderboard[0]:
            table.add_column(key, justify="right" if key != "model" else "left")

        for i, row in enumerate(leaderboard):
            style = "bold green" if i == 0 else None
            values = [str(row.get(k, "")) for k in leaderboard[0]]
            table.add_row(*values, style=style)

        console.print(table)
    else:
        df = pd.DataFrame(leaderboard)
        print(df.to_string(index=False))


def print_profile_summary(profile: Dict[str, Any]) -> None:
    """Print a compact dataset profile summary."""
    console = _get_console()
    shape = profile.get("shape", {})
    missing = profile.get("missing", {})
    classification = profile.get("classification", {})

    lines = [
        f"Dataset: {profile.get('name', 'unknown')}",
        f"Shape: {shape.get('rows', '?'):,} rows × {shape.get('columns', '?')} columns",
        f"Target: {profile.get('target', 'not set')}",
        f"Missing: {len(missing)} columns with nulls",
        f"Duplicates: {profile.get('duplicates', 0):,}",
    ]

    if classification:
        for ctype, cols in classification.items():
            if cols:
                lines.append(f"  {ctype}: {len(cols)} cols")

    if console:
        from rich.panel import Panel
        console.print(Panel("\n".join(lines), title="Dataset Profile", border_style="cyan"))
    else:
        print("\n".join(lines))
