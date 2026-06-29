"""Terminal PnL chart and uptime formatting for the dashboard."""

from __future__ import annotations

from rich.text import Text


def format_uptime(seconds: float) -> str:
    """Human-readable uptime (e.g. 1h 4m, 12m 30s)."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _resample(values: list[float], width: int) -> list[float]:
    if width <= 0:
        return []
    if not values:
        return [0.0] * width
    if len(values) == 1:
        return [values[0]] * width
    if len(values) <= width:
        out = list(values)
        while len(out) < width:
            out.append(out[-1])
        return out
    step = (len(values) - 1) / (width - 1)
    return [values[int(round(i * step))] for i in range(width)]


def build_pnl_chart(
    cumulative_pnl: list[float],
    *,
    width: int = 64,
    height: int = 7,
    title: str = "Cumulative PnL",
) -> Text:
    """
    Render a compact cumulative PnL line chart for the Rich dashboard.

    Values are dollar PnL (0 = break-even). Colors green above zero, red below.
    """
    text = Text()
    chart_width = max(12, width - 10)
    chart_height = max(4, height)

    if not cumulative_pnl:
        cumulative_pnl = [0.0]
    if cumulative_pnl[0] != 0.0:
        cumulative_pnl = [0.0, *cumulative_pnl]

    sampled = _resample(cumulative_pnl, chart_width)
    y_min = min(0.0, min(sampled))
    y_max = max(0.0, max(sampled))
    if y_max - y_min < 0.25:
        y_min -= 0.5
        y_max += 0.5

    def row_for(value: float) -> int:
        if y_max == y_min:
            return chart_height // 2
        return int(round((y_max - value) / (y_max - y_min) * (chart_height - 1)))

    zero_row = row_for(0.0)
    grid: list[list[tuple[str, str] | None]] = [
        [None] * chart_width for _ in range(chart_height)
    ]

    prev_row: int | None = None
    for x, value in enumerate(sampled):
        row = row_for(value)
        row = max(0, min(chart_height - 1, row))
        color = "green" if value >= 0 else "red"
        if prev_row is not None:
            step = 1 if row >= prev_row else -1
            for r in range(prev_row, row + step, step):
                if grid[r][x] is None:
                    grid[r][x] = ("line", color)
        grid[row][x] = ("point", color)
        prev_row = row

    point_char = "#"
    line_char = "|"
    zero_char = "-"

    latest = cumulative_pnl[-1]
    pnl_color = "green" if latest >= 0 else "red"
    text.append(f"{title}: ", style="bold")
    text.append(f"${latest:+.2f}\n", style=f"bold {pnl_color}")

    y_labels = [y_max, (y_max + y_min) / 2, y_min]
    label_rows = [0, chart_height // 2, chart_height - 1]

    for row_idx in range(chart_height):
        label = ""
        if row_idx in label_rows:
            label = f"${y_labels[label_rows.index(row_idx)]:+.2f}"[:8].rjust(8)
        else:
            label = " " * 8
        text.append(f"{label} ", style="dim")

        for col_idx in range(chart_width):
            cell = grid[row_idx][col_idx]
            on_zero = row_idx == zero_row
            if cell is not None:
                kind, style = cell
                char = point_char if kind == "point" else line_char
                text.append(char, style=style)
            elif on_zero:
                text.append(zero_char, style="dim")
            else:
                text.append(" ")
        text.append("\n")

    start_label = "start"
    end_label = "now"
    pad = chart_width - len(start_label) - len(end_label)
    text.append(" " * 8, style="dim")
    text.append(start_label, style="dim")
    text.append(" " * max(1, pad), style="dim")
    text.append(end_label, style="dim")

    return text


def empty_pnl_chart(*, message: str = "No closed trades yet - flat at $0.00") -> Text:
    text = Text()
    text.append("PnL ", style="bold")
    text.append("$0.00", style="bold dim")
    text.append(f"  |  {message}", style="dim")
    return text
