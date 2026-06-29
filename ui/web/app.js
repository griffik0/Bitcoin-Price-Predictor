let pnlChart = null;

function fmtMoney(n) {
  if (n == null || Number.isNaN(n)) return "--";
  const sign = n >= 0 ? "+" : "";
  return `${sign}$${n.toFixed(2)}`;
}

function fmtUptime(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function fmtTime(iso) {
  if (!iso) return "--";
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return iso;
  }
}

function pnlClass(v) {
  if (v > 0) return "positive";
  if (v < 0) return "negative";
  return "";
}

function ensureChart() {
  if (pnlChart) return pnlChart;
  const ctx = document.getElementById("pnl-chart");
  pnlChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Cumulative PnL",
        data: [],
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59, 130, 246, 0.12)",
        borderWidth: 2,
        fill: true,
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: "#3b82f6",
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e293b",
          titleColor: "#e2e8f0",
          bodyColor: "#cbd5e1",
          callbacks: {
            label(ctx) {
              return `PnL: ${fmtMoney(ctx.parsed.y)}`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: "#1e2a3844" },
          ticks: { color: "#8b9cb0", maxTicksLimit: 8 },
        },
        y: {
          grid: { color: "#1e2a3844" },
          ticks: {
            color: "#8b9cb0",
            callback: (v) => `$${v}`,
          },
        },
      },
    },
  });
  return pnlChart;
}

function updateChart(history) {
  const chart = ensureChart();
  const points = history && history.length ? history : [{ at: "start", cumulative: 0 }];
  const labels = points.map((p) => (p.at === "start" ? "Start" : fmtTime(p.at)));
  const values = points.map((p) => p.cumulative);
  const latest = values[values.length - 1] || 0;
  const color = latest >= 0 ? "#22c55e" : "#ef4444";
  const fill = latest >= 0 ? "rgba(34, 197, 94, 0.15)" : "rgba(239, 68, 68, 0.15)";

  chart.data.labels = labels;
  chart.data.datasets[0].data = values;
  chart.data.datasets[0].borderColor = color;
  chart.data.datasets[0].pointBackgroundColor = color;
  chart.data.datasets[0].backgroundColor = fill;
  chart.update("none");
}

function renderPicks(opportunities) {
  const body = document.getElementById("picks-body");
  document.getElementById("picks-count").textContent =
    `${opportunities.length} pick${opportunities.length === 1 ? "" : "s"}`;

  if (!opportunities.length) {
    body.innerHTML =
      '<tr><td colspan="7" class="empty">No picks this cycle</td></tr>';
    return;
  }

  body.innerHTML = opportunities
    .map((o) => {
      const dir = o.direction || "--";
      const tag = dir === "YES" ? "yes" : "no";
      return `<tr>
        <td><span class="tag ${tag}">${dir}</span></td>
        <td>${Math.round(o.contract_price)}c</td>
        <td>${(o.model_prob * 100).toFixed(0)}%</td>
        <td>${o.edge_pct >= 0 ? "+" : ""}${o.edge_pct.toFixed(1)}%</td>
        <td>${o.confidence.toFixed(2)}</td>
        <td>$${o.recommended_stake.toFixed(2)}</td>
        <td>${Math.round(o.minutes_to_close)}m</td>
      </tr>`;
    })
    .join("");
}

function renderTrades(trades, openCount) {
  const body = document.getElementById("trades-body");
  document.getElementById("open-count").textContent =
    `${openCount} open`;

  if (!trades.length) {
    body.innerHTML =
      '<tr><td colspan="5" class="empty">No trades yet</td></tr>';
    return;
  }

  body.innerHTML = trades
    .map((t) => {
      const status = (t.status || "open").toLowerCase();
      const pnl = t.pnl_usd;
      let pnlCell = "--";
      if (pnl != null && status !== "open") {
        const cls = pnl >= 0 ? "pnl-pos" : "pnl-neg";
        pnlCell = `<span class="${cls}">${fmtMoney(pnl)}</span>`;
      }
      const shortTicker = (t.ticker || "").replace("KXBTC15M-", "");
      return `<tr>
        <td title="${t.ticker || ""}">${shortTicker}</td>
        <td>${(t.direction || t.side || "--").toUpperCase()}</td>
        <td>${t.entry_price_cents ?? "--"}c</td>
        <td class="status-${status}">${status}</td>
        <td>${pnlCell}</td>
      </tr>`;
    })
    .join("");
}

function renderState(data) {
  document.getElementById("stat-uptime").textContent =
    fmtUptime(data.uptime_seconds);
  document.getElementById("stat-spot").textContent =
    data.spot_price ? `$${data.spot_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "--";

  const paper = data.paper;
  const bankroll = paper ? paper.current_bankroll : data.bankroll;
  document.getElementById("stat-bankroll").textContent =
    bankroll != null ? `$${bankroll.toFixed(2)}` : "--";

  const pnl = paper ? paper.realized_pnl : 0;
  const pnlEl = document.getElementById("stat-pnl");
  pnlEl.textContent = fmtMoney(pnl);
  pnlEl.className = `value ${pnlClass(pnl)}`;

  if (paper && paper.resolved_bets > 0) {
    document.getElementById("stat-winrate").textContent =
      `${(paper.win_rate * 100).toFixed(0)}%`;
    document.getElementById("stat-record").textContent =
      `${paper.wins}W-${paper.losses}L`;
  } else {
    document.getElementById("stat-winrate").textContent = "--";
    document.getElementById("stat-record").textContent = "0W-0L";
  }

  const mode = (data.trade_mode || "off").toLowerCase();
  const badge = document.getElementById("trade-mode");
  badge.textContent = mode;
  badge.className = `badge ${mode.replace("_", "-")}`;

  document.getElementById("last-updated").textContent =
    data.updated_at ? `Updated ${fmtTime(data.updated_at)}` : "Waiting...";

  updateChart(paper ? paper.pnl_history : [{ at: "start", cumulative: 0 }]);
  renderPicks(data.opportunities || []);
  renderTrades(paper ? paper.trades || [] : [], paper ? paper.open_positions || 0 : 0);
}

async function poll() {
  try {
    const res = await fetch("/api/state", { cache: "no-store" });
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    renderState(data);
  } catch (err) {
    document.getElementById("last-updated").textContent =
      `Offline (${err.message})`;
  }
}

poll();
setInterval(poll, 5000);
