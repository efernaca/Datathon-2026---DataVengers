(() => {
  const state = {
    paused: false,
    windowMin: 3,
    reqFilter: "",
    logModel: "",
    busyOp: false,
  };

  const el = (id) => document.getElementById(id);

  function esc(val) {
    return String(val ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function postJson(url, payload) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      alert(data.message || "Error executant l'accio");
    }
    await fetchAndRender(true);
  }

  function renderOperation(op) {
    const banner = el("op-banner");
    if (!op || op.state === "idle") {
      banner.classList.add("hidden");
      state.busyOp = false;
      return;
    }
    const inProgress = ["self_check", "training", "starting", "switching"].includes(op.state);
    state.busyOp = inProgress;

    const ageSec = op.updated_at ? Date.now() / 1000 - op.updated_at : 0;
    if (!inProgress && ageSec > 25) {
      banner.classList.add("hidden");
      return;
    }

    banner.classList.remove("hidden", "failed", "completed");
    if (op.state === "failed") banner.classList.add("failed");
    if (op.state === "completed") banner.classList.add("completed");

    el("op-title").textContent = `${op.action || "Operacio"} -> ${op.target_folder || ""}`;
    el("op-pct").textContent = `${op.progress ?? 0}%`;
    el("op-bar").style.width = `${Math.max(0, Math.min(100, op.progress ?? 0))}%`;
    el("op-msg").textContent = op.message || "";
  }

  function renderActiveModel(data) {
    const serves = (data.processes || []).filter((p) => p.role === "serve");
    const trainers = (data.processes || []).filter((p) => p.role === "trainer");
    const badge = el("active-badge");
    const body = el("active-model-body");

    if (serves.length === 0) {
      badge.className = "pill pill-bad";
      badge.textContent = "CAP MODEL ACTIU";
      body.innerHTML = `<p class="muted">No hi ha cap proces <code>serve.py</code> en marxa ara mateix. Selecciona un model a sota per iniciar-lo.</p>`;
      return;
    }

    badge.className = serves.length === 1 ? "pill pill-live" : "pill pill-warn";
    badge.textContent = serves.length === 1 ? "ACTIU EN DIRECTE" : `${serves.length} PROCESSOS ACTIUS`;

    const primary = serves[0];
    const hb = primary.heartbeat;
    const hbLine = hb
      ? `Snapshot v${esc(hb.snapshot_version)} · ml_alpha=${esc(hb.ml_alpha)} · ${esc(hb.regime)} · ${esc(hb.submitted_total)} enviats (fa ${esc(hb.age_s)}s)`
      : "Sense fitxer heartbeat JSON (model historic)";

    const multiNote =
      serves.length > 1
        ? `<p class="pill pill-warn">Atencio: hi ha ${serves.length} processos servint (${serves.map((s) => s.folder + "#" + s.pid).join(", ")})</p>`
        : "";
    const trainNote =
      trainers.length > 0
        ? `<p class="pill pill-enc">Entrenant en segon pla: ${trainers.map((t) => `${t.folder} (PID ${t.pid})`).join(", ")}</p>`
        : "";

    body.innerHTML = `
      <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
        <span style="font-size:20px; font-weight:700;">${esc(primary.folder)}</span>
        <span class="pill pill-model">${esc(primary.model_version)}</span>
        <span class="pill pill-muted">PID ${esc(primary.pid)}</span>
      </div>
      <div class="active-meta-grid">
        <div class="meta-item"><span>Temps actiu</span><strong>${esc(primary.uptime_str)}</strong></div>
        <div class="meta-item"><span>Memoria RAM</span><strong>${esc(primary.rss_mb)} MB</strong></div>
        <div class="meta-item"><span>Estat intern</span><strong>${hb && hb.ready ? "READY" : "SERVINT"}</strong></div>
      </div>
      <p class="muted" style="margin:10px 0 0; font-size:12px;">${hbLine}</p>
      ${multiNote}${trainNote}
    `;
  }

  function renderKPIs(telemetry) {
    const totals = telemetry.totals || {};
    el("kpi-window-label").textContent = `${telemetry.window_minutes || state.windowMin} min`;
    el("kpi-reqs").textContent = totals.reqs ?? 0;
    el("kpi-rpm").textContent = `${totals.req_per_min ?? 0} req/min`;
    el("kpi-ontime").textContent = `${totals.on_time_pct ?? 0}%`;
    el("kpi-p50").textContent = `${totals.p50_ms ?? 0} ms`;
    el("kpi-rev").textContent = `${Number(totals.revenue_eur ?? 0).toFixed(2)} €`;

    const errPill = el("rawtree-err");
    if (telemetry.error) {
      errPill.textContent = telemetry.error;
      errPill.classList.remove("hidden");
    } else {
      errPill.classList.add("hidden");
    }
  }

  function renderModels(models) {
    const grid = el("models-grid");
    grid.innerHTML = models
      .map((m) => {
        const badges = [];
        if (m.is_active) badges.push(`<span class="pill pill-live">EN VIU (PID ${m.running_serves.map((p) => p.pid).join(",")})</span>`);
        else if (m.is_training) badges.push(`<span class="pill pill-warn">ENTRENANT</span>`);
        else badges.push(`<span class="pill pill-idle">Aturat</span>`);

        const kindPill =
          m.kind === "encapsulated"
            ? `<span class="pill pill-enc">ENCAPSULAT</span>`
            : `<span class="pill pill-muted">HISTORIC</span>`;

        const disabled = state.busyOp ? "disabled" : "";
        return `
          <div class="model-card ${m.is_active ? "active" : ""}">
            <div>
              <div class="model-title-row">
                <span class="model-folder">${esc(m.folder)}</span>
                ${badges.join(" ")}
              </div>
              <div class="model-meta" style="margin-top:4px;">
                ${kindPill}
                <span class="pill pill-model">${esc(m.model_version)}</span>
                ${m.has_pretrain ? `<span class="pill pill-muted">Pre-entrenament</span>` : ""}
              </div>
              <p class="model-desc" style="margin-top:8px;">${esc(m.description || m.entrypoint)}</p>
            </div>
            <div class="model-actions">
              <button class="btn btn-primary" ${disabled} data-act="train" data-model="${esc(m.folder)}">
                🎓 Entrenar i Activar
              </button>
              <button class="btn btn-secondary" ${disabled} data-act="switch" data-model="${esc(m.folder)}">
                ⚡ Canvi Ràpid
              </button>
              ${
                m.is_active
                  ? `<button class="btn btn-danger-outline" data-act="stop" data-model="${esc(m.folder)}">⏹ Aturar</button>`
                  : ""
              }
            </div>
          </div>
        `;
      })
      .join("");

    grid.querySelectorAll("button[data-act]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const act = btn.getAttribute("data-act");
        const model = btn.getAttribute("data-model");
        if (act === "train") {
          await postJson("/api/train_activate", { model });
        } else if (act === "switch") {
          await postJson("/api/switch", { model });
        } else if (act === "stop") {
          await postJson("/api/stop", { model });
        }
      });
    });

    const logSel = el("log-model-select");
    const prevLog = state.logModel || logSel.value;
    if (logSel.options.length !== models.length) {
      logSel.innerHTML = models
        .map((m) => `<option value="${esc(m.folder)}">${esc(m.folder)} (${esc(m.model_version)})</option>`)
        .join("");
      if (prevLog && models.some((m) => m.folder === prevLog)) logSel.value = prevLog;
    }
  }

  function renderRequests(reqs) {
    const filterSel = el("req-model-filter");
    const versions = Array.from(new Set(reqs.map((r) => r.model_version))).sort();
    const curVal = filterSel.value;
    const desiredOpts = ["", ...versions];
    const currentOpts = Array.from(filterSel.options).map((o) => o.value);
    if (JSON.stringify(desiredOpts) !== JSON.stringify(currentOpts)) {
      filterSel.innerHTML = `<option value="">Tots els models</option>` + versions.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
      filterSel.value = desiredOpts.includes(curVal) ? curVal : "";
    }

    const filtered = state.reqFilter ? reqs.filter((r) => r.model_version === state.reqFilter) : reqs;
    const tbody = el("requests-tbody");
    if (!filtered.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="muted">Encara no hi ha sol·licituds recents per mostrar.</td></tr>`;
      return;
    }

    tbody.innerHTML = filtered
      .map((r) => {
        const latClass = r.latency_ms <= 1600 ? "lat-ok" : r.latency_ms <= 2800 ? "lat-mid" : "lat-high";
        const srcPill =
          r.served_by === "team" && r.model_version !== "fallback"
            ? `<span class="pill pill-live">team (on-time)</span>`
            : `<span class="pill pill-bad">${esc(r.served_by)}</span>`;
        let outcome = `<span class="muted">—</span>`;
        if (r.purchased) {
          outcome = `<span class="pill pill-live">COMPRA +${Number(r.revenue_eur).toFixed(2)} €</span>`;
        } else if (r.clicked) {
          outcome = `<span class="pill pill-warn">CLIC (${esc(r.clicked)})</span>`;
        }
        const itemsTxt = (r.items || []).slice(0, 5).join(", ") + ((r.items || []).length > 5 ? ", ..." : "");
        return `
          <tr>
            <td class="mono">${esc(r.time)}</td>
            <td class="mono">${esc(r.request_id)}</td>
            <td><span class="pill pill-model">${esc(r.model_version)}</span></td>
            <td>${srcPill}</td>
            <td class="${latClass}">${esc(r.latency_ms)} ms</td>
            <td>${outcome}</td>
            <td class="mono items-cell" title="${esc((r.items || []).join(", "))}">${esc(itemsTxt || "-")}</td>
          </tr>
        `;
      })
      .join("");
  }

  function renderTimelineAndSummary(telemetry) {
    const tlBody = el("timeline-tbody");
    const timeline = telemetry.timeline || [];
    tlBody.innerHTML = timeline.length
      ? timeline
          .map(
            (row) => `
        <tr>
          <td class="mono">${esc(row.minute)}</td>
          <td><span class="pill pill-model">${esc(row.model_version)}</span></td>
          <td>${esc(row.reqs)}</td>
          <td>${esc(row.on_time_pct)}%</td>
          <td>${esc(row.avg_ms)} ms</td>
          <td>${Number(row.revenue_eur).toFixed(2)} €</td>
        </tr>`
          )
          .join("")
      : `<tr><td colspan="6" class="muted">Sense dades en els ultims 15 minuts.</td></tr>`;

    const sumBody = el("summary-tbody");
    const summary = telemetry.models_summary || [];
    sumBody.innerHTML = summary.length
      ? summary
          .map(
            (s) => `
        <tr>
          <td><span class="pill pill-model">${esc(s.model_version)}</span></td>
          <td>${esc(s.reqs)} (${esc(s.traffic_pct)}%)</td>
          <td>${esc(s.on_time_pct)}%</td>
          <td class="mono">${esc(s.p50_ms)} / ${esc(s.p95_ms)} ms</td>
          <td class="mono">${esc(s.ctr_pct)}% / ${esc(s.cvr_pct)}%</td>
          <td><strong>${Number(s.eur_per_req).toFixed(4)} €</strong></td>
        </tr>`
          )
          .join("")
      : `<tr><td colspan="6" class="muted">Sense feedback a aquesta finestra.</td></tr>`;
  }

  function renderTransitionsAndLogs(data) {
    const tList = el("transitions-list");
    const transitions = data.transitions || [];
    tList.innerHTML = transitions.length
      ? transitions
          .map(
            (t) => `
        <div class="trans-item">
          <div><strong>${esc(t.action)}</strong> -> <span class="pill pill-model">${esc(t.to_folder)} (${esc(t.model_version)})</span> · PID ${esc(t.pid)} (${esc(t.elapsed_s)}s)</div>
          <div class="muted" style="margin-top:3px;">${esc(t.ts)} · ${esc(t.detail)}</div>
        </div>`
          )
          .join("")
      : `<p class="muted">Encara no s'han registrat canvis des del gestor.</p>`;

    const viewer = el("logs-viewer");
    const logs = data.logs || {};
    if (logs.folder && !state.logModel) {
      el("log-model-select").value = logs.folder;
    }
    const atBottom = viewer.scrollTop + viewer.clientHeight >= viewer.scrollHeight - 25;
    viewer.textContent = (logs.lines || []).join("\n") || "(Cap linia de log encara)";
    if (atBottom) viewer.scrollTop = viewer.scrollHeight;
  }

  async function fetchAndRender(force = false) {
    if (state.paused && !force) return;
    try {
      const qs = new URLSearchParams({
        window: String(state.windowMin),
        log_model: state.logModel,
      });
      const resp = await fetch(`/api/state?${qs.toString()}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      el("conn-dot").classList.remove("offline");
      el("last-sync").textContent = `Actualitzat ${new Date().toLocaleTimeString("ca-ES")}`;

      renderOperation(data.operation);
      renderActiveModel(data);
      renderKPIs(data.telemetry || {});
      renderModels(data.models || []);
      renderRequests((data.telemetry || {}).recent_requests || []);
      renderTimelineAndSummary(data.telemetry || {});
      renderTransitionsAndLogs(data);
    } catch (err) {
      el("conn-dot").classList.add("offline");
      el("last-sync").textContent = "Desconnectat";
    }
  }

  el("window-select").addEventListener("change", (e) => {
    state.windowMin = Number(e.target.value) || 3;
    fetchAndRender(true);
  });
  el("req-model-filter").addEventListener("change", (e) => {
    state.reqFilter = e.target.value;
    fetchAndRender(true);
  });
  el("log-model-select").addEventListener("change", (e) => {
    state.logModel = e.target.value;
    fetchAndRender(true);
  });
  el("toggle-pause-btn").addEventListener("click", () => {
    state.paused = !state.paused;
    el("toggle-pause-btn").textContent = state.paused ? "▶ Reprendre directe" : "⏸ Pausar directe";
    if (!state.paused) fetchAndRender(true);
  });
  el("refresh-now-btn").addEventListener("click", () => fetchAndRender(true));
  el("stop-all-btn").addEventListener("click", async () => {
    if (confirm("Segur que vols aturar tots els models en execucio?")) {
      await postJson("/api/stop", {});
    }
  });

  fetchAndRender(true);
  setInterval(() => fetchAndRender(false), 1500);
})();
