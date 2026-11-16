"""Zero-dependency HTML5 Canvas + WebSocket Simulation Cockpit UI served at GET /cockpit."""

from __future__ import annotations

COCKPIT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>DINOv2 + Laya 3D Simulation Cockpit</title>
  <style>
    :root {
      --bg: #0b101b;
      --panel: #131b2b;
      --panel2: #192338;
      --border: #273552;
      --text: #e6eefc;
      --muted: #8fa3c7;
      --accent: #38bdf8;
      --ok: #22c55e;
      --warn: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 10px 14px;
    }
    header {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 14px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 16px; }
    .badge {
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--accent);
    }
    .badge.ok { color: var(--ok); border-color: rgba(34,197,94,0.45); }
    .badge.warn { color: var(--warn); border-color: rgba(245,158,11,0.45); }
    .badge.danger { color: var(--danger); border-color: rgba(239,68,68,0.5); }
    .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    button, select {
      background: var(--panel2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 6px 11px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { border-color: var(--accent); }
    button.primary { background: #0284c7; border-color: #38bdf8; }
    button.danger { background: rgba(239,68,68,0.18); border-color: var(--danger); color: #fecaca; }
    button.active { background: rgba(34,197,94,0.20); border-color: var(--ok); color: #bbf7d0; }
    .grid-main {
      display: grid;
      grid-template-columns: 1.32fr 0.92fr 0.98fr 0.98fr;
      gap: 10px;
    }
    @media (max-width: 1420px) {
      .grid-main { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 900px) {
      .grid-main { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .card-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .cam-wrap {
      position: relative;
      background: #05080f;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
      aspect-ratio: 2 / 1;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    #camImg { width: 100%; height: 100%; object-fit: cover; display: block; }
    #camOverlay { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
    #bevCanvas, #navMapCanvas {
      width: 100%;
      height: 330px;
      background: #070c16;
      border-radius: 8px;
      border: 1px solid var(--border);
      display: block;
    }
    #navMapCanvas {
      height: 255px;
      cursor: crosshair;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
    }
    .kpi {
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 6px 8px;
    }
    .kpi-label { font-size: 10px; color: var(--muted); text-transform: uppercase; }
    .kpi-val { font-size: 16px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
    #sparkCanvas {
      width: 100%;
      height: 155px;
      background: #070c16;
      border-radius: 8px;
      border: 1px solid var(--border);
      display: block;
    }
    .deck {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
    }
    .deck-group {
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .deck-label { font-size: 11px; font-weight: 700; color: var(--muted); text-transform: uppercase; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 5px; }
    .bar-track { height: 7px; background: #0a1020; border-radius: 99px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--ok); transition: width 0.12s; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>DINOv2 + SpatialProjector + Laya Closed-Loop 3D Cockpit</span>
      <span id="connBadge" class="badge warn">CONNECTING</span>
      <span id="simBadge" class="badge">METADRIVE</span>
      <span id="latBadge" class="badge">0.0 ms</span>
    </div>
    <div class="toolbar">
      <button class="primary" onclick="sendCmd({action:'start'})">Start</button>
      <button onclick="sendCmd({action:'pause'})">Pause</button>
      <button onclick="sendCmd({action:'step'})">Step +1</button>
      <button onclick="sendCmd({action:'reset'})">Reset Episode</button>
      <button id="wasdBtn" onclick="toggleManual()">Manual WASD: OFF</button>
    </div>
  </header>

  <section class="grid-main">
    <div class="card">
      <div class="card-title">
        <span>3D Front / Chase Viewport + 6-WP E2E Trajectory</span>
        <span id="stepBadge">Ep 0 | Step 0</span>
      </div>
      <div class="cam-wrap">
        <img id="camImg" alt="Live 3D simulator feed (press Start or Step +1)" />
        <canvas id="camOverlay" width="640" height="320"></canvas>
      </div>
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">Maneuver (Gated)</div><div id="kManeuver" class="kpi-val">KEEP_LANE</div></div>
        <div class="kpi"><div class="kpi-label">Speed / Target</div><div id="kSpeed" class="kpi-val">0.0 / 40 km/h</div></div>
        <div class="kpi"><div class="kpi-label">AEB / Shield</div><div id="kShield" class="kpi-val">NOMINAL</div></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>360° BEV Radar / Lidar &amp; Blind-Spot Fusion</span>
        <span id="radarBadge">0 Targets</span>
      </div>
      <canvas id="bevCanvas" width="420" height="330"></canvas>
      <div style="display:flex; flex-direction:column; gap:4px;">
        <div style="display:flex; justify-content:space-between; font-size:11px; color:var(--muted);">
          <span>Camera Health Score (Fusion Auto-Fallback)</span>
          <span id="healthTxt">100%</span>
        </div>
        <div class="bar-track"><div id="healthBar" class="bar-fill" style="width:100%;"></div></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <span>ISO Safety &amp; Comfort Leaderboard</span>
        <span id="scoreBadge" class="badge ok">Score: 100.0%</span>
      </div>
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">Min TTC</div><div id="kTtc" class="kpi-val">99.0 s</div></div>
        <div class="kpi"><div class="kpi-label">RMS Long Jerk</div><div id="kJerk" class="kpi-val">0.00 m/s³</div></div>
        <div class="kpi"><div class="kpi-label">ISO Comfort</div><div id="kComfort" class="kpi-val">100.0%</div></div>
        <div class="kpi"><div class="kpi-label">Lead Range</div><div id="kLead" class="kpi-val">60.0 m</div></div>
        <div class="kpi"><div class="kpi-label">Progress</div><div id="kProg" class="kpi-val">0.0%</div></div>
        <div class="kpi"><div class="kpi-label">Infractions</div><div id="kInf" class="kpi-val">0</div></div>
      </div>
      <canvas id="sparkCanvas" width="440" height="155"></canvas>
    </div>

    <div class="card">
      <div class="card-title">
        <span>2D City A* Nav Map (Click to Re-Route)</span>
        <span id="navBadge" class="badge">HD_MAP</span>
      </div>
      <canvas id="navMapCanvas" width="420" height="255" title="Click anywhere or on a node to set a live destination"></canvas>
      <div class="kpi-grid">
        <div class="kpi"><div class="kpi-label">Map / Town</div><div id="kMap" class="kpi-val" style="font-size:13px;">Town10HD</div></div>
        <div class="kpi"><div class="kpi-label">Remaining</div><div id="kRem" class="kpi-val">68.0 m</div></div>
        <div class="kpi"><div class="kpi-label">Re-Routes / Tour</div><div id="kReroute" class="kpi-val">0 | L0</div></div>
      </div>
    </div>
  </section>

  <section class="deck">
    <div class="deck-group">
      <div class="deck-label">3D Simulator &amp; Stress Scenario</div>
      <div class="btn-row">
        <select id="simSel" onchange="sendCmd({action:'set_simulator', simulator:this.value})">
          <option value="metadrive">MetaDrive 0.4.3 (Real 3D)</option>
          <option value="carla">CARLA 0.9.16 (UE4)</option>
        </select>
        <select id="scSel" onchange="sendCmd({action:'set_scenario', scenario:this.value})">
          <option value="nominal_urban_cruise">1. Nominal Urban Cruise</option>
          <option value="aggressive_cut_in">2. Aggressive Cut-In</option>
          <option value="sudden_lead_ebrake">3. Sudden Lead E-Brake</option>
          <option value="blindspot_highspeed_overtaker">4. Blind-Spot Overtaker</option>
          <option value="occluded_jaywalker_dartout">5. Occluded Jaywalker</option>
          <option value="sotif_heavy_rain_glare_dropout">6. SOTIF Rain+Glare+Dropout</option>
        </select>
      </div>
      <div class="btn-row">
        <select id="percSel" onchange="sendCmd({action:'set_perception_mode', perception_mode:this.value})">
          <option value="vision_radar_fusion">Vision + 3D Radar Fusion</option>
          <option value="vision_estimated">Vision Only (Monocular)</option>
          <option value="gt">Simulator Ground Truth</option>
        </select>
        <button id="shieldBtn" class="active" onclick="toggleShield()">Shield: ON</button>
        <button id="actBtn" class="active" onclick="toggleActuator()">Actuator Lag: ON</button>
      </div>
    </div>

    <div class="deck-group">
      <div class="deck-label">Global A* Navigation, Multi-Town &amp; Mapless</div>
      <div class="btn-row">
        <select id="townSel" onchange="sendCmd({action:'set_town', town:this.value})">
          <option value="Town10HD_Opt">CARLA: Town10HD_Opt (Downtown)</option>
          <option value="Town03_Opt">CARLA: Town03_Opt (Roundabout/Slopes)</option>
          <option value="Town04_Opt">CARLA: Town04_Opt (8-Lane Highway)</option>
          <option value="Town05_Opt">CARLA: Town05_Opt (Multi-Lane Grid)</option>
          <option value="Town01_Opt">CARLA: Town01_Opt (Suburban)</option>
          <option value="Town02_Opt">CARLA: Town02_Opt (Compact Urban)</option>
        </select>
        <select id="blocksSel" onchange="sendCmd({action:'set_town', map_blocks:Number(this.value)})">
          <option value="1">MetaDrive: 1 Block</option>
          <option value="3">MetaDrive: 3 Blocks (Multi-Junction)</option>
          <option value="5">MetaDrive: 5 Blocks (Long Tour)</option>
        </select>
      </div>
      <div class="btn-row">
        <select id="locSel" onchange="sendCmd({action:'set_localization_mode', localization_mode:this.value})">
          <option value="hd_map">Loc: HD-Map Primary</option>
          <option value="hybrid_fallback">Loc: Hybrid Auto-Fallback</option>
          <option value="mapless_vision">Loc: 100% Mapless Vision+IMU</option>
        </select>
        <button class="primary" onclick="sendCmd({action:'random_destination'})">New Goal (200m+)</button>
        <button id="tourBtn" onclick="toggleEndlessTour()">Endless Tour: OFF</button>
        <button class="danger" onclick="sendCmd({action:'spawn_roadblock', distance_m:24})">+ Roadblock (Re-route)</button>
      </div>
    </div>

    <div class="deck-group">
      <div class="deck-label">Live Dynamic Hazard Injection</div>
      <div class="btn-row">
        <button class="danger" onclick="sendCmd({action:'inject_hazard', hazard:'cut_in'})">+ Cut-In Car</button>
        <button class="danger" onclick="sendCmd({action:'inject_hazard', hazard:'ebrake'})">+ Lead E-Brake</button>
        <button class="danger" onclick="sendCmd({action:'inject_hazard', hazard:'jaywalker'})">+ Jaywalker</button>
        <button onclick="sendCmd({action:'inject_hazard', hazard:'blindspot_left'})">+ Left BSM Car</button>
        <button onclick="sendCmd({action:'inject_hazard', hazard:'blindspot_right'})">+ Right BSM Car</button>
        <button onclick="sendCmd({action:'inject_hazard', hazard:'clear'})">Clear Hazards</button>
      </div>
    </div>

    <div class="deck-group">
      <div class="deck-label">Weather &amp; SOTIF Sensor Fault Injection</div>
      <div class="btn-row">
        <button onclick="sendCmd({action:'set_weather', weather:'clear'})">Clear Dry</button>
        <button onclick="sendCmd({action:'set_weather', weather:'wet'})">Wet Road</button>
        <button onclick="sendCmd({action:'set_weather', weather:'heavy_rain'})">Heavy Rain</button>
        <button onclick="sendCmd({action:'set_weather', weather:'night_rain'})">Night Rain</button>
      </div>
      <div class="btn-row">
        <button onclick="sendCmd({action:'inject_fault', fault_mode:'none'})">Sensors OK</button>
        <button onclick="sendCmd({action:'inject_fault', fault_mode:'rain_streaks'})">Rain Blur</button>
        <button onclick="sendCmd({action:'inject_fault', fault_mode:'sun_glare'})">Sun Glare</button>
        <button onclick="sendCmd({action:'inject_fault', fault_mode:'lens_occlusion'})">Lens Mud</button>
        <button class="danger" onclick="sendCmd({action:'inject_fault', fault_mode:'camera_dropout'})">Cam Blackout</button>
        <button onclick="sendCmd({action:'inject_fault', fault_mode:'radar_noise'})">Radar Jitter</button>
        <button class="danger" onclick="sendCmd({action:'inject_fault', fault_mode:'hd_map_dropout'})">HD-Map Dropout</button>
        <button class="danger" onclick="sendCmd({action:'inject_fault', fault_mode:'combined_sotif'})">Combined SOTIF</button>
      </div>
    </div>
  </section>

  <script>
    const proto = location.protocol === "https:" ? "wss" : "ws";
    let ws = null;
    let shieldOn = true;
    let actuatorOn = true;
    let endlessTourOn = false;
    let manualOn = false;
    let navViewport = { minX: -50, maxX: 250, minY: -120, maxY: 120, scale: 1, pad: 18, nodes: [] };
    const keysDown = new Set();
    const history = { speed: [], ttc: [], jerk: [] };

    function connectWS() {
      ws = new WebSocket(`${proto}://${location.host}/ws/sim`);
      ws.onopen = () => setBadge("connBadge", "LIVE WS", "ok");
      ws.onclose = () => {
        setBadge("connBadge", "RECONNECTING", "warn");
        setTimeout(connectWS, 1200);
      };
      ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data);
        if (msg.type === "status") applyStatus(msg.data);
        if (msg.type === "frame") applyFrame(msg.data);
      };
    }

    function sendCmd(obj) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(obj));
      } else {
        fetch("/v1/sim/control", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(obj),
        }).then(r => r.json()).then(d => {
          applyStatus(d);
          if (d.frame) applyFrame(d.frame);
        });
      }
    }

    function setBadge(id, text, cls) {
      const el = document.getElementById(id);
      el.textContent = text;
      el.className = "badge" + (cls ? " " + cls : "");
    }

    function applyStatus(st) {
      shieldOn = !!st.shield_enabled;
      actuatorOn = !!st.actuator_realism;
      endlessTourOn = !!st.endless_tour;
      document.getElementById("simSel").value = st.simulator || "metadrive";
      document.getElementById("scSel").value = st.scenario || "nominal_urban_cruise";
      document.getElementById("percSel").value = st.perception_mode || "vision_radar_fusion";
      if (st.town && document.getElementById("townSel")) {
        document.getElementById("townSel").value = st.town;
      }
      if (st.map_blocks && document.getElementById("blocksSel")) {
        document.getElementById("blocksSel").value = String(st.map_blocks);
      }
      if (st.localization_mode && document.getElementById("locSel")) {
        document.getElementById("locSel").value = st.localization_mode;
      }
      document.getElementById("shieldBtn").textContent = "Shield: " + (shieldOn ? "ON" : "OFF");
      document.getElementById("shieldBtn").className = shieldOn ? "active" : "danger";
      document.getElementById("actBtn").textContent = "Actuator Lag: " + (actuatorOn ? "ON" : "OFF");
      document.getElementById("actBtn").className = actuatorOn ? "active" : "";
      const tBtn = document.getElementById("tourBtn");
      if (tBtn) {
        tBtn.textContent = "Endless Tour: " + (endlessTourOn ? "ON" : "OFF");
        tBtn.className = endlessTourOn ? "active" : "";
      }
      const simTag = `${(st.simulator || "metadrive").toUpperCase()} ${st.is_real_3d ? "REAL 3D" : (st.env_initialized ? "MOCK" : "READY")}`;
      setBadge("simBadge", simTag, st.is_real_3d ? "ok" : "");
      if (st.nav_state && Object.keys(st.nav_state).length > 0) {
        drawNavMap(st.nav_state, st.localization_mode || "hd_map", {}, 0);
      }
      if (st.last_packet) applyFrame(st.last_packet);
    }

    function toggleShield() { sendCmd({ action: "set_shield", enabled: !shieldOn }); }
    function toggleActuator() { sendCmd({ action: "set_actuator_realism", enabled: !actuatorOn }); }
    function toggleEndlessTour() { sendCmd({ action: "set_endless_tour", enabled: !endlessTourOn }); }
    function toggleManual() {
      manualOn = !manualOn;
      document.getElementById("wasdBtn").textContent = "Manual WASD: " + (manualOn ? "ON" : "OFF");
      document.getElementById("wasdBtn").className = manualOn ? "danger" : "";
      pushManualControl();
    }

    function pushManualControl() {
      let steer = 0.0, tb = 0.0;
      if (keysDown.has("a") || keysDown.has("arrowleft")) steer += 0.55;
      if (keysDown.has("d") || keysDown.has("arrowright")) steer -= 0.55;
      if (keysDown.has("w") || keysDown.has("arrowup")) tb += 0.7;
      if (keysDown.has("s") || keysDown.has("arrowdown") || keysDown.has(" ")) tb -= 0.9;
      sendCmd({ action: "manual_override", enabled: manualOn, steering: steer, throttle_brake: tb });
    }
    window.addEventListener("keydown", (e) => {
      if (!manualOn) return;
      const k = e.key.toLowerCase();
      if (["w","a","s","d","arrowup","arrowdown","arrowleft","arrowright"," "].includes(k)) {
        e.preventDefault();
        if (!keysDown.has(k)) { keysDown.add(k); pushManualControl(); }
      }
    });
    window.addEventListener("keyup", (e) => {
      if (!manualOn) return;
      const k = e.key.toLowerCase();
      if (keysDown.delete(k)) pushManualControl();
    });

    function applyFrame(pkt) {
      if (pkt.frame_jpeg_b64) {
        document.getElementById("camImg").src = "data:image/jpeg;base64," + pkt.frame_jpeg_b64;
      }
      setBadge("simBadge", `${pkt.simulator.toUpperCase()} ${pkt.is_real_3d ? "REAL 3D" : "MOCK"}`, pkt.is_real_3d ? "ok" : "warn");
      setBadge("latBadge", `${pkt.latency_ms.toFixed(1)} ms`, pkt.latency_ms <= 40 ? "ok" : "warn");
      document.getElementById("stepBadge").textContent = `Ep ${pkt.episode_index} | Step ${pkt.step_index} | ${pkt.weather}`;

      const tel = pkt.telemetry || {};
      const ctrl = pkt.control || {};
      const dec = pkt.decision || {};
      const aux = pkt.aux_perception || {};
      const sc = pkt.safety_comfort || {};
      const ep = pkt.episode_summary || {};
      const inf = ep.infraction_counts || {};
      const totalInf = (inf.vehicle_collisions||0)+(inf.pedestrian_collisions||0)+(inf.red_light_infractions||0)+(inf.off_road_events||0)+(inf.lane_invasions||0);

      document.getElementById("kManeuver").textContent = String(ctrl.gated_maneuver || "keep_lane").toUpperCase();
      document.getElementById("kSpeed").textContent = `${Number(tel.speed_kmh||0).toFixed(1)} / ${Number(ctrl.target_speed_kmh||0).toFixed(0)} km/h`;
      const shEl = document.getElementById("kShield");
      shEl.textContent = ctrl.shield_triggered ? ctrl.shield_reason : `PASS (AEB ${(Number(dec.emergency_brake||0)*100).toFixed(0)}%)`;
      shEl.style.color = ctrl.shield_triggered ? "var(--danger)" : "var(--ok)";

      const hPct = Math.round(Number(aux.sensor_health_score ?? 1.0) * 100);
      document.getElementById("healthTxt").textContent = `${hPct}% (${aux.fusion_active ? "Fusion ON" : "No Radar"})`;
      const hBar = document.getElementById("healthBar");
      hBar.style.width = `${hPct}%`;
      hBar.style.background = hPct > 65 ? "var(--ok)" : (hPct > 35 ? "var(--warn)" : "var(--danger)");

      document.getElementById("kTtc").textContent = `${Number(ep.min_ttc_s ?? 99).toFixed(2)} s`;
      document.getElementById("kJerk").textContent = `${Number(ep.rms_long_jerk_mps3 ?? 0).toFixed(2)} m/s³`;
      document.getElementById("kComfort").textContent = `${Number(ep.iso_comfort_rate_pct ?? 100).toFixed(1)}%`;
      document.getElementById("kLead").textContent = `${Number(tel.lead_distance_m ?? 60).toFixed(1)} m`;
      document.getElementById("kProg").textContent = `${Number(pkt.route_completion_pct ?? 0).toFixed(1)}%`;
      document.getElementById("kInf").textContent = `${totalInf}`;
      const ds = Number(ep.driving_score_pct ?? 100);
      setBadge("scoreBadge", `Score: ${ds.toFixed(1)}%`, ds >= 80 ? "ok" : (ds >= 50 ? "warn" : "danger"));

      const targets = pkt.radar_targets || [];
      document.getElementById("radarBadge").textContent = `${targets.length} Targets | L:${tel.blind_spot_left?"OCC":"CLR"} R:${tel.blind_spot_right?"OCC":"CLR"}`;

      drawOverlayTraj(dec.predicted_trajectory || [], ctrl.shield_triggered);
      drawBEV(targets, dec.predicted_trajectory || [], tel);
      drawNavMap(
        pkt.nav_state || {},
        pkt.localization_mode || "hd_map",
        pkt.localization_diagnostics || {},
        Number(pkt.tour_legs_completed || 0)
      );

      history.speed.push(Number(tel.speed_kmh || 0));
      history.ttc.push(Math.min(10, Number(sc.ttc_s ?? 10)));
      history.jerk.push(Math.min(10, Math.abs(Number(sc.long_jerk_mps3 ?? 0))));
      for (const k of ["speed","ttc","jerk"]) if (history[k].length > 90) history[k].shift();
      drawSparklines();
    }

    function drawOverlayTraj(wps, shieldTriggered) {
      const c = document.getElementById("camOverlay");
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      if (!wps || wps.length === 0) return;
      ctx.strokeStyle = shieldTriggered ? "#ef4444" : "#22c55e";
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(c.width * 0.5, c.height - 8);
      for (const wp of wps) {
        const xm = Math.max(1, Number(wp[0] || 1));
        const ym = Number(wp[1] || 0);
        const px = c.width * 0.5 + (ym / xm) * 280;
        const py = c.height - Math.min(c.height * 0.72, Math.pow(xm / 35.0, 0.7) * c.height * 0.68);
        ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    function drawBEV(targets, wps, tel) {
      const c = document.getElementById("bevCanvas");
      const ctx = c.getContext("2d");
      const W = c.width, H = c.height;
      ctx.clearRect(0, 0, W, H);
      const cx = W * 0.5, cy = H * 0.68;
      const pxPerM = 4.2;

      // Range rings
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      for (const rM of [15, 30, 45]) {
        ctx.beginPath();
        ctx.arc(cx, cy, rM * pxPerM, 0, Math.PI * 2);
        ctx.stroke();
      }
      // 3-lane guides
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = "#334155";
      for (const offM of [-5.25, -1.75, 1.75, 5.25]) {
        ctx.beginPath();
        ctx.moveTo(cx + offM * pxPerM, 10);
        ctx.lineTo(cx + offM * pxPerM, H - 10);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // BSM side radar zones (+y is right, -y is left)
      const drawBSM = (side, occ) => {
        ctx.fillStyle = occ ? "rgba(239,68,68,0.38)" : "rgba(34,197,94,0.12)";
        ctx.strokeStyle = occ ? "#ef4444" : "#15803d";
        const x0 = cx + (side < 0 ? -4.5 * pxPerM : 1.5 * pxPerM);
        const w = 3.0 * pxPerM;
        const y0 = cy - 7.5 * pxPerM;
        const h = 18.0 * pxPerM;
        ctx.fillRect(x0, y0, w, h);
        ctx.strokeRect(x0, y0, w, h);
      };
      drawBSM(-1, !!tel.blind_spot_left);
      drawBSM(1, !!tel.blind_spot_right);

      // Predicted 6-WP trajectory
      if (wps && wps.length) {
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        for (const wp of wps) {
          ctx.lineTo(cx + Number(wp[1] || 0) * pxPerM, cy - Number(wp[0] || 0) * pxPerM);
        }
        ctx.stroke();
      }

      // Ego vehicle
      ctx.fillStyle = "#22c55e";
      ctx.fillRect(cx - 8, cy - 15, 16, 30);

      // Radar targets
      for (const t of targets) {
        const tx = cx + Number(t.y_m || 0) * pxPerM;
        const ty = cy - Number(t.x_m || 0) * pxPerM;
        const isPed = t.category === "pedestrian";
        ctx.fillStyle = isPed ? "#fb923c" : (t.source === "scenario_hazard" ? "#ef4444" : "#60a5fa");
        ctx.beginPath();
        ctx.arc(tx, ty, isPed ? 5 : 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#e2e8f0";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(tx + Number(t.vy_rel_mps||0) * 2.5, ty - Number(t.vx_rel_mps||0) * 2.5);
        ctx.stroke();
      }
    }

    function drawSparklines() {
      const c = document.getElementById("sparkCanvas");
      const ctx = c.getContext("2d");
      const W = c.width, H = c.height;
      ctx.clearRect(0, 0, W, H);
      const plot = (arr, maxV, color, label, yLabel) => {
        if (arr.length < 2) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        arr.forEach((v, i) => {
          const x = (i / 89) * (W - 20) + 10;
          const y = H - 14 - Math.min(1, Math.max(0, v / maxV)) * (H - 30);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = "11px sans-serif";
        ctx.fillText(label, 12, yLabel);
      };
      plot(history.speed, 80, "#38bdf8", "Speed (0-80 km/h)", 16);
      plot(history.ttc, 10, "#22c55e", "TTC (0-10 s)", 30);
      plot(history.jerk, 10, "#f59e0b", "|Long Jerk| (0-10 m/s^3)", 44);
    }

    function drawNavMap(nav, locMode, locDiag, tourLegs) {
      const c = document.getElementById("navMapCanvas");
      if (!c) return;
      const ctx = c.getContext("2d");
      const W = c.width, H = c.height;
      ctx.clearRect(0, 0, W, H);

      const activeSource = String((locDiag && locDiag.localization_source) || locMode || "hd_map").toUpperCase();
      const isMapless = activeSource.includes("MAPLESS") || String(locMode).includes("mapless");
      setBadge("navBadge", activeSource.replace("MAPLESS_", "VIS_"), isMapless ? "warn" : "ok");

      document.getElementById("kMap").textContent = String(nav.map_name || "Town10HD").slice(0, 15);
      document.getElementById("kRem").textContent = `${Number(nav.remaining_route_m ?? 68).toFixed(1)} m`;
      document.getElementById("kReroute").textContent = `${Number(nav.reroute_count ?? 0)} | L${Number(tourLegs || 0)}`;

      const routePts = Array.isArray(nav.active_route_xy) ? nav.active_route_xy : [];
      const nodes = Array.isArray(nav.spawn_nodes) ? nav.spawn_nodes : [];
      const ego = Array.isArray(nav.ego_xy_yaw) && nav.ego_xy_yaw.length >= 2 ? nav.ego_xy_yaw : [0, 0, 0];
      const dest = Array.isArray(nav.dest_xy) && nav.dest_xy.length >= 2 ? nav.dest_xy : null;

      const xs = [Number(ego[0] || 0)];
      const ys = [Number(ego[1] || 0)];
      for (const p of routePts) { xs.push(Number(p[0] || 0)); ys.push(Number(p[1] || 0)); }
      for (const n of nodes) { xs.push(Number(n.x || 0)); ys.push(Number(n.y || 0)); }
      if (dest) { xs.push(Number(dest[0] || 0)); ys.push(Number(dest[1] || 0)); }

      let minX = Math.min(...xs) - 18, maxX = Math.max(...xs) + 18;
      let minY = Math.min(...ys) - 18, maxY = Math.max(...ys) + 18;
      if (maxX - minX < 40) { minX -= 20; maxX += 20; }
      if (maxY - minY < 40) { minY -= 20; maxY += 20; }
      const pad = 18;
      const sx = (W - 2 * pad) / (maxX - minX);
      const sy = (H - 2 * pad) / (maxY - minY);
      const scale = Math.min(sx, sy);
      const offX = (W - (maxX - minX) * scale) * 0.5;
      const offY = (H - (maxY - minY) * scale) * 0.5;
      navViewport = { minX, maxX, minY, maxY, scale, offX, offY, W, H, nodes };

      const toPx = (wx, wy) => [
        offX + (wx - minX) * scale,
        offY + (wy - minY) * scale,
      ];

      // Subtle background grid
      ctx.strokeStyle = "#111a2c";
      ctx.lineWidth = 1;
      for (let gx = 40; gx < W; gx += 45) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke(); }
      for (let gy = 35; gy < H; gy += 45) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke(); }

      // Clickable spawn / intersection nodes
      for (const n of nodes) {
        const [px, py] = toPx(Number(n.x || 0), Number(n.y || 0));
        ctx.fillStyle = "rgba(148,163,184,0.45)";
        ctx.beginPath();
        ctx.arc(px, py, 3.3, 0, Math.PI * 2);
        ctx.fill();
      }

      // Active A* global route polyline
      if (routePts.length >= 2) {
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2.8;
        ctx.beginPath();
        routePts.forEach((p, idx) => {
          const [px, py] = toPx(Number(p[0] || 0), Number(p[1] || 0));
          if (idx === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      // Destination pin
      if (dest) {
        const [dx, dy] = toPx(Number(dest[0]), Number(dest[1]));
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(dx, dy, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = "#f59e0b";
        ctx.beginPath();
        ctx.arc(dx, dy, 3.8, 0, Math.PI * 2);
        ctx.fill();
      }

      // Ego directional arrow
      const [ex, ey] = toPx(Number(ego[0] || 0), Number(ego[1] || 0));
      const yaw = Number(ego[2] || 0);
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(yaw);
      ctx.fillStyle = "#22c55e";
      ctx.beginPath();
      ctx.moveTo(9, 0);
      ctx.lineTo(-6, -5.5);
      ctx.lineTo(-4, 0);
      ctx.lineTo(-6, 5.5);
      ctx.closePath();
      ctx.fill();
      ctx.restore();

      ctx.fillStyle = "#8fa3c7";
      ctx.font = "10px sans-serif";
      ctx.fillText("Click map or node -> live A* re-route", 8, H - 7);
    }

    document.getElementById("navMapCanvas").addEventListener("click", (evt) => {
      const c = evt.currentTarget;
      const rect = c.getBoundingClientRect();
      const px = ((evt.clientX - rect.left) / Math.max(1, rect.width)) * c.width;
      const py = ((evt.clientY - rect.top) / Math.max(1, rect.height)) * c.height;
      const wx = navViewport.minX + (px - (navViewport.offX || 18)) / Math.max(1e-5, navViewport.scale || 1);
      const wy = navViewport.minY + (py - (navViewport.offY || 18)) / Math.max(1e-5, navViewport.scale || 1);

      let nearestIdx = null;
      let bestDistPx = 14.0;
      for (const n of (navViewport.nodes || [])) {
        const nx = (navViewport.offX || 18) + (Number(n.x || 0) - navViewport.minX) * (navViewport.scale || 1);
        const ny = (navViewport.offY || 18) + (Number(n.y || 0) - navViewport.minY) * (navViewport.scale || 1);
        const d = Math.hypot(nx - px, ny - py);
        if (d < bestDistPx) {
          bestDistPx = d;
          nearestIdx = Number(n.index);
        }
      }
      if (nearestIdx !== null && Number.isFinite(nearestIdx)) {
        sendCmd({ action: "set_destination", dest_index: nearestIdx });
      } else {
        sendCmd({ action: "set_destination", dest_xy: [Number(wx.toFixed(2)), Number(wy.toFixed(2))] });
      }
    });

    connectWS();
  </script>
</body>
</html>
"""

__all__ = ["COCKPIT_HTML"]
