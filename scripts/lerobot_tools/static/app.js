/* Offline browser UI; the original synchronized grid remains visible after edits. */
"use strict";

function mergeIntervals(intervals) {
  const result = [];
  for (const [a, b] of intervals.map(x => [...x]).sort((a, b) => a[0] - b[0])) {
    if (result.length && a <= result[result.length - 1][1]) result[result.length - 1][1] = Math.max(b, result[result.length - 1][1]);
    else result.push([a, b]);
  }
  return result;
}
function isExcluded(seconds, edit) {
  const ns = Math.round(seconds * 1e9);
  return !!edit.drop || (edit.exclude || []).some(([a, b]) => ns >= Math.round(a * 1e9) && ns <= Math.round(b * 1e9));
}
function keptSummary(seconds, edit) {
  let count = 0, episodes = 0, previous = false;
  for (const t of seconds) {
    const keep = !isExcluded(t, edit);
    if (keep) { count++; if (!previous) episodes++; }
    previous = keep;
  }
  return {count, episodes};
}
function playbackIndex(anchorIndex, elapsedMs, fps, speed, lastIndex) {
  return Math.min(lastIndex, anchorIndex + Math.floor(Math.max(0, elapsedMs) * fps * speed / 1000));
}
if (typeof module !== "undefined") module.exports = {mergeIntervals, isExcluded, keptSummary, playbackIndex};

if (typeof document !== "undefined") {
  const $ = id => document.getElementById(id);
  const clone = obj => JSON.parse(JSON.stringify(obj));
  const state = {entries: [], plan: {version: 1, recordings: {}}, revision: "", savedPlan: "", config: {},
    name: null, timeline: null, index: 0, displayed: -1, playing: false, speed: 1, anchor: 0, anchorAt: 0,
    dirty: false, saving: false, loading: false, inflight: false, frameFailed: false, generation: 0, history: [], poll: null, markerSignature: null};
  const colors = ["#68dfbc", "#89b6ff", "#d6a5ff", "#f7b978", "#6bd7eb", "#e9a1b0", "#bbd773", "#ffd46e"];
  const error = message => { $("error").textContent = message || ""; $("error").hidden = !message; };
  async function api(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let text = await response.text();
      try { text = JSON.parse(text).detail || text; } catch (_) { /* text response */ }
      throw new Error(typeof text === "string" ? text : JSON.stringify(text));
    }
    return response.json();
  }
  function edit() { return state.plan.recordings[state.name] || {drop: false, exclude: []}; }
  function markDirty() {
    state.dirty = JSON.stringify(state.plan) !== state.savedPlan;
    $("saved").textContent = state.dirty ? "未保存" : "已保存";
    $("saved").classList.toggle("dirty", state.dirty);
    $("save").disabled = state.saving || !state.dirty;
    $("export").disabled = state.dirty || state.saving || !!state.exporting;
  }
  function mutate(fn) {
    if (!state.timeline || state.saving || state.playing || state.loading) return;
    state.history.push(clone(state.plan));
    if (state.history.length > 100) state.history.shift();
    const value = clone(edit());
    fn(value);
    value.fingerprint = state.timeline.fingerprint;
    state.plan.recordings[state.name] = value;
    markDirty(); renderClips(); renderList(); draw();
  }
  function shortName(name) {
    const m = name.match(/recording_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_/);
    return m ? `${m[1]}-${m[2]}-${m[3]}  ${m[4]}:${m[5]}:${m[6]}` : name;
  }
  function renderList() {
    const query = $("search").value.toLowerCase();
    const validOnly = $("validOnly").checked;
    $("recordings").replaceChildren();
    for (const entry of state.entries.filter(e => (!validOnly || e.valid) && e.name.toLowerCase().includes(query))) {
      const button = document.createElement("button");
      button.className = "record" + (entry.name === state.name ? " selected" : "") + (!entry.valid ? " invalid" : "");
      button.title = entry.name + (entry.error ? "\n" + entry.error : "");
      const strong = document.createElement("strong"); strong.textContent = shortName(entry.name); button.append(strong);
      const meta = document.createElement("span");
      meta.textContent = entry.valid ? `${entry.duration.toFixed(1)} s · ${entry.frames} 帧` : "异常录制 · 将跳过";
      button.append(meta);
      const savedEdit = state.plan.recordings[entry.name];
      if (savedEdit && (savedEdit.drop || savedEdit.exclude.length)) {
        const badge = document.createElement("span"); badge.className = "badge";
        badge.textContent = savedEdit.drop ? "整条剔除" : `${savedEdit.exclude.length} 处剪辑`; button.append(badge);
      }
      button.onclick = () => selectRecording(entry.name);
      $("recordings").append(button);
    }
  }
  async function selectRecording(name) {
    pause(); state.generation++; const generation = state.generation;
    state.name = name; state.timeline = null; state.index = 0; state.displayed = -1; state.frameFailed = false;
    state.loading = true; $("play").disabled = true; $("seek").max = "0";
    for (const camera of ["kinec2", "realsense"]) { $(camera).classList.remove("loaded"); $(camera).removeAttribute("src"); }
    $("title").textContent = name; error(""); renderList(); renderClips(); updateControls(); draw();
    const entry = state.entries.find(e => e.name === name);
    if (!entry.valid) { error(entry.error); state.loading = false; $("subtitle").textContent = "该录制将被导出流程跳过"; return; }
    try {
      const data = await api(`/api/recordings/${encodeURIComponent(name)}/timeline`);
      if (generation !== state.generation) return;
      state.timeline = data; state.loading = false; $("play").disabled = false;
      $("seek").max = String(data.frames - 1); $("seek").value = "0";
      $("subtitle").textContent = `${data.frames} 帧 · ${state.config.fps} FPS · 原始录制时间轴`;
      const gaps = Object.values(data.gaps).reduce((sum, list) => sum + list.length, 0);
      $("syncInfo").textContent = gaps ? `${gaps} 处采样缺口，沿用上一有效样本` : "各路按时间戳同步";
      $("start").value = ""; $("end").value = "";
      renderClips(); updateControls(); draw(); requestFrame();
    } catch (exc) { if (generation === state.generation) { state.loading = false; error(exc.message); } }
  }
  function pause(freezeDisplayed = true) {
    if (freezeDisplayed && state.playing && state.displayed >= 0) state.index = state.displayed;
    state.playing = false; $("play").textContent = "播放"; updateControls();
  }
  function play() {
    if (!state.timeline) return;
    if (state.index === state.timeline.frames - 1) state.index = 0;
    state.frameFailed = false; state.anchor = state.index; state.anchorAt = performance.now(); state.playing = true;
    $("play").textContent = "暂停"; updateControls();
  }
  function seek(index) {
    pause();
    if (!state.timeline) return;
    state.index = Math.max(0, Math.min(state.timeline.frames - 1, index));
    state.frameFailed = false;
    requestFrame();
  }
  function updateControls() {
    const disabled = !state.timeline || state.playing || state.loading || state.saving;
    for (const id of ["start", "end", "markStart", "markEnd", "addClip", "undo", "drop"]) $(id).disabled = disabled;
    $("prev").disabled = !state.timeline || state.loading;
    $("next").disabled = !state.timeline || state.loading;
    $("undo").disabled = disabled || !state.history.length;
    document.querySelectorAll(".clip button").forEach(button => { button.disabled = disabled; });
  }
  async function requestFrame() {
    if (!state.timeline || state.inflight || state.frameFailed || state.index === state.displayed) return;
    const index = state.index, name = state.name, generation = state.generation;
    state.inflight = true;
    try {
      const data = await api(`/api/recordings/${encodeURIComponent(name)}/frame?index=${index}`);
      if (generation !== state.generation || name !== state.name || (!state.playing && index !== state.index)) return;
      // Decode both before replacing either; old requests cannot overwrite a new selection.
      const images = await Promise.all(["kinec2", "realsense"].map(camera => new Promise((resolve, reject) => {
        const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = data.images[camera];
      })));
      if (generation !== state.generation || (!state.playing && index !== state.index)) return;
      for (const [i, camera] of ["kinec2", "realsense"].entries()) { $(camera).src = images[i].src; $(camera).classList.add("loaded"); }
      state.displayed = index;
      const t = data.seconds;
      $("time").textContent = `${t.toFixed(3)} / ${state.timeline.end.toFixed(3)} s`;
      $("frame").textContent = `帧 ${index + 1} / ${state.timeline.frames}`;
      $("seek").value = String(index);
      $("kinectAge").textContent = `源样本距前帧 ${data.age_ms.kinec2_rgb.toFixed(0)} ms`;
      $("realsenseAge").textContent = `源样本距前帧 ${data.age_ms.realsense_rgb.toFixed(0)} ms`;
      for (let j = 0; j < 8; j++) $("value" + j).textContent = data.action[j].toFixed(5);
      draw(index);
    } catch (exc) { if (generation === state.generation) { pause(); state.frameFailed = true; error("画面加载失败：" + (exc.message || "无法解码图像") + "；逐帧或播放可重试。"); } }
    finally { state.inflight = false; }
  }
  function tick(now) {
    if (state.playing && state.timeline) {
      state.index = playbackIndex(state.anchor, now - state.anchorAt, state.config.fps, state.speed, state.timeline.frames - 1);
      if (state.index === state.timeline.frames - 1) pause(false);
    }
    requestFrame(); requestAnimationFrame(tick);
  }
  function renderClips() {
    $("clips").replaceChildren();
    if (!state.timeline) { $("clipInfo").textContent = "—"; return; }
    const item = edit(), summary = keptSummary(state.timeline.seconds, item);
    $("drop").textContent = item.drop ? "恢复整条" : "剔除整条";
    $("clipInfo").textContent = `保留 ${summary.count} / ${state.timeline.frames} 帧 · 导出 ${summary.episodes} 个 episode`;
    if (item.drop) { const info = document.createElement("span"); info.className = "clip"; info.textContent = "整条录制已剔除，已有区间标记保留"; $("clips").append(info); }
    (item.exclude || []).forEach(([a, b], i) => {
      const chip = document.createElement("span"); chip.className = "clip";
      chip.textContent = `${a.toFixed(3)} → ${b.toFixed(3)} s`;
      const remove = document.createElement("button"); remove.textContent = "×"; remove.title = "移除此删除区间";
      remove.onclick = () => mutate(value => value.exclude.splice(i, 1)); chip.append(remove); $("clips").append(chip);
    });
    updateControls();
  }
  function setupCharts() {
    for (let j = 0; j < 8; j++) {
      const div = document.createElement("div"); div.className = "chart";
      const label = document.createElement("div"); label.className = "chart-label";
      const text = document.createElement("span"); text.textContent = j < 7 ? `JOINT ${j + 1} · rad` : "GRIPPER · close ratio";
      const value = document.createElement("span"); value.className = "chart-value"; value.id = "value" + j; value.textContent = "—";
      label.append(text, value); const canvas = document.createElement("canvas"); canvas.id = "chart" + j;
      canvas.onclick = event => { if (!state.timeline) return; const rect = canvas.getBoundingClientRect(); seek(Math.round((event.clientX - rect.left) / rect.width * (state.timeline.frames - 1))); };
      const plot = document.createElement("div"); plot.className = "chart-plot";
      const markers = document.createElement("div"); markers.className = "selection-markers";
      markers.setAttribute("data-selection-markers", ""); markers.setAttribute("aria-label", "待选区间端点");
      plot.append(canvas, markers); div.append(label, plot); $("charts").append(div);
    }
  }
  function prepare(canvas) {
    const ratio = window.devicePixelRatio || 1, width = canvas.clientWidth, height = canvas.clientHeight;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio); return {ctx, width, height};
  }
  function shade(ctx, width, height) {
    const timeline = state.timeline;
    if (!timeline) return;
    const start = timeline.seconds[0], span = timeline.end - start || 1, item = edit();
    ctx.fillStyle = "#fb8b9140";
    if (item.drop) ctx.fillRect(0, 0, width, height);
    else for (const [a, b] of item.exclude || []) {
      const x = Math.max(0, (a - start) / span * width), right = Math.min(width, (b - start) / span * width);
      ctx.fillRect(x, 0, Math.max(2, right - x), height);
    }
  }
  function draftBounds() {
    if (!state.timeline) return {};
    const bounds = {};
    for (const key of ["start", "end"]) {
      const input = $(key).value.trim(), value = Number(input);
      if (input && Number.isFinite(value) && value >= 0 && value <= state.timeline.duration + 1e-9) bounds[key] = value;
    }
    return bounds;
  }
  function selectionPosition(seconds) {
    const first = state.timeline.seconds[0], span = state.timeline.end - first || 1;
    return Math.max(0, Math.min(1, (seconds - first) / span));
  }
  function renderDraftMarkers() {
    const signature = JSON.stringify([state.name, !!state.timeline, $("start").value, $("end").value]);
    if (state.markerSignature === signature) return;
    state.markerSignature = signature;
    const bounds = draftBounds();
    document.querySelectorAll("[data-selection-markers]").forEach(container => {
      container.replaceChildren();
      for (const key of ["start", "end"]) {
        if (bounds[key] === undefined) continue;
        const position = selectionPosition(bounds[key]);
        const marker = document.createElement("span");
        marker.className = "draft-marker " + key;
        marker.setAttribute("data-boundary", key);
        marker.style.left = `${position * 100}%`;
        const label = document.createElement("span"); label.className = "draft-label";
        label.textContent = `${key === "start" ? "Start" : "End"} ${bounds[key].toFixed(3)} s`;
        label.style.transform = position > .75 ? "translateX(-100%)" : position < .25 ? "none" : "translateX(-50%)";
        marker.append(label); container.append(marker);
      }
    });
  }
  function shadeDraft(ctx, width, height) {
    const bounds = draftBounds();
    if (bounds.start === undefined || bounds.end === undefined || bounds.end < bounds.start) return;
    const left = selectionPosition(bounds.start) * width, right = selectionPosition(bounds.end) * width;
    ctx.fillStyle = "#ffd46e30";
    ctx.fillRect(left, 0, Math.max(2, right - left), height);
  }
  function draw(index = state.displayed < 0 ? state.index : state.displayed) {
    renderDraftMarkers();
    const ruler = prepare($("selectionCanvas")); ruler.ctx.fillStyle = "#283847"; ruler.ctx.fillRect(0, 0, ruler.width, ruler.height); shade(ruler.ctx, ruler.width, ruler.height); shadeDraft(ruler.ctx, ruler.width, ruler.height);
    for (let j = 0; j < 8; j++) {
      const {ctx, width, height} = prepare($("chart" + j));
      if (!state.timeline) continue;
      shade(ctx, width, height);
      shadeDraft(ctx, width, height);
      const values = state.timeline.action.map(row => row[j]);
      let min = Infinity, max = -Infinity;
      for (const value of values) { min = Math.min(min, value); max = Math.max(max, value); }
      const pad = Math.max((max - min) * .12, .01); min -= pad; max += pad;
      ctx.strokeStyle = "#273546"; ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
      ctx.strokeStyle = colors[j]; ctx.lineWidth = 1.3; ctx.beginPath();
      for (let i = 0; i < values.length; i++) {
        const x = i / Math.max(1, values.length - 1) * width, y = height - 4 - (values[i] - min) / (max - min) * (height - 8);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      }
      ctx.stroke(); const x = index / Math.max(1, values.length - 1) * width;
      ctx.strokeStyle = "#ecf3ff99"; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
  }
  async function pollExport() {
    try {
      const data = await api("/api/export"), job = data.job;
      state.exporting = !["idle", "complete", "failed"].includes(job.phase);
      $("export").textContent = state.exporting ? "正在导出…" : "导出已保存方案";
      const descriptions = {idle: "等待导出", queued: "已加入导出队列", scan: "检查源录制", write: "正在写入数据和视频", validate: "正在验证数据集", complete: "导出完成", failed: "导出失败"};
      let text = descriptions[job.phase] || job.phase;
      if (state.exporting && job.total_frames) text += ` · ${job.frames || 0} / ${job.total_frames} 帧`;
      if (job.error) text += `：${job.error}`;
      if (data.exists) text += data.stale ? "\n当前数据集与已保存方案不同，需要重新导出。" : `\n已保存方案已同步 · ${data.current.total_episodes} episodes · ${data.current.total_frames} 帧`;
      else text += "\n尚无已验证的数据集。";
      if (state.dirty) text += "\n浏览器还有未保存编辑。";
      const skipped = job.skipped || (data.current && data.current.skipped) || [];
      if (skipped.length) text += "\n跳过：" + skipped.map(row => row.name + " (" + row.reason + ")").join("；");
      $("exportStatus").textContent = text;
      $("progress").hidden = !state.exporting; $("progress").value = job.total_frames ? (job.frames || 0) / job.total_frames : 0;
      markDirty();
    } catch (exc) { $("exportStatus").textContent = "无法读取导出状态：" + exc.message; }
  }
  async function reloadPlan() {
    const data = await api("/api/plan");
    state.plan = data.plan; state.revision = data.revision; state.savedPlan = JSON.stringify(state.plan); state.history = [];
    markDirty(); renderClips(); renderList(); draw(); await pollExport();
  }
  $("play").onclick = () => state.playing ? pause() : play();
  $("prev").onclick = () => seek((state.displayed < 0 ? state.index : state.displayed) - 1);
  $("next").onclick = () => seek((state.displayed < 0 ? state.index : state.displayed) + 1);
  $("seek").oninput = event => seek(Number(event.target.value));
  $("speed").onchange = event => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value) || value < .25 || value > 4) { event.target.value = String(state.speed); error("播放倍速需在 0.25–4 之间"); return; }
    state.speed = value;
    if (state.playing) { state.anchor = state.index; state.anchorAt = performance.now(); }
  };
  $("markStart").onclick = () => { if (state.timeline) { $("start").value = String(state.timeline.seconds[state.displayed < 0 ? state.index : state.displayed]); draw(); } };
  $("markEnd").onclick = () => { if (state.timeline) { $("end").value = String(state.timeline.seconds[state.displayed < 0 ? state.index : state.displayed]); draw(); } };
  $("start").oninput = () => draw();
  $("end").oninput = () => draw();
  $("addClip").onclick = () => {
    if (!state.timeline) return;
    const a = Number($("start").value), b = Number($("end").value);
    if (!$("start").value || !$("end").value || !Number.isFinite(a) || !Number.isFinite(b) || a < 0 || b < a || b > state.timeline.duration + 1e-9) { error("请输入有效区间：0 ≤ Start ≤ End ≤ 录制时长"); return; }
    error(""); mutate(value => { value.exclude = mergeIntervals([...(value.exclude || []), [a, b]]); });
  };
  $("drop").onclick = () => mutate(value => { value.drop = !value.drop; });
  $("undo").onclick = () => {
    if (state.playing || state.saving || !state.history.length) return;
    state.plan = state.history.pop(); markDirty(); renderClips(); renderList(); draw();
  };
  $("save").onclick = async () => {
    pause(); state.saving = true; markDirty(); updateControls(); error("");
    try {
      const data = await api("/api/plan", {method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({plan: state.plan, base_revision: state.revision})});
      state.plan = data.plan; state.revision = data.revision; state.savedPlan = JSON.stringify(state.plan);
    } catch (exc) { error("保存失败：" + exc.message); }
    finally { state.saving = false; markDirty(); updateControls(); await pollExport(); }
  };
  $("reload").onclick = async () => {
    if (state.saving) return;
    if (state.dirty && !window.confirm("放弃浏览器中未保存的编辑，重新载入已保存方案？")) return;
    pause(); state.saving = true; markDirty(); updateControls();
    try { await reloadPlan(); error(""); } catch (exc) { error(exc.message); }
    finally { state.saving = false; markDirty(); updateControls(); }
  };
  $("export").onclick = async () => {
    if (state.dirty || state.exporting) return;
    $("export").disabled = true; error("");
    try { await api("/api/export", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({revision: state.revision})}); }
    catch (exc) { error("导出未启动：" + exc.message); }
    await pollExport();
  };
  $("search").oninput = renderList; $("validOnly").onchange = renderList;
  window.addEventListener("resize", () => draw());
  window.addEventListener("beforeunload", event => { if (state.dirty) { event.preventDefault(); event.returnValue = ""; } });
  document.addEventListener("keydown", event => {
    if (["INPUT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)) return;
    if (event.code === "Space") { event.preventDefault(); state.playing ? pause() : play(); }
    if (event.code === "ArrowLeft") { event.preventDefault(); seek(state.index - 1); }
    if (event.code === "ArrowRight") { event.preventDefault(); seek(state.index + 1); }
  });
  async function init() {
    setupCharts();
    try {
      const data = await api("/api/recordings"); state.entries = data.recordings; state.config = data.config;
      $("selection").textContent = `${data.recordings.filter(e => e.valid).length} 有效 / ${data.recordings.length} 条 · ${data.config.since}`;
      $("output").textContent = `${data.config.output} · task: ${data.config.task}`;
      try { const player = await api("/api/player"); state.speed = player.speed; $("speed").value = String(player.speed); } catch (_) { /* test app defaults to 1x */ }
      await reloadPlan();
      const first = data.recordings.find(e => e.valid);
      if (first) await selectRecording(first.name); else error("没有可播放的有效录制，请检查输入目录和起始时间。");
      state.poll = setInterval(pollExport, 1500); requestAnimationFrame(tick);
    } catch (exc) { error("初始化失败：" + exc.message); }
  }
  init();
}
