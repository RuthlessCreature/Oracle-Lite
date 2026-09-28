from __future__ import annotations

import copy
import json
import logging
import shutil
import threading
import time
import webbrowser
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gb(value: int | float) -> float:
    return round(float(value) / (1024 ** 3), 2)


class TrainingMonitor:
    """Thread-safe runtime state for the local training console."""

    def __init__(self, output_dir: str | Path, *, max_logs: int = 1200):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_path = self.log_dir / f"training-console-{self.session_id}.jsonl"
        self._lock = threading.RLock()
        self._logs: deque[dict[str, Any]] = deque(maxlen=max_logs)
        self._stop_event = threading.Event()
        self._sampler_thread: threading.Thread | None = None
        self._started_monotonic = time.monotonic()
        self._state: dict[str, Any] = {
            "status": "starting",
            "phase": "bootstrap",
            "phase_label": "Starting Oracle-Lite",
            "started_at": _utcnow(),
            "updated_at": _utcnow(),
            "finished_at": None,
            "error": None,
            "dashboard": {},
            "corpus": {},
            "dataset": {},
            "model": {},
            "training": {
                "step": 0,
                "total_steps": 0,
                "progress_percent": 0.0,
                "epoch": None,
                "loss": None,
                "learning_rate": None,
                "grad_norm": None,
                "elapsed_seconds": 0,
                "eta_seconds": None,
                "checkpoint": None,
            },
            "system": {
                "cpu_percent": None,
                "cpu_count": None,
                "ram_total_gb": None,
                "ram_used_gb": None,
                "ram_percent": None,
                "process_rss_gb": None,
                "disk_total_gb": None,
                "disk_used_gb": None,
                "disk_free_gb": None,
                "disk_percent": None,
                "gpus": [],
                "gpu_monitor_error": None,
            },
        }

    def start_sampling(self, interval_seconds: float = 1.0) -> None:
        if self._sampler_thread and self._sampler_thread.is_alive():
            return
        self._sampler_thread = threading.Thread(
            target=self._sample_loop,
            args=(interval_seconds,),
            name="oracle-lite-system-sampler",
            daemon=True,
        )
        self._sampler_thread.start()

    def stop_sampling(self) -> None:
        self._stop_event.set()
        if self._sampler_thread and self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=2.0)

    def set_dashboard(self, *, url: str, host: str, port: int) -> None:
        self.update("dashboard", {"url": url, "host": host, "port": port})

    def update_phase(self, phase: str, label: str, *, status: str = "running") -> None:
        with self._lock:
            self._state["phase"] = phase
            self._state["phase_label"] = label
            self._state["status"] = status
            self._state["updated_at"] = _utcnow()
        self.log("INFO", label)

    def update(self, section: str, values: dict[str, Any]) -> None:
        with self._lock:
            target = self._state.setdefault(section, {})
            if isinstance(target, dict):
                target.update(values)
            else:
                self._state[section] = dict(values)
            self._state["updated_at"] = _utcnow()

    def update_training(self, **values: Any) -> None:
        elapsed = max(0, int(time.monotonic() - self._started_monotonic))
        values.setdefault("elapsed_seconds", elapsed)
        self.update("training", values)

    def set_error(self, exc: BaseException | str) -> None:
        message = str(exc)
        with self._lock:
            self._state["status"] = "failed"
            self._state["phase"] = "failed"
            self._state["phase_label"] = "Failed"
            self._state["error"] = message
            self._state["finished_at"] = _utcnow()
            self._state["updated_at"] = _utcnow()
        self.log("ERROR", message)

    def finish(self, *, status: str = "completed", label: str = "Completed") -> None:
        with self._lock:
            self._state["status"] = status
            self._state["phase"] = status
            self._state["phase_label"] = label
            self._state["finished_at"] = _utcnow()
            self._state["updated_at"] = _utcnow()
            if status == "completed":
                self._state["training"]["progress_percent"] = 100.0
        self.log("INFO", label)

    def log(self, level: str, message: str, **fields: Any) -> None:
        entry = {"time": _utcnow(), "level": str(level).upper(), "message": str(message)}
        if fields:
            entry["fields"] = fields
        with self._lock:
            self._logs.append(entry)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            state = copy.deepcopy(self._state)
            state["logs"] = list(self._logs)
            state["log_file"] = str(self.log_path)
            return state

    def save_final_state(self) -> Path:
        path = self.log_dir / f"training-console-{self.session_id}-final.json"
        payload = self.snapshot()
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _sample_loop(self, interval_seconds: float) -> None:
        try:
            import psutil
        except ImportError:
            self.update("system", {"sampler_error": "psutil is not installed"})
            return

        nvml = None
        nvml_error: str | None = None
        try:
            import pynvml
            pynvml.nvmlInit()
            nvml = pynvml
        except Exception as exc:
            nvml_error = f"{type(exc).__name__}: {exc}"

        process = psutil.Process()
        psutil.cpu_percent(interval=None)

        try:
            while not self._stop_event.wait(interval_seconds):
                try:
                    vm = psutil.virtual_memory()
                    disk_root = self.output_dir.anchor or str(self.output_dir.resolve())
                    disk = shutil.disk_usage(disk_root)
                    proc_mem = process.memory_info()
                    tree_rss = int(proc_mem.rss)
                    for child in process.children(recursive=True):
                        try:
                            tree_rss += int(child.memory_info().rss)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    system: dict[str, Any] = {
                        "cpu_percent": round(psutil.cpu_percent(interval=None), 1),
                        "cpu_count": psutil.cpu_count(logical=True),
                        "ram_total_gb": _gb(vm.total),
                        "ram_used_gb": _gb(vm.used),
                        "ram_percent": round(vm.percent, 1),
                        "process_rss_gb": _gb(proc_mem.rss),
                        "process_tree_rss_gb": _gb(tree_rss),
                        "disk_total_gb": _gb(disk.total),
                        "disk_used_gb": _gb(disk.used),
                        "disk_free_gb": _gb(disk.free),
                        "disk_percent": round((disk.used / disk.total) * 100, 1) if disk.total else 0.0,
                        "gpus": [],
                        "gpu_monitor_error": nvml_error,
                    }

                    if nvml is not None:
                        gpus: list[dict[str, Any]] = []
                        try:
                            for index in range(nvml.nvmlDeviceGetCount()):
                                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                                name = nvml.nvmlDeviceGetName(handle)
                                if isinstance(name, bytes):
                                    name = name.decode("utf-8", errors="replace")
                                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                                mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                                try:
                                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                                except Exception:
                                    temp = None
                                try:
                                    power_w = round(nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1)
                                except Exception:
                                    power_w = None
                                try:
                                    power_limit_w = round(nvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0, 1)
                                except Exception:
                                    power_limit_w = None
                                gpus.append({
                                    "index": index,
                                    "name": str(name),
                                    "utilization_percent": float(util.gpu),
                                    "memory_total_gb": _gb(mem.total),
                                    "memory_used_gb": _gb(mem.used),
                                    "memory_free_gb": _gb(mem.free),
                                    "memory_percent": round((mem.used / mem.total) * 100, 1) if mem.total else 0.0,
                                    "temperature_c": temp,
                                    "power_w": power_w,
                                    "power_limit_w": power_limit_w,
                                })
                            system["gpus"] = gpus
                            system["gpu_monitor_error"] = None
                        except Exception as exc:
                            system["gpu_monitor_error"] = f"{type(exc).__name__}: {exc}"

                    self.update("system", system)
                    self.update_training(
                        elapsed_seconds=max(
                            0,
                            int(time.monotonic() - self._started_monotonic),
                        )
                    )
                except Exception as exc:
                    self.update("system", {"sampler_error": f"{type(exc).__name__}: {exc}"})
        finally:
            if nvml is not None:
                try:
                    nvml.nvmlShutdown()
                except Exception:
                    pass


class MonitorLoggingHandler(logging.Handler):
    def __init__(self, monitor: TrainingMonitor):
        super().__init__(level=logging.INFO)
        self.monitor = monitor

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.monitor.log(record.levelname, self.format(record))
        except Exception:
            pass


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_cls, monitor: TrainingMonitor):
        self.monitor = monitor
        super().__init__(server_address, handler_cls)


class _DashboardHandler(BaseHTTPRequestHandler):
    server: _DashboardHTTPServer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            self._send_json(self.server.monitor.snapshot())
            return
        if path in {"/", "/index.html"}:
            self._send_html(DASHBOARD_HTML)
            return
        if path == "/health":
            self._send_json({"ok": True})
            return
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TrainingDashboard:
    def __init__(self, monitor: TrainingMonitor, *, host: str = "127.0.0.1", preferred_port: int = 7860, open_browser: bool = True):
        self.monitor = monitor
        self.host = host
        self.preferred_port = preferred_port
        self.open_browser = open_browser
        self.server: _DashboardHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.url: str | None = None

    def start(self) -> str:
        try:
            server = _DashboardHTTPServer((self.host, self.preferred_port), _DashboardHandler, self.monitor)
        except OSError:
            server = _DashboardHTTPServer((self.host, 0), _DashboardHandler, self.monitor)

        self.server = server
        port = int(server.server_address[1])
        self.url = f"http://{self.host}:{port}/"
        self.monitor.set_dashboard(url=self.url, host=self.host, port=port)
        self.monitor.start_sampling()

        self.thread = threading.Thread(target=server.serve_forever, name="oracle-lite-dashboard", daemon=True)
        self.thread.start()
        self.monitor.log("INFO", f"Training Console: {self.url}")

        if self.open_browser:
            threading.Thread(target=lambda: webbrowser.open(self.url or ""), name="oracle-lite-browser-opener", daemon=True).start()

        return self.url

    def stop(self) -> None:
        self.monitor.stop_sampling()
        if self.server is not None:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass


DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Oracle-Lite Training Console</title>
<style>
:root{color-scheme:dark;--bg:#070a0f;--panel:#0d131c;--line:#1c2a38;--text:#e8f0f7;--muted:#7f93a6;--accent:#57d3ff;--ok:#5ee6a8;--warn:#ffd166;--bad:#ff6b78;--purple:#a58bff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% -10%,#122235 0,#070a0f 38%);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.wrap{max-width:1500px;margin:0 auto;padding:24px}header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:18px}.brand h1{font-size:22px;margin:0 0 5px;letter-spacing:.4px}.brand .sub{color:var(--muted)}.status{padding:7px 12px;border:1px solid var(--line);border-radius:999px;background:#0b1119;color:var(--accent);font-weight:700}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.card{background:linear-gradient(180deg,rgba(16,25,35,.94),rgba(10,16,24,.94));border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 12px 40px rgba(0,0,0,.18)}.span12{grid-column:span 12}.span8{grid-column:span 8}.span6{grid-column:span 6}.span4{grid-column:span 4}h2{font-size:12px;text-transform:uppercase;letter-spacing:1.3px;color:var(--muted);margin:0 0 12px}.phase{font-size:22px;font-weight:800;margin:0 0 8px}.progress{height:14px;background:#070b10;border-radius:999px;overflow:hidden;border:1px solid #172330}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--purple));transition:width .35s}.row{display:flex;justify-content:space-between;gap:12px;margin-top:9px;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.metric{background:#091019;border:1px solid #182534;border-radius:10px;padding:11px}.metric b{display:block;font-size:19px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.metric span{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}.gauge{margin:10px 0}.gauge .label{display:flex;justify-content:space-between;color:var(--muted);margin-bottom:5px}.track{height:7px;border-radius:99px;background:#060a0e;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--ok),var(--accent));width:0;transition:width .3s}.gpus{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}.gpu{background:#081018;border:1px solid #182535;border-radius:10px;padding:12px}.gpu strong{font-size:14px}.kv{display:grid;grid-template-columns:120px 1fr;gap:6px 12px}.kv div:nth-child(odd){color:var(--muted)}.logbox{height:360px;overflow:auto;background:#05080c;border:1px solid #172330;border-radius:10px;padding:10px;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}.logline{display:grid;grid-template-columns:185px 60px 1fr;gap:8px;padding:2px 0;border-bottom:1px solid rgba(255,255,255,.025)}.INFO{color:#9ab0c3}.WARNING{color:var(--warn)}.ERROR{color:var(--bad)}.DEBUG{color:#6d7e8d}.error{display:none;background:rgba(255,107,120,.08);border:1px solid rgba(255,107,120,.35);color:#ff9aa4;padding:10px;border-radius:10px;margin-top:10px}small{color:var(--muted)}@media(max-width:900px){.span8,.span6,.span4{grid-column:span 12}.metrics{grid-template-columns:repeat(2,1fr)}header{flex-direction:column}.logline{grid-template-columns:1fr}.logline span:first-child{display:none}}
</style>
</head>
<body><div class="wrap">
<header><div class="brand"><h1>ORACLE-LITE / TRAINING CONSOLE</h1><div class="sub">Local multimodal training telemetry · auto-refresh 1s</div></div><div id="status" class="status">STARTING</div></header>
<div class="grid">
<section class="card span8"><h2>Training Progress</h2><div id="phase" class="phase">Starting Oracle-Lite</div><div class="progress"><div id="bar" class="bar"></div></div><div class="row"><span id="step">Step 0 / —</span><span id="eta">ETA —</span></div><div id="error" class="error"></div></section>
<section class="card span4"><h2>Run</h2><div class="kv"><div>Snapshot</div><div id="snapshot">—</div><div>Run ID</div><div id="runid">—</div><div>Checkpoint</div><div id="checkpoint">—</div><div>Log file</div><div id="logfile">—</div></div></section>
<section class="card span12"><h2>Training Metrics</h2><div class="metrics"><div class="metric"><b id="loss">—</b><span>Loss</span></div><div class="metric"><b id="lr">—</b><span>Learning Rate</span></div><div class="metric"><b id="epoch">—</b><span>Epoch</span></div><div class="metric"><b id="elapsed">—</b><span>Elapsed</span></div></div></section>
<section class="card span6"><h2>GPU / VRAM</h2><div id="gpus" class="gpus"><small>Waiting for NVML…</small></div></section>
<section class="card span6"><h2>CPU / RAM / Disk</h2><div class="gauge"><div class="label"><span>CPU</span><span id="cpuv">—</span></div><div class="track"><div id="cpub" class="fill"></div></div></div><div class="gauge"><div class="label"><span>RAM</span><span id="ramv">—</span></div><div class="track"><div id="ramb" class="fill"></div></div></div><div class="gauge"><div class="label"><span>Disk</span><span id="diskv">—</span></div><div class="track"><div id="diskb" class="fill"></div></div></div><div class="row"><span>Oracle-Lite RSS</span><span id="rss">—</span></div><div class="row"><span>Process tree RSS</span><span id="treerss">—</span></div></section>
<section class="card span6"><h2>Corpus</h2><div class="metrics"><div class="metric"><b id="files">—</b><span>Files Seen</span></div><div class="metric"><b id="newfiles">—</b><span>New</span></div><div class="metric"><b id="changed">—</b><span>Changed</span></div><div class="metric"><b id="failed">—</b><span>Parse Failed</span></div></div><div class="row"><span>Current file</span><span id="currentfile">—</span></div><div class="row"><span>Hash progress</span><span id="hashprogress">—</span></div><div class="row"><span>Parser memory</span><span id="parsermem">—</span></div></section>
<section class="card span6"><h2>Dataset</h2><div class="metrics"><div class="metric"><b id="records">—</b><span>Records</span></div><div class="metric"><b id="textrecords">—</b><span>Text</span></div><div class="metric"><b id="visualrecords">—</b><span>Visual</span></div><div class="metric"><b id="chars">—</b><span>Characters</span></div></div></section>
<section class="card span12"><h2>Live Logs</h2><div id="logs" class="logbox"></div></section>
</div></div>
<script>
var el=function(id){return document.getElementById(id)};
var pct=function(v){return Math.max(0,Math.min(100,Number(v)||0))};
var fmt=function(n){return n===null||n===undefined?"—":Number(n).toLocaleString()};
var dur=function(s){if(s===null||s===undefined)return "—";s=Math.max(0,Math.floor(Number(s)||0));var h=Math.floor(s/3600),m=Math.floor((s%3600)/60),x=s%60;return h?(h+"h "+m+"m "+x+"s"):m?(m+"m "+x+"s"):(x+"s")};
var gauge=function(bar,val){bar.style.width=pct(val)+"%"};
var esc=function(s){var d=document.createElement("div");d.textContent=String(s);return d.innerHTML};
function renderGpu(gpus,err){var box=el("gpus");if(!gpus||!gpus.length){box.innerHTML="<small>"+esc(err||"No GPU telemetry yet")+"</small>";return}var html="";gpus.forEach(function(g){html+='<div class="gpu"><strong>GPU '+g.index+": "+esc(g.name)+'</strong><div class="gauge"><div class="label"><span>Compute</span><span>'+fmt(g.utilization_percent)+'%</span></div><div class="track"><div class="fill" style="width:'+pct(g.utilization_percent)+'%"></div></div></div><div class="gauge"><div class="label"><span>VRAM</span><span>'+fmt(g.memory_used_gb)+" / "+fmt(g.memory_total_gb)+' GB</span></div><div class="track"><div class="fill" style="width:'+pct(g.memory_percent)+'%"></div></div></div><div class="row"><span>'+(g.temperature_c==null?"—":g.temperature_c)+' °C</span><span>'+(g.power_w==null?"—":g.power_w)+" / "+(g.power_limit_w==null?"—":g.power_limit_w)+" W</span></div></div>"});box.innerHTML=html}
function renderLogs(logs){var box=el("logs"),near=box.scrollTop+box.clientHeight>=box.scrollHeight-40,html="";(logs||[]).forEach(function(x){html+='<div class="logline"><span>'+esc(x.time||"")+'</span><span class="'+esc(x.level||"INFO")+'">'+esc(x.level||"")+"</span><span>"+esc(x.message||"")+"</span></div>"});box.innerHTML=html;if(near)box.scrollTop=box.scrollHeight}
async function tick(){try{var r=await fetch("/api/state",{cache:"no-store"}),s=await r.json(),t=s.training||{},sys=s.system||{},c=s.corpus||{},d=s.dataset||{};el("status").textContent=(s.status||"unknown").toUpperCase();el("phase").textContent=s.phase_label||s.phase||"—";gauge(el("bar"),t.progress_percent);el("step").textContent="Step "+fmt(t.step)+" / "+(t.total_steps?fmt(t.total_steps):"—")+" · "+Number(t.progress_percent||0).toFixed(1)+"%";el("eta").textContent="ETA "+dur(t.eta_seconds);el("loss").textContent=t.loss==null?"—":Number(t.loss).toFixed(5);el("lr").textContent=t.learning_rate==null?"—":Number(t.learning_rate).toExponential(3);el("epoch").textContent=t.epoch==null?"—":Number(t.epoch).toFixed(3);el("elapsed").textContent=dur(t.elapsed_seconds);el("snapshot").textContent=d.snapshot_id||s.snapshot_id||"—";el("runid").textContent=t.run_id||s.run_id||"—";el("checkpoint").textContent=t.checkpoint||"—";el("logfile").textContent=s.log_file||"—";el("cpuv").textContent=fmt(sys.cpu_percent)+"%";gauge(el("cpub"),sys.cpu_percent);el("ramv").textContent=fmt(sys.ram_used_gb)+" / "+fmt(sys.ram_total_gb)+" GB · "+fmt(sys.ram_percent)+"%";gauge(el("ramb"),sys.ram_percent);el("diskv").textContent=fmt(sys.disk_used_gb)+" / "+fmt(sys.disk_total_gb)+" GB · "+fmt(sys.disk_percent)+"%";gauge(el("diskb"),sys.disk_percent);el("rss").textContent=fmt(sys.process_rss_gb)+" GB";el("treerss").textContent=fmt(sys.process_tree_rss_gb)+" GB";renderGpu(sys.gpus,sys.gpu_monitor_error);el("files").textContent=fmt(c.files_seen);el("newfiles").textContent=fmt(c.new);el("changed").textContent=fmt(c.changed);el("failed").textContent=fmt(c.failed);el("currentfile").textContent=c.current_parse_file||c.current_file||"—";var hp=(c.current_file_size&&c.hashing)?((Number(c.current_file_hashed_bytes||0)/Number(c.current_file_size))*100).toFixed(1)+"% · "+fmt(c.current_file_hashed_bytes)+" / "+fmt(c.current_file_size)+" bytes":(c.hashing?"Hashing…":"—");el("hashprogress").textContent=hp;var pm=c.parse_state?c.parse_state+" · worker "+fmt(c.worker_rss_gb)+" / "+fmt(c.max_worker_rss_gb)+" GB · avail "+fmt(c.memory_available_gb)+" GB":"—";el("parsermem").textContent=pm;el("records").textContent=fmt(d.records);el("textrecords").textContent=fmt(d.text_records);el("visualrecords").textContent=fmt(d.visual_records);el("chars").textContent=fmt(d.characters);var er=el("error");if(s.error){er.style.display="block";er.textContent=s.error}else{er.style.display="none"}renderLogs(s.logs)}catch(e){el("status").textContent="OFFLINE"}}
tick();setInterval(tick,1000);
</script>
</body></html>
"""
