#!/usr/bin/env python3
"""Render a standalone HTML report from a PyTorch CUDA memory snapshot.

Snapshots are pickle files and must only be loaded from trusted sources.
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LIBRARY_PARTS = (
    "/site-packages/torch/",
    "/site-packages/einops/",
    "/lib/python",
)


@dataclass
class Lifetime:
    address: int
    size: int
    start_us: int
    end_us: int
    frames: list[dict[str, Any]]
    before_capture: bool = False
    after_capture: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Trusted PyTorch snapshot pickle")
    parser.add_argument("--output", "-o", type=Path, default=Path("memory-report.html"))
    parser.add_argument("--device", type=int, default=0, help="Device trace index")
    parser.add_argument(
        "--max-events",
        type=int,
        default=2000,
        help="Largest lifetimes to show; 0 shows all (default: 2000)",
    )
    parser.add_argument(
        "--min-size-mib",
        type=float,
        default=0.0,
        help="Minimum lifetime size shown in the table",
    )
    return parser.parse_args()


def active_bytes_at_dump(snapshot: dict[str, Any], device: int) -> int:
    total = 0
    for segment in snapshot.get("segments", []):
        if segment.get("device", 0) != device:
            continue
        for block in segment.get("blocks", []):
            if block.get("state") == "active_allocated":
                total += int(block.get("requested_size", block.get("size", 0)))
    return total


def active_blocks_at_dump(
    snapshot: dict[str, Any], device: int
) -> list[dict[str, Any]]:
    blocks = []
    for segment in snapshot.get("segments", []):
        if segment.get("device", 0) != device:
            continue
        blocks.extend(
            block
            for block in segment.get("blocks", [])
            if block.get("state") == "active_allocated"
        )
    return blocks


def reconstruct(
    snapshot: dict[str, Any], device: int
) -> tuple[list[Lifetime], list[tuple[int, int]], int, int, int]:
    traces = snapshot.get("device_traces", [])
    if device < 0 or device >= len(traces):
        raise ValueError(
            f"device {device} is unavailable; snapshot has {len(traces)} trace(s)"
        )
    trace = traces[device]
    timed = [event for event in trace if "time_us" in event]
    if not timed:
        raise ValueError("snapshot contains no timestamped events")
    first_us = int(timed[0]["time_us"])
    last_us = int(timed[-1]["time_us"])
    final_active = active_bytes_at_dump(snapshot, device)
    net_change = sum(
        (
            int(event.get("size", 0))
            if event.get("action") == "alloc"
            else (
                -int(event.get("size", 0))
                if event.get("action") == "free_requested"
                else 0
            )
        )
        for event in trace
    )
    initial_active = final_active - net_change
    active_bytes = initial_active
    peak_bytes = active_bytes
    points = [(0, active_bytes)]
    live: dict[int, Lifetime] = {}
    lifetimes: list[Lifetime] = []

    for event in trace:
        action = event.get("action")
        if action not in {"alloc", "free_requested"}:
            continue
        timestamp = int(event.get("time_us", first_us))
        address = int(event.get("addr", 0))
        size = int(event.get("size", 0))
        if action == "alloc":
            # Reuse should follow a free. Preserve malformed overlaps instead of hiding them.
            previous = live.pop(address, None)
            if previous is not None:
                previous.end_us = timestamp
                lifetimes.append(previous)
            live[address] = Lifetime(
                address=address,
                size=size,
                start_us=timestamp,
                end_us=last_us,
                frames=list(event.get("frames", [])),
                after_capture=True,
            )
            active_bytes += size
        else:
            lifetime = live.pop(address, None)
            if lifetime is None:
                lifetime = Lifetime(
                    address=address,
                    size=size,
                    start_us=first_us,
                    end_us=timestamp,
                    frames=list(event.get("frames", [])),
                    before_capture=True,
                )
            else:
                lifetime.end_us = timestamp
                lifetime.after_capture = False
            lifetimes.append(lifetime)
            active_bytes -= size
        peak_bytes = max(peak_bytes, active_bytes)
        points.append((timestamp - first_us, active_bytes))

    lifetimes.extend(live.values())
    # Snapshot segments describe allocations that are live at dump time. Add
    # any address with no in-trace allocation so persistent parameters,
    # gradients, and optimizer state are visible as capture-spanning rows.
    live_addresses = set(live)
    for block in active_blocks_at_dump(snapshot, device):
        address = int(block.get("address", 0))
        if address in live_addresses:
            continue
        lifetimes.append(
            Lifetime(
                address=address,
                size=int(block.get("requested_size", block.get("size", 0))),
                start_us=first_us,
                end_us=last_us,
                frames=list(block.get("frames", [])),
                before_capture=True,
                after_capture=True,
            )
        )
    return lifetimes, points, initial_active, final_active, peak_bytes


def downsample(
    points: list[tuple[int, int]], limit: int = 4000
) -> list[tuple[int, int]]:
    if len(points) <= limit:
        return points
    bucket_size = max(1, len(points) // (limit // 2))
    sampled = [points[0]]
    for start in range(1, len(points) - 1, bucket_size):
        bucket = points[start : start + bucket_size]
        if not bucket:
            continue
        low = min(bucket, key=lambda point: point[1])
        high = max(bucket, key=lambda point: point[1])
        sampled.extend(sorted({low, high}, key=lambda point: point[0]))
    sampled.append(points[-1])
    return sampled


def frame_label(frame: dict[str, Any]) -> str:
    filename = str(frame.get("filename") or "<native>")
    name = str(frame.get("name") or "<unknown>")
    line = int(frame.get("line") or 0)
    return f"{filename}:{line} in {name}" if line else f"{filename} in {name}"


def best_frame(lifetime: Lifetime) -> str:
    if lifetime.before_capture:
        return "Before capture (allocation origin unavailable)"
    python_frames = [
        frame
        for frame in lifetime.frames
        if str(frame.get("filename", "")).endswith(".py")
    ]
    user_frames = [
        frame
        for frame in python_frames
        if not any(part in str(frame.get("filename", "")) for part in LIBRARY_PARTS)
    ]
    if user_frames:
        return frame_label(user_frames[0])
    if python_frames:
        return frame_label(python_frames[0])
    return "Native/autograd allocation (no Python frame)"


def active_at(points: list[tuple[int, int]], timestamp_us: int) -> int:
    value = points[0][1]
    for point_time, point_value in points:
        if point_time > timestamp_us:
            break
        value = point_value
    return value


def annotations(
    snapshot: dict[str, Any],
    first_us: int,
    device: int,
    points: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    starts: dict[str, list[int]] = {}
    ranges: list[dict[str, Any]] = []
    for item in sorted(
        snapshot.get("external_annotations", []),
        key=lambda value: value.get("time_us", 0),
    ):
        if item.get("device", device) != device:
            continue
        name = str(item.get("name", "annotation"))
        timestamp = int(item.get("time_us", first_us))
        if item.get("stage") == "START":
            starts.setdefault(name, []).append(timestamp)
        elif item.get("stage") == "END" and starts.get(name):
            start = starts[name].pop()
            relative_start = start - first_us
            relative_end = timestamp - first_us
            in_range = [
                point_value
                for point_time, point_value in points
                if relative_start <= point_time <= relative_end
            ]
            allocated = sum(
                int(event.get("size", 0))
                for event in snapshot["device_traces"][device]
                if event.get("action") == "alloc"
                and start <= int(event.get("time_us", first_us)) <= timestamp
            )
            ranges.append(
                {
                    "name": name,
                    "start_us": relative_start,
                    "end_us": relative_end,
                    "allocated": allocated,
                    "start_active": active_at(points, relative_start),
                    "end_active": active_at(points, relative_end),
                    "peak_active": (
                        max(in_range) if in_range else active_at(points, relative_start)
                    ),
                }
            )
    return ranges


def fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{number:.2f} TiB"


def build_insights(
    lifetimes: list[Lifetime],
    annotation_ranges: list[dict[str, Any]],
    initial_active: int,
    final_active: int,
    peak_bytes: int,
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    peak_delta = max(0, peak_bytes - initial_active)
    if initial_active:
        peak_percent = peak_delta / initial_active * 100
        insights.append(
            {
                "kind": "Peak growth",
                "finding": (
                    f"Active memory rises by {fmt_bytes(peak_delta)} "
                    f"({peak_percent:.1f}%) above the capture baseline."
                ),
                "action": (
                    "Prioritize allocations that remain live at the peak; reducing "
                    "short-lived allocations elsewhere may not lower the high-water mark."
                ),
            }
        )

    end_growth = final_active - initial_active
    if end_growth > max(64 * 1024**2, initial_active * 0.02):
        insights.append(
            {
                "kind": "End-of-capture retention",
                "finding": (
                    f"The capture ends {fmt_bytes(end_growth)} above its initial active memory."
                ),
                "action": (
                    "Check whether outputs, losses, gradients, or state are intentionally retained. "
                    "Repeat with more iterations before calling this a leak."
                ),
            }
        )

    persistent = [lifetime for lifetime in lifetimes if lifetime.after_capture]
    persistent_bytes = sum(lifetime.size for lifetime in persistent)
    if persistent:
        insights.append(
            {
                "kind": "Live at capture end",
                "finding": (
                    f"{len(persistent):,} recorded lifetimes totaling {fmt_bytes(persistent_bytes)} "
                    "are still live at the dump."
                ),
                "action": (
                    "Inspect the largest end-live rows. Persistent parameters and optimizer state "
                    "are expected; unexpected activations or outputs are optimization candidates."
                ),
            }
        )

    pre_capture = [lifetime for lifetime in lifetimes if lifetime.before_capture]
    pre_bytes = sum(lifetime.size for lifetime in pre_capture)
    if pre_capture:
        insights.append(
            {
                "kind": "Attribution blind spot",
                "finding": (
                    f"{len(pre_capture):,} lifetimes totaling {fmt_bytes(pre_bytes)} began before "
                    "memory-history recording, so their origins are unavailable."
                ),
                "action": (
                    "Start memory history before model and optimizer initialization when those "
                    "origins matter; keep the current steady-state capture for iteration analysis."
                ),
            }
        )

    attributed = [
        lifetime
        for lifetime in lifetimes
        if not lifetime.before_capture and lifetime.frames
    ]
    if lifetimes and len(attributed) / len(lifetimes) < 0.5:
        coverage = len(attributed) / len(lifetimes) * 100
        insights.append(
            {
                "kind": "Low stack coverage",
                "finding": f"Only {coverage:.1f}% of lifetimes have an in-capture stack.",
                "action": (
                    "Treat code-path rankings as incomplete. Native autograd and pre-capture "
                    "allocations need an earlier or complementary profiler capture."
                ),
            }
        )

    grouped: dict[str, dict[str, int]] = {}
    for lifetime in lifetimes:
        label = best_frame(lifetime)
        if label.startswith(("Before capture", "Native/autograd")):
            continue
        item = grouped.setdefault(label, {"bytes": 0, "count": 0, "largest": 0})
        item["bytes"] += lifetime.size
        item["count"] += 1
        item["largest"] = max(item["largest"], lifetime.size)
    if grouped:
        label, item = max(grouped.items(), key=lambda pair: pair[1]["bytes"])
        lower = label.lower()
        if "adamw" in lower or "optimizer" in lower:
            suggestion = (
                "Inspect optimizer temporaries and state precision; fused or foreach updates and "
                "in-place-safe formulations may reduce churn, but verify numerical equivalence."
            )
        elif "attention" in lower or "scaled_dot_product" in lower:
            suggestion = (
                "Inspect attention score/materialization shapes; fused scaled-dot-product attention, "
                "shorter sequences, or checkpointing may reduce the peak if compatible."
            )
        elif "layers.py" in lower or "transform" in lower or "forward" in lower:
            suggestion = (
                "Inspect the tensor shape and lifetime at this line. Activation checkpointing, "
                "smaller batches/sequences, or avoiding simultaneous intermediates may help."
            )
        else:
            suggestion = (
                "Inspect tensor shape and lifetime at this source line, then confirm whether these "
                "allocations overlap the peak before optimizing it."
            )
        insights.append(
            {
                "kind": "Dominant attributed traffic",
                "finding": (
                    f"{label} accounts for {fmt_bytes(item['bytes'])} across "
                    f"{item['count']:,} allocations; its largest is {fmt_bytes(item['largest'])}."
                ),
                "action": suggestion + " Cumulative traffic is not retained memory.",
            }
        )

    if annotation_ranges:
        busiest = max(annotation_ranges, key=lambda item: item["allocated"])
        insights.append(
            {
                "kind": "Highest annotated churn",
                "finding": (
                    f"{busiest['name']} allocates {fmt_bytes(busiest['allocated'])} cumulatively "
                    f"and reaches {fmt_bytes(busiest['peak_active'])} active memory."
                ),
                "action": (
                    "Use its stack-ranked rows to find repeated temporaries. Optimize this range "
                    "for allocation overhead only after checking whether it also owns the peak."
                ),
            }
        )
    else:
        insights.append(
            {
                "kind": "Missing stage ranges",
                "finding": "No paired external annotations are available in this snapshot.",
                "action": (
                    "Add profiler-recorded ranges around forward, backward, and optimizer work to "
                    "compare stage-level allocation traffic and peaks."
                ),
            }
        )

    return insights[:6]


def render(
    snapshot_path: Path,
    lifetimes: list[Lifetime],
    points: list[tuple[int, int]],
    annotation_ranges: list[dict[str, Any]],
    initial_active: int,
    final_active: int,
    peak_bytes: int,
    max_events: int,
    min_size: int,
    first_us: int,
) -> str:
    eligible = [lifetime for lifetime in lifetimes if lifetime.size >= min_size]
    eligible.sort(key=lambda lifetime: (-lifetime.size, lifetime.start_us))
    displayed = eligible if max_events == 0 else eligible[:max_events]
    rows = []
    for index, lifetime in enumerate(displayed, 1):
        start = 0 if lifetime.before_capture else lifetime.start_us - first_us
        end = lifetime.end_us - first_us
        stack = (
            "\n".join(frame_label(frame) for frame in lifetime.frames)
            or "No stack captured"
        )
        rows.append(
            "<tr class='event-row' "
            f"data-size='{lifetime.size}' data-start='{start}' data-path='{html.escape(best_frame(lifetime).lower())}'>"
            f"<td>{index}</td><td><code>{html.escape(best_frame(lifetime))}</code>"
            f"<details><summary>Full stack</summary><pre>{html.escape(stack)}</pre></details></td>"
            f"<td data-sort='{start}'>{'Before capture' if lifetime.before_capture else f'{start / 1000:.3f} ms'}</td>"
            f"<td data-sort='{end}'>{'End of capture' if lifetime.after_capture else f'{end / 1000:.3f} ms'}</td>"
            f"<td data-sort='{max(0, end-start)}'>{max(0, end-start) / 1000:.3f} ms</td>"
            f"<td data-sort='{lifetime.size}'>{fmt_bytes(lifetime.size)}</td>"
            f"<td><code>0x{lifetime.address:x}</code></td></tr>"
        )
    annotation_rows = []
    for index, item in enumerate(annotation_ranges, 1):
        annotation_rows.append(
            f"<tr><td>{index}</td><td><code>{html.escape(item['name'])}</code></td>"
            f"<td>{item['start_us'] / 1000:.3f} ms</td>"
            f"<td>{item['end_us'] / 1000:.3f} ms</td>"
            f"<td>{(item['end_us'] - item['start_us']) / 1000:.3f} ms</td>"
            f"<td>{fmt_bytes(item['allocated'])}</td>"
            f"<td>{fmt_bytes(item['start_active'])}</td>"
            f"<td>{fmt_bytes(item['end_active'])}</td>"
            f"<td>{fmt_bytes(item['peak_active'])}</td></tr>"
        )
    insights = build_insights(
        lifetimes, annotation_ranges, initial_active, final_active, peak_bytes
    )
    insight_cards = "".join(
        f"<article class='insight'><h3>{html.escape(item['kind'])}</h3>"
        f"<p>{html.escape(item['finding'])}</p>"
        f"<p class='action'><strong>Investigate:</strong> {html.escape(item['action'])}</p></article>"
        for item in insights
    )
    graph_data = json.dumps(downsample(points), separators=(",", ":"))
    annotation_data = json.dumps(annotation_ranges, separators=(",", ":"))
    title = html.escape(snapshot_path.name)
    subtitle = html.escape(str(snapshot_path.resolve()))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyTorch memory report - {title}</title>
<style>
:root{{--bg:#0b1020;--panel:#131a2b;--muted:#9aa7bd;--text:#edf2f7;--line:#53d8fb;--accent:#a78bfa;--grid:#29334b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 6px;font-size:26px}} .subtitle{{color:var(--muted);word-break:break-all}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0}}
.card,.panel{{background:var(--panel);border:1px solid #26314a;border-radius:12px;padding:16px}} .card b{{display:block;font-size:22px;margin-top:6px}}
.insights{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-bottom:22px}} .insight{{background:#131a2b;border:1px solid #34405b;border-left:4px solid var(--accent);border-radius:10px;padding:14px}} .insight h3{{margin:0 0 8px;font-size:15px}} .insight p{{margin:6px 0;line-height:1.45}} .insight .action{{color:#bec9db}}
.chart-wrap{{height:390px;position:relative}} canvas{{width:100%;height:100%}} #tooltip{{position:absolute;display:none;pointer-events:none;background:#050814;border:1px solid #53617c;border-radius:7px;padding:8px;white-space:nowrap}}
.controls{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}} input{{background:#0d1425;color:var(--text);border:1px solid #34405b;border-radius:7px;padding:9px 11px}}
.table-wrap{{overflow:auto;max-height:720px}} table{{border-collapse:collapse;width:100%;min-width:1050px}} th{{position:sticky;top:0;background:#1c2539;cursor:pointer;text-align:left}} th,td{{padding:9px;border-bottom:1px solid #273149;vertical-align:top}}
code,pre{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}} pre{{white-space:pre-wrap;color:#c6d1e4}} details{{margin-top:6px;color:var(--muted)}} .note{{color:var(--muted);line-height:1.5}}
</style></head><body><main>
<h1>PyTorch active-memory report</h1><div class="subtitle">{subtitle}</div>
<section class="cards"><div class="card">Initial active<b>{fmt_bytes(initial_active)}</b></div><div class="card">Peak active<b>{fmt_bytes(peak_bytes)}</b></div><div class="card">Final active<b>{fmt_bytes(final_active)}</b></div><div class="card">Allocation lifetimes<b>{len(lifetimes):,}</b></div></section>
<h2>Insights and possible improvements</h2><p class="note">Heuristic leads derived from this snapshot. Confirm against source and a controlled comparison before changing code.</p><section class="insights">{insight_cards}</section>
<section class="panel"><h2>Active memory over time</h2><p class="note">Reconstructed from alloc and free_requested events. Hover for time and memory. Colored spans are external annotations when available.</p><div class="chart-wrap"><canvas id="chart"></canvas><div id="tooltip"></div></div></section>
<h2>Recorded annotation ranges</h2><p class="note">Ranges emitted into the PyTorch snapshot, when present. Allocated is cumulative allocation traffic during the range; peak active is the high-water mark, so these values answer different questions.</p>
<div class="panel table-wrap"><table><thead><tr><th>#</th><th>Event</th><th>Start</th><th>End</th><th>Duration</th><th>Allocated during event</th><th>Active at start</th><th>Active at end</th><th>Peak active</th></tr></thead><tbody>{''.join(annotation_rows) if annotation_rows else '<tr><td colspan="9">No paired external annotations were recorded.</td></tr>'}</tbody></table></div>
<h2>Largest allocation lifetimes</h2><p class="note">Showing {len(displayed):,} of {len(eligible):,} allocations matching the size filter, ordered largest first. Click headers to sort; expand a row for its captured stack.</p>
<div class="controls"><input id="search" placeholder="Filter code path"><input id="sizeFilter" type="number" min="0" step="1" placeholder="Minimum MiB"></div>
<div class="panel table-wrap"><table id="events"><thead><tr><th>#</th><th>Event / code path</th><th>Start</th><th>End</th><th>Duration</th><th>Allocated</th><th>Address</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">This report shows PyTorch allocator active memory, not total process VRAM or CUDA reserved memory. Pickle input must be trusted.</p>
<script>
const points={graph_data}, annotations={annotation_data};
const canvas=document.getElementById('chart'), ctx=canvas.getContext('2d'), tip=document.getElementById('tooltip');
let mapped=[]; function draw(){{const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);const w=r.width,h=r.height,p={{l:68,r:18,t:20,b:42}};ctx.clearRect(0,0,w,h);if(!points.length)return;const maxX=Math.max(1,...points.map(x=>x[0])),maxY=Math.max(1,...points.map(x=>x[1]));const x=v=>p.l+v/maxX*(w-p.l-p.r),y=v=>h-p.b-v/maxY*(h-p.t-p.b);annotations.forEach((a,i)=>{{ctx.fillStyle=i%2?'#a78bfa18':'#53d8fb18';ctx.fillRect(x(a.start_us),p.t,Math.max(1,x(a.end_us)-x(a.start_us)),h-p.t-p.b);}});ctx.strokeStyle='#29334b';ctx.fillStyle='#9aa7bd';ctx.font='12px system-ui';for(let i=0;i<=4;i++){{const yy=p.t+i*(h-p.t-p.b)/4;ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(w-p.r,yy);ctx.stroke();ctx.fillText(((maxY*(4-i)/4)/1073741824).toFixed(1)+' GiB',5,yy+4)}}ctx.beginPath();ctx.strokeStyle='#53d8fb';ctx.lineWidth=2;mapped=points.map(q=>[x(q[0]),y(q[1]),q]);mapped.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.stroke();ctx.fillStyle='#9aa7bd';ctx.fillText('0 ms',p.l,h-12);ctx.fillText((maxX/1000).toFixed(1)+' ms',w-p.r-55,h-12)}}
canvas.addEventListener('mousemove',e=>{{if(!mapped.length)return;const rect=canvas.getBoundingClientRect(),mx=e.clientX-rect.left;let q=mapped.reduce((a,b)=>Math.abs(b[0]-mx)<Math.abs(a[0]-mx)?b:a);tip.style.display='block';tip.style.left=Math.min(rect.width-170,q[0]+12)+'px';tip.style.top=Math.max(4,q[1]-40)+'px';tip.textContent=(q[2][0]/1000).toFixed(3)+' ms · '+(q[2][1]/1073741824).toFixed(3)+' GiB'}});canvas.addEventListener('mouseleave',()=>tip.style.display='none');addEventListener('resize',draw);draw();
function filter(){{const text=document.getElementById('search').value.toLowerCase(),mib=(+document.getElementById('sizeFilter').value||0)*1048576;document.querySelectorAll('.event-row').forEach(r=>r.hidden=!r.dataset.path.includes(text)||+r.dataset.size<mib)}}document.getElementById('search').addEventListener('input',filter);document.getElementById('sizeFilter').addEventListener('input',filter);
document.querySelectorAll('th').forEach((th,col)=>th.addEventListener('click',()=>{{const body=th.closest('table').tBodies[0],rows=[...body.rows],asc=th.dataset.asc!=='1';document.querySelectorAll('th').forEach(x=>delete x.dataset.asc);th.dataset.asc=asc?'1':'0';rows.sort((a,b)=>{{let x=a.cells[col].dataset.sort??a.cells[col].innerText,y=b.cells[col].dataset.sort??b.cells[col].innerText;let nx=+x,ny=+y,v=Number.isNaN(nx)||Number.isNaN(ny)?x.localeCompare(y):nx-ny;return asc?v:-v}});rows.forEach(r=>body.appendChild(r))}}));
</script></main></body></html>"""


def main() -> None:
    args = parse_args()
    if not args.snapshot.is_file():
        raise SystemExit(f"snapshot not found: {args.snapshot}")
    if args.max_events < 0 or args.min_size_mib < 0:
        raise SystemExit("--max-events and --min-size-mib must be non-negative")
    with args.snapshot.open("rb") as handle:
        snapshot = pickle.load(handle)  # noqa: S301 - trusted snapshots only
    lifetimes, points, initial, final, peak = reconstruct(snapshot, args.device)
    first_us = int(snapshot["device_traces"][args.device][0]["time_us"])
    report = render(
        args.snapshot,
        lifetimes,
        points,
        annotations(snapshot, first_us, args.device, points),
        initial,
        final,
        peak,
        args.max_events,
        int(args.min_size_mib * 1024 * 1024),
        first_us,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output.resolve()}")
    print(f"Peak active memory: {fmt_bytes(peak)}; lifetimes: {len(lifetimes):,}")


if __name__ == "__main__":
    main()
