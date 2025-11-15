import re
from collections import defaultdict, deque
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==================== CONFIGURATION ====================
CSV_PATH        = "cleaned_data_coordinates_216367.csv"
TARGET_FPS      = 30
MAX_FRAMES      = 900
POINT_SIZE      = 14
FINGERS         = ["T", "I", "M", "R", "L"]      
WINDOW_SEC      = 10.0
PALM_CANDIDATES = ["Win", "Wout", "Ain", "Aout", "Cin", "Cout"]
VELOCITY_UNIT   = "m/s"                          

COLORS = {
    "T":    "#1f77b4",
    "I":    "#ff7f0e",
    "M":    "#2ca02c",
    "R":    "#d62728",
    "L":    "#9467bd",
    "PALM": "#444444",
    "GREY": "#bdbdbd",
    "LABEL":"#111111",
}

# ==================== DATA INGESTION & PREPROCESSING ====================
df = pd.read_csv(CSV_PATH)

def normalize_time(series):
    t = series.to_numpy(dtype=float)
    t -= np.nanmin(t)
    diffs = np.diff(t[np.isfinite(t)])
    dt_med = np.nanmedian(diffs) if diffs.size else 1.0
    return t / 1000.0 if dt_med > 1.5 else t

has_time = 'time' in df.columns and np.issubdtype(df['time'].dtype, np.number)
t_sec = normalize_time(df['time']) if has_time else None
def frame_time(i): return (t_sec[i] if t_sec is not None else i / TARGET_FPS)

# Parse joint columns
pat = re.compile(r'^Hands_(?P<hand>[LR])_(?P<joint>[A-Za-z0-9]+)_(?P<axis>[xyz])$')
joint_axes = defaultdict(dict)
hands = {'L': set(), 'R': set()}
for col in df.columns:
    m = pat.match(col)
    if m:
        h = m.group('hand'); j = m.group('joint'); a = m.group('axis')
        joint_axes[(h, j)][a] = col
        hands[h].add(j)
hands = {h: sorted(v) for h, v in hands.items()}

def finger_chain(joints, f):
    return [f+str(k) for k in [1,2,3,4] if (f+str(k)) in joints]

def pick_wrist(joints):
    for c in PALM_CANDIDATES:
        if c in joints: return c
    return joints[0] if joints else None

def build_hand_struct(joints):
    present = set(joints)
    order = []
    for p in PALM_CANDIDATES:
        if p in present: order.append(p)
    chains = {f: finger_chain(joints, f) for f in FINGERS}
    for f in FINGERS: order += chains[f]
    rest = [j for j in joints if j not in set(order)]
    order += rest

    edges_meta = []
    palm_edges = [
        ("Win","Ain"), ("Ain","Cin"), ("Cin","Wout"),
        ("Win","Aout"), ("Aout","Cout"), ("Cout","Wout"),
        ("Ain","Aout"), ("Cin","Cout")
    ]
    for a,b in palm_edges:
        if a in present and b in present: edges_meta.append((a,b,'palm',None))
    for f in FINGERS:
        segs = chains[f]
        for a,b in zip(segs, segs[1:]):
            edges_meta.append((a,b,'finger',f))
    for f in FINGERS:
        segs = chains[f]
        if segs:
            base = segs[0]
            for cand in PALM_CANDIDATES:
                if cand in present:
                    edges_meta.append((cand, base, 'finger', f)); break
            else:
                w = pick_wrist(joints)
                if w: edges_meta.append((w, base, 'finger', f))
    w = pick_wrist(joints)
    if w:
        for p in ["Ain","Aout","Cin","Cout","Wout"]:
            if p in present and p != w:
                edges_meta.append((w,p,'palm',None))

    # Deduplicate edges (keep kind/finger)
    seen = set(); uniq = []
    for a,b,k,f in edges_meta:
        key = (tuple(sorted((a,b))), k, f)
        if key not in seen:
            seen.add(key); uniq.append((a,b,k,f))

    idx = {j:i for i,j in enumerate(order)}
    groups = {"PALM": [], "T": [], "I": [], "M": [], "R": [], "L": []}
    for j in order:
        if j in PALM_CANDIDATES: groups["PALM"].append(idx[j])
    for f in FINGERS:
        for j in chains[f]:
            groups[f].append(idx[j])

    # Deepest available tip per finger
    tips = {}
    for f in FINGERS:
        for k in (4,3,2,1):
            name = f+str(k)
            if name in present:
                tips[f] = name; break
        else:
            tips[f] = None

    return uniq, order, groups, tips

edges_L, order_L, groups_L, tips_L = build_hand_struct(hands.get('L', []))
edges_R, order_R, groups_R, tips_R = build_hand_struct(hands.get('R', []))

def extract_points(row, hand, ordered):
    P = np.full((len(ordered), 3), np.nan, dtype=float)
    for i, j in enumerate(ordered):
        cols = joint_axes.get((hand, j), {})
        try:
            P[i, 0] = row[cols['x']]
            P[i, 1] = row[cols['y']]
            P[i, 2] = row[cols['z']]
        except Exception:
            pass
    return P

stride = int(np.ceil(len(df) / MAX_FRAMES)) if len(df)>MAX_FRAMES else 1
frame_indices = list(range(0, len(df), stride))

# Determine 3D bounds
mins = np.array([np.inf]*3); maxs = np.array([-np.inf]*3)
for i in np.linspace(0, len(df)-1, num=min(len(df), 1200), dtype=int):
    row = df.iloc[i]
    for hand, order in [('L', order_L), ('R', order_R)]:
        if order:
            P = extract_points(row, hand, order)
            if np.isfinite(P).any():
                mins = np.minimum(mins, np.nanmin(P, axis=0))
                maxs = np.maximum(maxs, np.nanmax(P, axis=0))
if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
    mins = np.array([-1,-1,-1]); maxs = np.array([1,1,1])

# ==================== TKINTER UI SETUP ====================
root = tk.Tk()
root.title("Typing Hands — 3D Kinematics and Finger Velocities (Real-time)")

top = ttk.Frame(root)
top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

ttk.Label(top, text="Analysis Hand:").pack(side=tk.LEFT)
hand_var = tk.StringVar(value="Right")
combo = ttk.Combobox(top, textvariable=hand_var, values=["Left","Right"], state="readonly", width=8)
combo.pack(side=tk.LEFT, padx=8)

# Swap L/R datasets toggle
swap_var = tk.BooleanVar(value=False)
swap_cb = ttk.Checkbutton(top, text="Swap L/R datasets", variable=swap_var)
swap_cb.pack(side=tk.LEFT, padx=12)

# ==================== MATPLOTLIB FIGURE SETUP ====================
fig = plt.Figure(figsize=(11, 7), dpi=100)
gs = GridSpec(5, 2, figure=fig, width_ratios=[2.4, 1.6], height_ratios=[1,1,1,1,1], wspace=0.35, hspace=0.55)

ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
def set_equal_aspect(ax, mins, maxs):
    rng = maxs - mins
    L = np.nanmax(rng) if np.isfinite(rng).all() else 1.0
    if L == 0: L = 1.0
    mid = (maxs + mins)/2.0
    ax.set_xlim(mid[0]-L/2, mid[0]+L/2)
    ax.set_ylim(mid[1]-L/2, mid[1]+L/2)
    ax.set_zlim(mid[2]-L/2, mid[2]+L/2)
set_equal_aspect(ax3d, mins, maxs)
title3d = ax3d.set_title("Typing hands — frame 0")

status_text = fig.text(0.04, 0.95, "Selected hand: Right (dataset: Hands_R)", fontsize=11, color="#000")

# Dataset labels
label_L = ax3d.text(0,0,0,"L", color=COLORS["LABEL"], fontsize=14, weight='bold')
label_R = ax3d.text(0,0,0,"R", color=COLORS["LABEL"], fontsize=14, weight='bold')

# Velocity plots
axes_vel = {}
lines_vel = {}
for i, f in enumerate(FINGERS):
    ax = fig.add_subplot(gs[i, 1])
    ax.set_title(f"Speed {f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(f"|v| ({VELOCITY_UNIT})")
    (lv,) = ax.plot([], [], color=COLORS[f])
    axes_vel[f] = ax
    lines_vel[f] = lv

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# ==================== GRAPHIC OBJECT INITIALIZATION ====================
def make_lines(edges_meta):
    lines = []
    for (a,b,kind,finger) in edges_meta:
        (ln,) = ax3d.plot([], [], [], linewidth=2)
        lines.append({"line": ln, "a": a, "b": b, "kind": kind, "finger": finger})
    return lines

lines_L = make_lines(edges_L)
lines_R = make_lines(edges_R)

def make_scatter_groups():
    return {
        "PALM": ax3d.scatter([], [], [], s=POINT_SIZE),
        "T": ax3d.scatter([], [], [], s=POINT_SIZE),
        "I": ax3d.scatter([], [], [], s=POINT_SIZE),
        "M": ax3d.scatter([], [], [], s=POINT_SIZE),
        "R": ax3d.scatter([], [], [], s=POINT_SIZE),
        "L": ax3d.scatter([], [], [], s=POINT_SIZE),
        "GREY": ax3d.scatter([], [], [], s=POINT_SIZE)
    }

scat_L = make_scatter_groups()
scat_R = make_scatter_groups()

def update_line_geom(ln_obj, P, name_to_idx, a, b):
    ia = name_to_idx.get(a); ib = name_to_idx.get(b)
    if ia is None or ib is None:
        ln_obj.set_data([], []); ln_obj.set_3d_properties([]); return
    pa, pb = P[ia], P[ib]
    if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
        ln_obj.set_data([], []); ln_obj.set_3d_properties([]); return
    ln_obj.set_data([pa[0], pb[0]], [pa[1], pb[1]])
    ln_obj.set_3d_properties([pa[2], pb[2]])

def set_hand_styles(selected_dataset): 
    for LINES, hand in [(lines_L,'L'), (lines_R,'R')]:
        for it in LINES:
            ln, kind, finger = it["line"], it["kind"], it["finger"]
            if hand == selected_dataset:
                if kind == 'palm':
                    ln.set_color(COLORS["PALM"]); ln.set_alpha(1.0); ln.set_linewidth(3.0)
                else:
                    ln.set_color(COLORS.get(finger, COLORS["PALM"])); ln.set_alpha(1.0); ln.set_linewidth(3.0)
            else:
                ln.set_color(COLORS["GREY"]); ln.set_alpha(0.25); ln.set_linewidth(1.4)

    for SCAT, hand in [(scat_L,'L'), (scat_R,'R')]:
        for key in ["PALM","T","I","M","R","L"]:
            col = COLORS[key] if hand == selected_dataset else COLORS["GREY"]
            SCAT[key]._facecolor3d = matplotlib.colors.to_rgba(col)
            SCAT[key]._edgecolor3d = matplotlib.colors.to_rgba(col)
        SCAT["GREY"]._facecolor3d = matplotlib.colors.to_rgba(COLORS["GREY"])
        SCAT["GREY"]._edgecolor3d = matplotlib.colors.to_rgba(COLORS["GREY"])

set_hand_styles('R')

# Velocity buffers
vel_t = {f: deque() for f in FINGERS}
vel_v = {f: deque() for f in FINGERS}
def clear_velocity_buffers():
    for f in FINGERS:
        vel_t[f].clear()
        vel_v[f].clear()
        lines_vel[f].set_data([], [])
        axes_vel[f].set_xlim(0, WINDOW_SEC)
        axes_vel[f].set_ylim(0, 1.0)

# State variables for selection/swap handling
prev_idx = None
last_dataset_sel = None 
restart_time0 = None
sel_or_swap_changed = False

def current_dataset_selected():
    """Returns 'L' or 'R' dataset key based on UI selection and swap toggle."""
    ui_right = (hand_var.get() == "Right")
    swap = swap_var.get()
    return ('R' if ui_right else 'L') if not swap else ('L' if ui_right else 'R')

def on_selection_change(event=None):
    global sel_or_swap_changed, prev_idx, last_dataset_sel
    dataset_sel = current_dataset_selected()
    set_hand_styles(dataset_sel)
    last_dataset_sel = dataset_sel
    clear_velocity_buffers()
    prev_idx = None
    sel_or_swap_changed = True
    status_text.set_text(
        f"Selected hand: {hand_var.get()} (dataset: {'Hands_R' if dataset_sel=='R' else 'Hands_L'})"
    )

combo.bind("<<ComboboxSelected>>", on_selection_change)

def on_swap_toggle(*_):
    on_selection_change()
swap_var.trace_add("write", on_swap_toggle)

# ==================== ANIMATION UPDATE LOGIC ====================
def centroid(P, idxs):
    if not idxs: return None
    Q = P[idxs]; Q = Q[np.isfinite(Q).all(axis=1)]
    return None if Q.size==0 else Q.mean(axis=0)

def update_scatter_groups(SCAT, P, groups, selected):
    def set_offsets(sc, pts):
        if pts.size == 0:
            sc._offsets3d = ([], [], [])
        else:
            sc._offsets3d = (pts[:,0], pts[:,1], pts[:,2])

    if selected:
        for key, idxs in groups.items():
            sub = P[idxs] if idxs else np.empty((0,3))
            sub = sub[np.isfinite(sub).all(axis=1)]
            set_offsets(SCAT[key], sub)
        set_offsets(SCAT["GREY"], np.empty((0,3)))
    else:
        Pfin = P[np.isfinite(P).all(axis=1)]
        set_offsets(SCAT["GREY"], Pfin)
        for key in ["PALM","T","I","M","R","L"]:
            set_offsets(SCAT[key], np.empty((0,3)))

def tip_speed(row_prev, row_cur, dt, hand, tips_map):
    out = {}
    for f in FINGERS:
        tip = tips_map.get(f)
        if tip is None: out[f] = np.nan; continue
        cols = joint_axes.get((hand, tip), {})
        try:
            p0 = np.array([row_prev[cols['x']], row_prev[cols['y']], row_prev[cols['z']]], dtype=float)
            p1 = np.array([row_cur[cols['x']],  row_cur[cols['y']],  row_cur[cols['z']]],  dtype=float)
            out[f] = float(np.linalg.norm((p1 - p0) / dt)) if (dt > 0 and np.isfinite(p0).all() and np.isfinite(p1).all()) else np.nan
        except Exception:
            out[f] = np.nan
    return out

def update(frame_idx):
    global prev_idx, last_dataset_sel, restart_time0, sel_or_swap_changed

    row = df.iloc[frame_idx]
    PL = extract_points(row, 'L', order_L) if order_L else np.empty((0,3))
    PR = extract_points(row, 'R', order_R) if order_R else np.empty((0,3))

    mapL = {j:i for i,j in enumerate(order_L)}
    mapR = {j:i for i,j in enumerate(order_R)}
    for it in lines_L: update_line_geom(it["line"], PL, mapL, it["a"], it["b"])
    for it in lines_R: update_line_geom(it["line"], PR, mapR, it["a"], it["b"])

    dataset_sel = current_dataset_selected()
    if dataset_sel != last_dataset_sel:
        set_hand_styles(dataset_sel)
        last_dataset_sel = dataset_sel
        clear_velocity_buffers()
        prev_idx = None
        sel_or_swap_changed = True
        status_text.set_text(
            f"Selected hand: {hand_var.get()} (dataset: {'Hands_R' if dataset_sel=='R' else 'Hands_L'})"
        )

    # Color the selected dataset; the other is gray
    update_scatter_groups(scat_L, PL, groups_L, selected=(dataset_sel=='L'))
    update_scatter_groups(scat_R, PR, groups_R, selected=(dataset_sel=='R'))

    # Update dataset labels near palms
    cL = centroid(PL, groups_L["PALM"])
    cR = centroid(PR, groups_R["PALM"])
    if cL is not None: label_L.set_position((cL[0], cL[1])); label_L.set_3d_properties(cL[2])
    if cR is not None: label_R.set_position((cR[0], cR[1])); label_R.set_3d_properties(cR[2])

    t_abs = frame_time(frame_idx)
    title3d.set_text(f"Typing hands — frame {frame_idx} — t={t_abs:.3f}s")

    # Reset plots on selection/swap change
    if sel_or_swap_changed or restart_time0 is None:
        restart_time0 = t_abs
        sel_or_swap_changed = False
        prev_idx = frame_idx
        return [it["line"] for it in lines_L+lines_R] + \
               list(scat_L.values()) + list(scat_R.values()) + \
               [label_L, label_R, title3d, status_text] + list(lines_vel.values())

    # Calculate and buffer velocities for the SELECTED dataset only
    if prev_idx is not None and frame_idx > prev_idx:
        dt = max(frame_time(frame_idx) - frame_time(prev_idx), 1e-12)
        if dataset_sel == 'R':
            speeds = tip_speed(df.iloc[prev_idx], row, dt, 'R', tips_R)
        else:
            speeds = tip_speed(df.iloc[prev_idx], row, dt, 'L', tips_L)

        t_rel = t_abs - restart_time0
        for f in FINGERS:
            vel_t[f].append(t_rel)
            vel_v[f].append(speeds.get(f, np.nan))
            # Sliding window maintenance
            while len(vel_t[f])>1 and (t_rel - vel_t[f][0]) > WINDOW_SEC:
                vel_t[f].popleft(); vel_v[f].popleft()
            times = np.fromiter(vel_t[f], dtype=float)
            vals  = np.fromiter(vel_v[f], dtype=float)
            lines_vel[f].set_data(times, vals)
            ax = axes_vel[f]
            ax.set_xlim(max(0.0, t_rel - WINDOW_SEC), t_rel if t_rel>0 else WINDOW_SEC)
            finite_vals = vals[np.isfinite(vals)]
            ymax = max(np.percentile(finite_vals, 98), 1e-3) if finite_vals.size else 1.0
            ax.set_ylim(0.0, ymax)

    prev_idx = frame_idx
    artists = [it["line"] for it in lines_L+lines_R]
    artists += list(scat_L.values()) + list(scat_R.values())
    artists += [label_L, label_R, title3d, status_text]
    artists += list(lines_vel.values())
    return artists

interval_ms = int(1000 / TARGET_FPS)
anim = FuncAnimation(fig, update, frames=frame_indices, interval=interval_ms, blit=False)

def on_close():
    try: anim.event_source.stop()
    except Exception: pass
    root.quit(); root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
canvas.draw_idle()

# One-time initial centroid log
try:
    row0 = df.iloc[0]
    def safe_centroid(P, idxs):
        if not idxs: return None
        Q = P[idxs]; Q = Q[np.isfinite(Q).all(axis=1)]
        return None if Q.size==0 else Q.mean(axis=0)
    print("Initial palm centroids (Hands_L vs Hands_R):")
    print("  L centroid:", safe_centroid(extract_points(row0, 'L', order_L), groups_L["PALM"]))
    print("  R centroid:", safe_centroid(extract_points(row0, 'R', order_R), groups_R["PALM"]))
except Exception as e:
    print("Centroid check error:", e)

root.mainloop()
