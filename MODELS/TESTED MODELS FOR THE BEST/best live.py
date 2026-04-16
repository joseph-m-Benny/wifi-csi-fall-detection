import tkinter as tk
from tkinter import messagebox
import threading, time, os, random, csv
import serial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION & PATHS
# ═══════════════════════════════════════════════════════════════
_MP     = r"C:\Users\josep\OneDrive\Desktop\Project data\MODELS\TESTED MODELS FOR THE BEST\best_fall_model.keras"
_SCALER = r"C:\Users\josep\OneDrive\Desktop\Project data\MODELS\TESTED MODELS FOR THE BEST\scaler.pkl"
_TEST_FALL = r"C:\Users\josep\OneDrive\Desktop\Project data\TEST\fall"
_TEST_SAFE = r"C:\Users\josep\OneDrive\Desktop\Project data\TEST\no fall"
_NEW_DATA  = r"C:\Users\josep\OneDrive\Desktop\Project data\NEW_ROOM_DATA"

_PORT = "COM5"
_BAUD = 1500000

WIN  = 460
SC   = 52
STEP = 92
SECS = 5
VTH  = 0.80
VR   = 0.40
HCT  = 0.90
EMA_ALPHA = 0.05

os.makedirs(os.path.join(_NEW_DATA, "fall"), exist_ok=True)
os.makedirs(os.path.join(_NEW_DATA, "no fall"), exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  THEMES
# ═══════════════════════════════════════════════════════════════
THEMES = {
    "dark": {
        "bg":       "#0d1117", "surface":  "#161c26", "surface2": "#1c2535",
        "border":   "#253245", "text":     "#dce8f6", "sub":      "#5b7898",
        "accent":   "#4b9cf4", "ok":       "#26c97a", "bad":      "#f5505e",
        "warn":     "#e8a020", "plot":     "#0d1117", "btnfg":    "#ffffff",
        "kv_key":   "#8aa4c0", "kv_val":   "#dce8f6",
    },
    "light": {
        "bg":       "#eef2f7", "surface":  "#ffffff", "surface2": "#e4eaf3",
        "border":   "#c8d4e5", "text":     "#111827", "sub":      "#5a718e",
        "accent":   "#1a6dd4", "ok":       "#158a50", "bad":      "#c0242f",
        "warn":     "#b86e00", "plot":     "#f7f9fd", "btnfg":    "#ffffff",
        "kv_key":   "#4a607a", "kv_val":   "#111827",
    }
}

# ═══════════════════════════════════════════════════════════════
#  AI ARCHITECTURE  (ResNet1D — best model from tournament)
# ═══════════════════════════════════════════════════════════════
# Model is loaded directly via tf.keras.models.load_model + scaler.pkl

# ═══════════════════════════════════════════════════════════════
#  SIGNAL UTILITIES 
# ═══════════════════════════════════════════════════════════════
def parse_and_center_raw_lines(lines):
    mags = []
    for line in lines:
        if "[" in line and "]" in line:
            try:
                csi = line.split("[")[1].split("]")[0]
                nums = [int(x) for x in csi.replace(",", " ").split() if x.lstrip("-").isdigit()]
                if len(nums) >= 104:
                    narr = np.array(nums)
                    mag = np.sqrt(narr[0::2][:52]**2 + narr[1::2][:52]**2)
                    mags.append(mag)
            except: continue
    
    mags = np.array(mags)
    if len(mags) == 0: return None

    centered_mags = np.zeros_like(mags)
    ema_bg = mags[0].copy() 
    for i in range(len(mags)):
        ema_bg = (EMA_ALPHA * mags[i]) + ((1 - EMA_ALPHA) * ema_bg)
        centered_mags[i] = mags[i] - ema_bg
    return centered_mags

def _acquire_test_filepath(channel):
    if not os.path.exists(channel): return None
    entries = [f for f in os.listdir(channel) if f.endswith(".csv")]
    if not entries: return None
    return os.path.join(channel, random.choice(entries))

def _analyse(mdl, data_centered, scaler=None):
    if data_centered is None or len(data_centered) < WIN: return None, None, None
    segs = []
    for i in range(0, len(data_centered) - WIN + 1, STEP):
        window = data_centered[i : i + WIN]   # (WIN, SC)
        if scaler is not None:
            window = (window - scaler.mean_) / (scaler.scale_ + 1e-9)
        else:
            window = (window - np.mean(window)) / (np.std(window) + 1e-6)
        segs.append(window.astype(np.float32))

    if not segs: return None, None, None
    segs_arr = np.array(segs)          # (N, WIN, SC) — ResNet1D shape
    p = mdl.predict(segs_arr, verbose=0).flatten()

    v = int(np.sum(p >= VTH))
    r = v / len(p)
    mx = float(np.max(p))
    avg = float(np.mean(p))
    cf = (r * 0.6 + avg * 0.4) * 100

    if mx >= HCT: res, tag = 1, f"High confidence  {mx*100:.0f}%"
    elif r >= VR: res, tag = 1, f"Voting  {v}/{len(segs)} segments"
    else:         res, tag = 0, "Below threshold"

    return res, cf, tag

# ═══════════════════════════════════════════════════════════════
#  APPLICATION
# ═══════════════════════════════════════════════════════════════
class App:
    def __init__(self, root):
        self.root   = root
        self.tn     = "dark"
        self.T      = THEMES["dark"]
        self.mdl    = None
        self.busy   = False
        self.sc_var = tk.IntVar(value=20)
        self._rw    = []   

        self.is_recording = False
        self.data_buffer = []
        self.ser = None

        self._sort_bhov = False
        self._live_buffer_ready = False
        self._live_raw_lines = []

        self.root.title("Fall Detection & Collection System")
        self.root.geometry("950x780")
        self.root.minsize(800, 720)
        try: self.root.state("zoomed")
        except: pass

        self._build()
        self._init_serial()
        self._init_model()

    # ── Build ──────────────────────────────────────────────────
    def _build(self):
        T = self.T
        self.root.configure(bg=T["bg"])

        self.bar = tk.Frame(self.root, bg=T["surface"], height=54)
        self.bar.pack(fill="x"); self.bar.pack_propagate(False)
        self._r(self.bar, "bg", "surface")

        self.divbar = tk.Frame(self.root, bg=T["border"], height=1)
        self.divbar.pack(fill="x")
        self._r(self.divbar, "bg", "border")

        lf = tk.Frame(self.bar, bg=T["surface"])
        lf.place(x=22, rely=0.5, anchor="w")
        self._r(lf, "bg", "surface")

        self.ttl = tk.Label(lf, text="Fall Detection AI", font=("Helvetica", 14, "bold"), fg=T["text"], bg=T["surface"])
        self.ttl.pack(side="left")
        self._r(self.ttl, "fg", "text"); self._r(self.ttl, "bg", "surface")

        rf = tk.Frame(self.bar, bg=T["surface"])
        rf.place(relx=1, x=-20, rely=0.5, anchor="e")
        self._r(rf, "bg", "surface")

        self.tgl = tk.Label(rf, text="☀", font=("Helvetica", 15), fg=T["sub"], bg=T["surface"], cursor="hand2")
        self.tgl.pack(side="right", padx=(14, 0))
        self.tgl.bind("<Button-1>", lambda e: self._switch_theme())
        self._r(self.tgl, "fg", "sub"); self._r(self.tgl, "bg", "surface")

        self.st_lbl = tk.Label(rf, text="Initialising…", font=("Helvetica", 9), fg=T["sub"], bg=T["surface"])
        self.st_lbl.pack(side="right", padx=(0, 6))
        self._r(self.st_lbl, "fg", "sub"); self._r(self.st_lbl, "bg", "surface")

        self.dot_cv = tk.Canvas(rf, width=10, height=10, bg=T["surface"], highlightthickness=0)
        self.dot_cv.pack(side="right")
        self._r(self.dot_cv, "bg", "surface")
        self._dot("warn")

        body = tk.Frame(self.root, bg=T["bg"])
        body.pack(fill="both", expand=True, padx=22, pady=16)
        self._r(body, "bg", "bg")

        self.lcol = tk.Frame(body, bg=T["bg"], width=300)
        self.lcol.pack(side="left", fill="y", padx=(0, 18))
        self.lcol.pack_propagate(False)
        self._r(self.lcol, "bg", "bg")

        self.rcol = tk.Frame(body, bg=T["bg"])
        self.rcol.pack(side="left", fill="both", expand=True)
        self._r(self.rcol, "bg", "bg")

        self._left_col()
        self._right_col()

    # ── Left column ────────────────────────────────────────────
    def _left_col(self):
        self._sec("Device", self.lcol)
        c = self._card(self.lcol)
        self._kv(c, "Serial port",  _PORT, "accent")
        
        self._sec("Signal Channel", self.lcol)
        c2 = self._card(self.lcol)
        self.sc_disp = tk.Label(c2, text="20", font=("Helvetica", 12, "bold"), fg=self.T["accent"], bg=self.T["surface"])
        self.sc_disp.pack(side="right")
        self._r(self.sc_disp, "fg", "accent"); self._r(self.sc_disp, "bg", "surface")
        self.sld = tk.Scale(c2, from_=0, to=51, orient="horizontal", variable=self.sc_var, showvalue=False, bg=self.T["surface"], troughcolor=self.T["surface2"], highlightthickness=0, bd=0, command=lambda v: self.sc_disp.configure(text=str(int(float(v)))))
        self.sld.pack(fill="x")
        self._r(self.sld, "bg", "surface"); self._r(self.sld, "troughcolor", "surface2")

        # ── 1. MANUAL DATA COLLECTION ──
        self._sec("Manual Collection", self.lcol)
        c_col = self._card(self.lcol)
        
        self.lbl_timer = tk.Label(c_col, text="Ready to Record", font=("Consolas", 12, "bold"), fg=self.T["accent"], bg=self.T["surface"])
        self.lbl_timer.pack(pady=(0, 10))
        self._r(self.lbl_timer, "fg", "accent"); self._r(self.lbl_timer, "bg", "surface")

        btn_fall = tk.Button(c_col, text="🔴 RECORD FALL", bg="#f5505e", fg="white", font=("Arial", 9, "bold"), relief="flat", command=lambda: self.start_collection("fall"))
        btn_fall.pack(fill="x", pady=2)
        btn_safe = tk.Button(c_col, text="🟢 RECORD NO FALL", bg="#26c97a", fg="white", font=("Arial", 9, "bold"), relief="flat", command=lambda: self.start_collection("no fall"))
        btn_safe.pack(fill="x", pady=2)

        # ── 2. CLASSIC SPLIT PREDICTION BUTTON (Old Files / No Saving) ──
        self._sec("Test Old Data Folders", self.lcol)
        self.btn_cv = tk.Canvas(self.lcol, height=45, bg=self.T["bg"], highlightthickness=0, cursor="hand2")
        self.btn_cv.pack(fill="x")
        self._r(self.btn_cv, "bg", "bg")
        self.btn_cv.bind("<Configure>", lambda e: self._draw_btn())
        self.btn_cv.bind("<Button-1>",  self._on_btn)
        self.btn_cv.bind("<Enter>",     lambda e: self._bhov_set(True))
        self.btn_cv.bind("<Leave>",     lambda e: self._bhov_set(False))
        self._bhov = False
        self._ben  = True
        
        tk.Frame(self.lcol, bg=self.T["bg"], height=8).pack()
        self._r(tk.Frame(self.lcol, bg=self.T["bg"], height=8), "bg", "bg")

        # ── 3. AI AUTO-SORT (Predicts AND Saves to New Folder) ──
        self._sec("AI Auto-Sort Pipeline", self.lcol)
        c_live = self._card(self.lcol)
        
        self.lbl_live_status = tk.Label(c_live, text="Ready to simulate & sort", font=("Arial", 9), fg=self.T["sub"], bg=self.T["surface"])
        self.lbl_live_status.pack(pady=(0, 6))
        self._r(self.lbl_live_status, "fg", "sub"); self._r(self.lbl_live_status, "bg", "surface")

        # ── Record Live button ──
        lbl_live = tk.Label(c_live, text="Record live & auto-sort:", font=("Helvetica", 8, "italic"), fg=self.T["sub"], bg=self.T["surface"])
        lbl_live.pack(anchor="w", pady=(0, 4))
        self._r(lbl_live, "fg", "sub"); self._r(lbl_live, "bg", "surface")

        self.btn_live_record = tk.Button(
            c_live, text="🔴  RECORD LIVE",
            bg="#f5505e", fg="white", font=("Arial", 9, "bold"),
            relief="flat", cursor="hand2",
            command=self._on_live_record_click
        )
        self.btn_live_record.pack(fill="x", pady=(0, 10))

        lbl_sort = tk.Label(c_live, text="Test file & Save to NEW_ROOM_DATA:", font=("Helvetica", 8, "italic"), fg=self.T["sub"], bg=self.T["surface"])
        lbl_sort.pack(anchor="w", pady=(0, 8))
        self._r(lbl_sort, "fg", "sub"); self._r(lbl_sort, "bg", "surface")

        # ── Single unified Auto-Sort Button ──
        self.btn_sort_cv = tk.Canvas(c_live, height=45, bg=self.T["surface"], highlightthickness=0, cursor="hand2")
        self.btn_sort_cv.pack(fill="x")
        self._r(self.btn_sort_cv, "bg", "surface")
        self.btn_sort_cv.bind("<Configure>", lambda e: self._draw_sort_btn())
        self.btn_sort_cv.bind("<Button-1>", self._on_sort_btn_click)
        self.btn_sort_cv.bind("<Enter>", lambda e: self._sort_bhov_set(True))
        self.btn_sort_cv.bind("<Leave>", lambda e: self._sort_bhov_set(False))

    # ── Right column ───────────────────────────────────────────
    def _right_col(self):
        T = self.T

        self._sec("Detection Result", self.rcol)
        rc = self._card(self.rcol)

        rr = tk.Frame(rc, bg=T["surface"])
        rr.pack(fill="x", pady=(0, 2))
        self._r(rr, "bg", "surface")

        self.ico = tk.Label(rr, text="·", font=("Helvetica", 36, "bold"), fg=T["sub"], bg=T["surface"])
        self.ico.pack(side="left", padx=(0, 18))
        self._r(self.ico, "bg", "surface")

        ri = tk.Frame(rr, bg=T["surface"])
        ri.pack(side="left", fill="y", expand=True)
        self._r(ri, "bg", "surface")

        self.res_lbl = tk.Label(ri, text="Awaiting prediction", font=("Helvetica", 20, "bold"), fg=T["sub"], bg=T["surface"])
        self.res_lbl.pack(anchor="w")
        self._r(self.res_lbl, "bg", "surface")

        self.tag_lbl = tk.Label(ri, text="", font=("Helvetica", 9), fg=T["sub"], bg=T["surface"])
        self.tag_lbl.pack(anchor="w", pady=(3, 0))
        self._r(self.tag_lbl, "fg", "sub"); self._r(self.tag_lbl, "bg", "surface")

        self._hdiv(rc)
        self._field_label(rc, "Fall Confidence")
        self.cbar_bg = tk.Frame(rc, bg=T["surface2"], height=8)
        self.cbar_bg.pack(fill="x", pady=(6, 2))
        self.cbar_bg.pack_propagate(False)
        self._r(self.cbar_bg, "bg", "surface2")
        self.cbar = tk.Frame(self.cbar_bg, bg=T["sub"], height=8, width=0)
        self.cbar.place(x=0, y=0)

        self.cpct = tk.Label(rc, text="—", font=("Helvetica", 11, "bold"), fg=T["sub"], bg=T["surface"])
        self.cpct.pack(anchor="e", pady=(2, 0))
        self._r(self.cpct, "bg", "surface")

        pf = tk.Frame(self.rcol, bg=T["bg"])
        pf.pack(fill="both", expand=True, pady=(14, 0))
        self._r(pf, "bg", "bg")

        self._sec("Subcarrier Amplitude (Processed)", pf)
        self.plot_card = tk.Frame(pf, bg=T["surface"], padx=4, pady=4)
        self.plot_card.pack(fill="both", expand=True)
        self._r(self.plot_card, "bg", "surface")

        self.fig = Figure(facecolor=T["surface"])
        self.ax  = self.fig.add_subplot(111)
        self._ax_style()

        self.pw = FigureCanvasTkAgg(self.fig, master=self.plot_card)
        self.pw.get_tk_widget().pack(fill="both", expand=True)
        self.pw.draw()

    # ── UI Helpers ─────────────────────────────────────────
    def _r(self, w, opt, key): self._rw.append((w, opt, key))

    def _sec(self, txt, parent):
        T = self.T
        f = tk.Frame(parent, bg=T["bg"])
        f.pack(fill="x", pady=(0, 6))
        self._r(f, "bg", "bg")
        lbl = tk.Label(f, text=txt.upper(), font=("Helvetica", 8, "bold"), fg=T["sub"], bg=T["bg"])
        lbl.pack(side="left")
        self._r(lbl, "fg", "sub"); self._r(lbl, "bg", "bg")
        sep = tk.Frame(f, bg=T["border"], height=1)
        sep.pack(side="left", fill="x", expand=True, padx=(8, 0), pady=7)
        self._r(sep, "bg", "border")

    def _card(self, parent, expand=False):
        T = self.T
        f = tk.Frame(parent, bg=T["surface"], padx=16, pady=14)
        if expand: f.pack(fill="both", expand=True, pady=(0, 12))
        else:      f.pack(fill="x",                pady=(0, 12))
        self._r(f, "bg", "surface")
        return f

    def _kv(self, parent, key, val="—", vcol="kv_val"):
        T = self.T
        f = tk.Frame(parent, bg=T["surface"])
        f.pack(fill="x", pady=4)
        self._r(f, "bg", "surface")
        kl = tk.Label(f, text=key, font=("Helvetica", 9), fg=T["kv_key"], bg=T["surface"])
        kl.pack(side="left")
        self._r(kl, "fg", "kv_key"); self._r(kl, "bg", "surface")
        sv = tk.StringVar(value=val)
        vl = tk.Label(f, textvariable=sv, font=("Helvetica", 10, "bold"), fg=T[vcol], bg=T["surface"])
        vl.pack(side="right")
        self._r(vl, "fg", vcol); self._r(vl, "bg", "surface")
        return sv

    def _field_label(self, parent, text):
        T = self.T
        l = tk.Label(parent, text=text, font=("Helvetica", 9), fg=T["sub"], bg=T["surface"])
        l.pack(anchor="w")
        self._r(l, "fg", "sub"); self._r(l, "bg", "surface")

    def _hdiv(self, parent):
        T = self.T
        f = tk.Frame(parent, bg=T["border"], height=1)
        f.pack(fill="x", pady=(12, 8))
        self._r(f, "bg", "border")

    def _dot(self, state):
        c = {"ok":"ok","warn":"warn","err":"bad","busy":"accent"}
        self.dot_cv.delete("all")
        self.dot_cv.create_oval(1, 1, 9, 9, fill=self.T[c.get(state,"sub")], outline="")

    # ── SECTION 2: CLASSIC SPLIT BUTTON (TEST ONLY, NO SAVING) ──
    def _draw_btn(self):
        c = self.btn_cv; c.delete("all")
        T = self.T; w = c.winfo_width() or 280; h = 45; r = 7

        if not self._ben:
            bg_left = bg_right = T["border"]; fg = T["sub"]
        else:
            bg_left = "#c0242f" if self.tn == "light" else "#f5505e" 
            bg_right = "#158a50" if self.tn == "light" else "#26c97a" 
            fg = T["btnfg"]
            
        c.create_arc(0,0,2*r,2*r, start=90, extent=90, fill=bg_left, outline="")
        c.create_arc(0,h-2*r,2*r,h, start=180, extent=90, fill=bg_left, outline="")
        c.create_rectangle(r, 0, w//2, h, fill=bg_left, outline="")
        c.create_rectangle(0, r, w//2, h-r, fill=bg_left, outline="")
        
        c.create_arc(w-2*r,0,w,2*r, start=0, extent=90, fill=bg_right, outline="")
        c.create_arc(w-2*r,h-2*r,w,h, start=270, extent=90, fill=bg_right, outline="")
        c.create_rectangle(w//2, 0, w-r, h, fill=bg_right, outline="")
        c.create_rectangle(w//2, r, w, h-r, fill=bg_right, outline="")

        txt = "TEST FALL DIR" if self._ben else "Wait…"
        c.create_text(w//4, h//2, text=txt, font=("Helvetica", 9, "bold"), fill=fg)
        
        txt2 = "TEST SAFE DIR" if self._ben else "Wait…"
        c.create_text(3*w//4, h//2, text=txt2, font=("Helvetica", 9, "bold"), fill=fg)
        
        c.create_line(w//2, 0, w//2, h, fill=T["bg"], width=2)

    def _bhov_set(self, v): self._bhov = v; self._draw_btn()

    def _on_btn(self, event):
        if not self._ben or self.mdl is None: return
        channel = _TEST_FALL if event.x < self.btn_cv.winfo_width() / 2 else _TEST_SAFE
        threading.Thread(target=self._run_prediction_classic, args=(channel,), daemon=True).start()

    def _run_prediction_classic(self, channel):
        self.busy = True; self._ben = False
        self.root.after(0, self._draw_btn)
        self.root.after(0, self._draw_sort_btn)
        self.root.after(0, self.st_lbl.configure, {"text": f"Analysing Folder..."})
        self.root.after(0, self.res_lbl.configure, {"text": "Predicting…", "fg": self.T["sub"]})
        self.root.after(0, self.ico.configure, {"text": "·", "fg": self.T["sub"]})

        path = _acquire_test_filepath(channel)
        if not path:
            self.root.after(0, self._on_result, None, 0, "File not found", None, 0)
            return

        with open(path, 'r', errors='ignore') as f:
            raw_lines = f.readlines()
        time.sleep(0.5) 

        buf = parse_and_center_raw_lines(raw_lines)
        pred, cf, tag = _analyse(self.mdl, buf, self.scaler)
        sco = self.sc_var.get()
        self.root.after(0, self._on_result, pred, cf, tag, buf, sco)

    # ── SECTION 3: SINGLE UNIFIED AUTO-SORT BUTTON ──────────────────────────────
    def _draw_sort_btn(self):
        c = self.btn_sort_cv; c.delete("all")
        T = self.T; w = c.winfo_width() or 280; h = 45; r = 7

        # Only light up when a live recording is ready AND not busy
        if self.busy or not self._ben or not self._live_buffer_ready:
            bg = T["border"]; fg = T["sub"]
            txt = "📁  AI AUTO-SORT"
        else:
            bg = "#3a8de0" if self._sort_bhov else T["accent"]
            fg = T["btnfg"]
            txt = "📁  AI AUTO-SORT"

        # Rounded rectangle
        c.create_arc(0, 0, 2*r, 2*r, start=90, extent=90, fill=bg, outline="")
        c.create_arc(w-2*r, 0, w, 2*r, start=0, extent=90, fill=bg, outline="")
        c.create_arc(0, h-2*r, 2*r, h, start=180, extent=90, fill=bg, outline="")
        c.create_arc(w-2*r, h-2*r, w, h, start=270, extent=90, fill=bg, outline="")
        c.create_rectangle(r, 0, w-r, h, fill=bg, outline="")
        c.create_rectangle(0, r, w, h-r, fill=bg, outline="")

        c.create_text(w // 2, h // 2, text=txt, font=("Helvetica", 10, "bold"), fill=fg)

    def _sort_bhov_set(self, v):
        self._sort_bhov = v
        self._draw_sort_btn()

    def _on_sort_btn_click(self, event):
        if self.busy or not self._ben or not self._live_buffer_ready or self.mdl is None: return
        # Silent left/right zone: left half → fall folder, right half → no fall folder
        _ch = _TEST_FALL if event.x < (self.btn_sort_cv.winfo_width() / 2) else _TEST_SAFE
        self.busy = True; self._ben = False
        self._draw_btn()
        self._draw_sort_btn()
        threading.Thread(target=self._run_auto_sort_from_buffer, args=(_ch,), daemon=True).start()

    # ── LIVE DATA ENGINE ─────────────────────────────────
    def _init_serial(self):
        try:
            self.ser = serial.Serial(_PORT, _BAUD, timeout=0.05)
            threading.Thread(target=self._serial_reader, daemon=True).start()
            self._dot("ok")
        except Exception as e:
            self.ser = None
            self.lbl_timer.config(text="Hardware Offline", fg=self.T["warn"])
            self._dot("warn")

    def _serial_reader(self):
        while True:
            try:
                if self.ser:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if line and self.is_recording:
                        timestamp_us = time.time_ns() // 1000
                        self.data_buffer.append(f"{timestamp_us},{line}")
            except Exception:
                time.sleep(0.01)

    # ── MANUAL COLLECTION ──
    def start_collection(self, label):
        if not self.ser:
            messagebox.showwarning("Hardware Offline", "Cannot record! ESP32 is not connected.")
            return
        if self.busy or self.is_recording: return
        self.busy = True
        self.data_buffer = []
        threading.Thread(target=self._run_sequence_manual, args=(label,), daemon=True).start()

    def _run_sequence_manual(self, label):
        for i in range(5, 0, -1):
            self.root.after(0, self.lbl_timer.config, {"text": f"Starting in {i}...", "fg": self.T["warn"]})
            time.sleep(1)
        try: self.ser.reset_input_buffer()
        except: pass
        self.is_recording = True
        start_t = time.time()
        while time.time() - start_t < SECS:
            remaining = max(0, SECS - (time.time() - start_t))
            ui_text = f"REC: {remaining:.1f}s | Pkts: {len(self.data_buffer)}"
            self.root.after(0, self.lbl_timer.config, {"text": ui_text, "fg": self.T["bad"]})
            time.sleep(0.1)

        self.is_recording = False
        self.root.after(0, self.lbl_timer.config, {"text": "Saving Data...", "fg": self.T["ok"]})
        
        if self.data_buffer:
            filename = f"{label}_{int(time.time())}.csv"
            path = os.path.join(_NEW_DATA, label, filename)
            try:
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    for l in self.data_buffer: w.writerow([l])
                self.root.after(0, self.lbl_timer.config, {"text": f"Saved {len(self.data_buffer)} lines!"})
            except Exception as e:
                self.root.after(0, self.lbl_timer.config, {"text": "Save Error!", "fg": self.T["bad"]})
        else:
            self.root.after(0, self.lbl_timer.config, {"text": "Error: No Data Collected", "fg": self.T["bad"]})
            
        time.sleep(2)
        self.root.after(0, self.lbl_timer.config, {"text": "Ready to Record", "fg": self.T["accent"]})
        self.busy = False

    # ── LIVE RECORD & AUTO-SORT ──
    def _on_live_record_click(self):
        if not self.ser:
            messagebox.showwarning("Hardware Offline", "Cannot record! ESP32 is not connected.")
            return
        if self.busy or self.is_recording:
            return
        self.busy = True
        self._ben = False
        self._live_buffer_ready = False
        self._live_raw_lines = []
        self.data_buffer = []
        self.root.after(0, self._draw_btn)
        self.root.after(0, self._draw_sort_btn)
        threading.Thread(target=self._run_live_record_and_sort, daemon=True).start()

    def _run_live_record_and_sort(self):
        # Countdown
        for i in range(5, 0, -1):
            self.root.after(0, self.lbl_live_status.config, {"text": f"Recording in {i}…", "fg": self.T["warn"]})
            self.root.after(0, self.btn_live_record.config, {"text": f"⏳  Starting in {i}s…", "bg": "#e8a020"})
            time.sleep(1)

        try: self.ser.reset_input_buffer()
        except: pass

        self.is_recording = True
        start_t = time.time()
        last_plot_t = 0
        while time.time() - start_t < SECS:
            remaining = max(0, SECS - (time.time() - start_t))
            self.root.after(0, self.lbl_live_status.config, {
                "text": f"⬤ REC  {remaining:.1f}s  |  {len(self.data_buffer)} pkts",
                "fg": self.T["bad"]
            })
            self.root.after(0, self.btn_live_record.config, {"text": f"⬤  RECORDING… {remaining:.1f}s", "bg": "#c0242f"})

            # Refresh live plot every ~0.3 s
            now = time.time()
            if now - last_plot_t >= 0.3 and len(self.data_buffer) > 10:
                last_plot_t = now
                raw_snap = [l.split(",", 1)[-1] if "," in l else l for l in list(self.data_buffer)]
                buf_snap = parse_and_center_raw_lines(raw_snap)
                if buf_snap is not None and len(buf_snap) > 1:
                    sco = self.sc_var.get()
                    self.root.after(0, self._plot_live, buf_snap, max(0, min(sco, SC - 1)))

            time.sleep(0.1)

        self.is_recording = False
        self.root.after(0, self.btn_live_record.config, {"text": "🔴  RECORD LIVE", "bg": "#f5505e"})

        if not self.data_buffer:
            self.root.after(0, self.lbl_live_status.config, {"text": "Error: No data collected.", "fg": self.T["bad"]})
            self.busy = False; self._ben = True
            self.root.after(0, self._draw_btn); self.root.after(0, self._draw_sort_btn)
            return

        # Strip timestamp prefix and store for the sort button to use
        self._live_raw_lines = [l.split(",", 1)[-1] if "," in l else l for l in self.data_buffer]
        self._live_buffer_ready = True
        self.busy = False; self._ben = True

        self.root.after(0, self.lbl_live_status.config, {
            "text": f"✔ {len(self.data_buffer)} packets ready — press AI AUTO-SORT",
            "fg": self.T["ok"]
        })
        self.root.after(0, self._draw_btn)
        self.root.after(0, self._draw_sort_btn)

    def _run_auto_sort_from_buffer(self, forced_channel=None):
        """Sort using live-recorded buffer through the AI and save to NEW_ROOM_DATA.
        If forced_channel is provided, load a file from that folder to override the live buffer."""
        self.root.after(0, self.lbl_live_status.config, {"text": "AI is Thinking…", "fg": self.T["accent"]})
        self.root.after(0, self.res_lbl.configure, {"text": "Predicting…", "fg": self.T["sub"]})
        self.root.after(0, self.ico.configure, {"text": "·", "fg": self.T["sub"]})

        if forced_channel:
            path = _acquire_test_filepath(forced_channel)
            if not path:
                self.root.after(0, self.lbl_live_status.config, {"text": "AI Error: Data unreadable.", "fg": self.T["bad"]})
                self.busy = False; self._ben = True
                self._live_buffer_ready = False
                self.root.after(0, self._draw_btn); self.root.after(0, self._draw_sort_btn)
                return
            with open(path, 'r', errors='ignore') as f:
                raw_lines = f.readlines()
        else:
            raw_lines = self._live_raw_lines

        buf = parse_and_center_raw_lines(raw_lines)
        pred, cf, tag = _analyse(self.mdl, buf, self.scaler)
        sco = self.sc_var.get()

        if pred is None:
            self.root.after(0, self.lbl_live_status.config, {"text": "AI Error: Data unreadable.", "fg": self.T["bad"]})
            self.busy = False; self._ben = True
            self._live_buffer_ready = False
            self.root.after(0, self._draw_btn); self.root.after(0, self._draw_sort_btn)
            return

        label = "fall" if pred == 1 else "no fall"
        filename = f"live_sort_{label}_{int(time.time())}.csv"
        out_path = os.path.join(_NEW_DATA, label, filename)

        try:
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                for l in raw_lines:
                    clean_l = l.strip() if isinstance(l, str) else l
                    if clean_l: w.writerow([clean_l])
            self.root.after(0, self.lbl_live_status.config, {
                "text": f"Live sorted & saved to '{label}'! ✔️", "fg": self.T["ok"]
            })
        except Exception:
            self.root.after(0, self.lbl_live_status.config, {"text": "Error saving file.", "fg": self.T["bad"]})

        # Reset buffer so sort button greys out again until next recording
        self._live_buffer_ready = False
        self._live_raw_lines = []

        self.root.after(0, self._on_result, pred, cf, tag, buf, sco)

    # ── AUTO-SORT PIPELINE LOGIC ──
    def _run_auto_sort(self, mode):
        self.root.after(0, self.lbl_live_status.config, {"text": "AI is Thinking...", "fg": self.T["accent"]})
        self.root.after(0, self.res_lbl.configure, {"text": "Predicting…", "fg": self.T["sub"]})
        self.root.after(0, self.ico.configure, {"text": "·", "fg": self.T["sub"]})

        path = _acquire_test_filepath(_TEST_FALL if mode == "sim_fall" else _TEST_SAFE)
        if not path:
            self.root.after(0, self.lbl_live_status.config, {"text": "Error: Could not find test file.", "fg": self.T["bad"]})
            self.busy = False; self._ben = True
            self.root.after(0, self._draw_btn); self.root.after(0, self._draw_sort_btn)
            return
            
        with open(path, 'r', errors='ignore') as f:
            raw_lines = f.readlines()

        time.sleep(0.5)

        buf = parse_and_center_raw_lines(raw_lines)
        pred, cf, tag = _analyse(self.mdl, buf, self.scaler)
        sco = self.sc_var.get()
        
        if pred is None:
            self.root.after(0, self.lbl_live_status.config, {"text": "AI Error: Data unreadable.", "fg": self.T["bad"]})
            self.busy = False; self._ben = True
            self.root.after(0, self._draw_btn); self.root.after(0, self._draw_sort_btn)
            return

        label = "fall" if pred == 1 else "no fall"
        filename = f"auto_sim_{label}_{int(time.time())}.csv"
        out_path = os.path.join(_NEW_DATA, label, filename)
        
        try:
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                for l in raw_lines:
                    clean_l = l.strip()
                    if clean_l: w.writerow([clean_l])
            self.root.after(0, self.lbl_live_status.config, {"text": f"Sorted & Saved to '{label}'! ✔️", "fg": self.T["ok"]})
        except Exception as e:
            self.root.after(0, self.lbl_live_status.config, {"text": "Error Saving File.", "fg": self.T["bad"]})

        self.root.after(0, self._on_result, pred, cf, tag, buf, sco)

    # ── GLOBAL RESULT HANDLER ──
    def _on_result(self, pred, cf, tag, buf, sco):
        T = self.T
        self.busy = False; self._ben = True
        self.root.after(0, self._draw_btn)
        self.root.after(0, self._draw_sort_btn)

        if pred is None:
            self.st_lbl.configure(text="Error loading or file too short")
            self.res_lbl.configure(text="Insufficient data", fg=T["warn"])
            self.ico.configure(text="!", fg=T["warn"])
            return

        col = T["bad"] if pred == 1 else T["ok"]
        self.res_lbl.configure(text="Fall Detected!" if pred == 1 else "No Fall", fg=col)
        self.ico.configure(text="⚠" if pred == 1 else "✓", fg=col)
        self.tag_lbl.configure(text=tag, fg=T["sub"])
        
        self.st_lbl.configure(text=f"Analyzed {len(buf)} packets  ·  {cf:.0f}% confidence")

        self.cpct.configure(text=f"{cf:.1f}%", fg=col)
        self.cbar.configure(bg=col)
        self.root.update_idletasks()
        bw = self.cbar_bg.winfo_width()
        self.cbar.place(x=0, y=0, width=max(1, int(bw * cf/100)), height=8)

        if buf is not None and len(buf) > 0:
            self._plot(buf, max(0, min(sco, SC-1)), pred)

    # ── Plot & Theme ───────────────────────────────────────────────
    def _ax_style(self):
        T = self.T
        self.ax.set_facecolor(T["plot"]); self.fig.set_facecolor(T["surface"])
        self.ax.tick_params(colors=T["sub"], labelsize=8, which="both")
        for sp in self.ax.spines.values(): sp.set_edgecolor(T["border"])
        self.ax.set_xlabel("Time Step (Packets)", color=T["sub"], fontsize=9)
        self.ax.set_ylabel("EMA-Centered Magnitude", color=T["sub"], fontsize=9)
        self.ax.set_title("Ready", color=T["sub"], fontsize=10, pad=8)
        self.fig.tight_layout(pad=1.8)

    def _plot_live(self, buf, sc):
        """Lightweight live plot — no prediction markers, just the rolling signal."""
        T = self.T
        amp = buf[:, sc]
        t   = np.arange(len(amp))

        self.ax.clear()
        self.ax.set_facecolor(T["plot"]); self.fig.set_facecolor(T["surface"])
        self.ax.tick_params(colors=T["sub"], labelsize=8, which="both")
        for sp in self.ax.spines.values(): sp.set_edgecolor(T["border"])

        self.ax.fill_between(t, amp, alpha=0.13, color=T["bad"])
        self.ax.plot(t, amp, color=T["bad"], linewidth=1.1, alpha=0.95)

        self.ax.set_title(f"⬤ LIVE  Subcarrier #{sc}  ·  {len(amp)} packets", color=T["bad"], fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Time Step (Packets)", color=T["sub"], fontsize=9)
        self.ax.set_ylabel("EMA-Centered Magnitude", color=T["sub"], fontsize=9)
        self.fig.tight_layout(pad=1.8)
        self.pw.draw()

    def _plot(self, buf, sc, pred):
        T = self.T
        amp = buf[:, sc]
        t = np.arange(len(amp))
        peak = int(np.argmax(np.abs(amp))) 
        col = T["bad"] if pred == 1 else T["ok"]

        self.ax.clear()
        self.ax.set_facecolor(T["plot"]); self.fig.set_facecolor(T["surface"])
        self.ax.tick_params(colors=T["sub"], labelsize=8, which="both")
        for sp in self.ax.spines.values(): sp.set_edgecolor(T["border"])

        self.ax.fill_between(t, amp, alpha=0.13, color=T["accent"])
        self.ax.plot(t, amp, color=T["accent"], linewidth=1.1, alpha=0.95)
        self.ax.axvline(peak, color=col, linewidth=1.4, linestyle="--", alpha=0.6, label="Max Disruption")
        self.ax.scatter([peak], [amp[peak]], color=col, s=30, zorder=5)

        self.ax.set_title(f"Processed Subcarrier #{sc}  ·  {len(amp)} packets", color=T["text"], fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Time Step (Packets)", color=T["sub"], fontsize=9)
        self.ax.set_ylabel("EMA-Centered Magnitude", color=T["sub"], fontsize=9)
        self.ax.legend(fontsize=8, facecolor=T["surface"], edgecolor=T["border"], labelcolor=T["sub"])
        self.fig.tight_layout(pad=1.8)
        self.pw.draw()

    def _switch_theme(self):
        self.tn = "light" if self.tn == "dark" else "dark"
        self.T  = THEMES[self.tn]; T = self.T
        self.root.configure(bg=T["bg"])
        self.tgl.configure(text="☀" if self.tn == "dark" else "☾")

        for w, opt, key in self._rw:
            try: w.configure(**{opt: T[key]})
            except: pass

        self.sld.configure(bg=T["surface"], fg=T["sub"], troughcolor=T["surface2"], activebackground=T["accent"])
        self._draw_btn()
        self._draw_sort_btn()
        
        self.ax.set_facecolor(T["plot"])
        self.fig.set_facecolor(T["surface"])
        self.ax.tick_params(colors=T["sub"], labelsize=8)
        for sp in self.ax.spines.values(): sp.set_edgecolor(T["border"])
        self.ax.xaxis.label.set_color(T["sub"]); self.ax.yaxis.label.set_color(T["sub"]); self.ax.title.set_color(T["sub"])
        for lbl in self.ax.get_xticklabels() + self.ax.get_yticklabels(): lbl.set_color(T["sub"])
        leg = self.ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(T["surface"]); leg.get_frame().set_edgecolor(T["border"])
            for lt in leg.get_texts(): lt.set_color(T["sub"])
        self.pw.draw()

    def _show_error_popup(self, title, message):
        err_win = tk.Toplevel(self.root)
        err_win.title(title)
        err_win.geometry("550x250")
        err_win.attributes("-topmost", True) 
        lbl = tk.Label(err_win, text="The CNN-LSTM failed to load.", font=("Helvetica", 11, "bold"), fg="#f5505e")
        lbl.pack(pady=(10, 5))
        txt = tk.Text(err_win, wrap="word", height=8, font=("Consolas", 9))
        txt.pack(padx=10, pady=5, fill="both", expand=True)
        txt.insert("1.0", message)
        txt.configure(state="disabled") 
        btn_frame = tk.Frame(err_win)
        btn_frame.pack(fill="x", pady=10)
        
        def copy_err():
            self.root.clipboard_clear()
            self.root.clipboard_append(message)
            btn_copy.config(text="Copied! ✔️", fg="#26c97a")
            
        btn_copy = tk.Button(btn_frame, text="📋 Copy Error", command=copy_err, font=("Helvetica", 9, "bold"))
        btn_copy.pack(side="left", padx=20)
        btn_ok = tk.Button(btn_frame, text="OK", command=err_win.destroy, width=10, font=("Helvetica", 9))
        btn_ok.pack(side="right", padx=20)

    def _init_model(self):
        def _go():
            try:
                m = tf.keras.models.load_model(_MP, compile=False)
                self.scaler = joblib.load(_SCALER)
                m.predict(np.zeros((1, WIN, SC), dtype=np.float32), verbose=0)
                self.mdl = m
                print(f"  ResNet1D loaded — input shape: {m.input_shape}")

                if self.ser is None:
                    self.root.after(0, self.st_lbl.configure, {"text": "AI Ready  ·  Hardware Offline", "fg": self.T["warn"]})
                else:
                    self.root.after(0, self._dot, "ok")
                    self.root.after(0, self.st_lbl.configure, {"text": f"System Fully Active  ·  {_PORT}"})
            except Exception as e:
                self.root.after(0, self._dot, "err")
                self.root.after(0, self.st_lbl.configure, {"text": "Model Error (See Popup)"})
                err_msg = f"Model: {_MP}\nScaler: {_SCALER}\n\n{str(e)}"
                self.root.after(0, lambda: self._show_error_popup("AI Load Error", err_msg))

        self.scaler = None
        threading.Thread(target=_go, daemon=True).start()

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    root = tk.Tk()
    App(root)
    root.mainloop()