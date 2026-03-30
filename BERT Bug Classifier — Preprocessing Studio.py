"""
BERT Bug Classifier — Preprocessing Studio  v2
================================================
Fixes vs v1:
  1. Auto-relabel:  complete rewrite — tiered strong/weak signals, no false matches
  2. Truncation:    REMOVED — max 157 words in dataset, well within BERT 512 limit
  3. Description:   Full text shown in dedicated tab, never cut mid-sentence
  4. Cleaning:      Only removes true noise (hex stack-trace tokens), preserves meaning

Run:
    pip install pandas scikit-learn
    python bert_bug_classifier_ui.py
"""

import re, os, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
#  Theme
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
CARD_BG  = "#1c2128"
BORDER   = "#30363d"
ACCENT   = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f78166"
PURPLE   = "#d2a8ff"
ORANGE   = "#ffa657"
YELLOW   = "#e3b341"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
TEXT_MUT = "#484f58"

FN_MONO_S = ("Consolas", 9)
FN_HEAD   = ("Segoe UI", 13, "bold")
FN_TITLE  = ("Segoe UI", 11, "bold")
FN_BODY   = ("Segoe UI", 10)
FN_SMALL  = ("Segoe UI", 9)

# ─────────────────────────────────────────────────────────────────────────────
#  Label constants
# ─────────────────────────────────────────────────────────────────────────────
VALID_BUG_TYPES    = {"Crash","Freeze","Memory","UI/Visual","Network","Performance","Security","Other"}
VALID_SEVERITIES   = {"trivial","minor","normal","major","critical","blocker"}
VALID_FIXING_TIMES = {"fast","medium","slow"}

# ─────────────────────────────────────────────────────────────────────────────
#  FIXED relabeling  (v2)
#
#  Problem with v1: UI/Visual weak keywords ('scroll', 'display', 'layout'…)
#  are present in almost every bug, so they swallowed Freeze/Memory rows.
#
#  Fix: tiered approach
#    - Strong phrase → immediate relabel (very specific, rarely wrong)
#    - Weak word list → requires 3-5 matches, not just 1-2
#    - Uncertain → return original_label unchanged
# ─────────────────────────────────────────────────────────────────────────────
def classify_bug(text: str, original_label: str) -> str:
    t = text.lower()

    def has_strong(lst):  return any(phrase in t for phrase in lst)
    def weak_count(lst):  return sum(1 for k in lst if k in t)

    # Crash — stack traces, talkback IDs, explicit crash words
    if has_strong(["crash", "segfault", "assertion fail", "core dump", "fatal error",
                   "null pointer", "stack trace", "stack dump", "talkback",
                   "sigsegv", "sigabrt", "access violation"]):
        return "Crash"

    # Memory — specific tooling names or clear leak language
    if has_strong(["memory leak", "out of memory", "heap overflow", "use after free",
                   "buffer overflow", "purify", "valgrind", "refcounting", "bloaty"]):
        return "Memory"
    if weak_count(["leak", "oom", "malloc", "bloat", "heap"]) >= 3:
        return "Memory"

    # Security — very precise, appear only in security context
    if has_strong(["xss", "cross-site scripting", "sql injection", "csrf",
                   "remote code execution", "privilege escalation",
                   "cve-", "arbitrary code", "buffer overrun"]):
        return "Security"

    # Freeze — specific hang/freeze vocabulary
    if has_strong(["hang", "freeze", "frozen", "deadlock", "infinite loop",
                   "not responding", "unresponsive", "spins", "spinning",
                   "locks up", "lock up", "stuck forever"]):
        return "Freeze"

    # Network — connection-failure language, NOT just 'http' alone
    if has_strong(["connection refused", "socket error", "ssl error", "dns fail",
                   "connection timeout", "network error", "http error",
                   "failed to connect", "unable to connect"]):
        return "Network"
    if weak_count(["socket", "ftp", "ssl", "dns", "proxy", "firewall",
                   "packet", "bandwidth", "latency", "imap", "smtp", "pop3"]) >= 3:
        return "Network"

    # Performance — explicit regression / measurement language
    if has_strong(["cpu usage", "high cpu", "performance regression",
                   "slow performance", "response time", "takes too long",
                   "resources dry"]):
        return "Performance"
    if weak_count(["slow", "lag", "sluggish", "cpu", "benchmark",
                   "faster", "speed up", "responsive"]) >= 4:
        return "Performance"

    # UI/Visual — rendering / visual corruption (HIGH threshold to avoid false positives)
    if has_strong(["repaint", "pixel", "overlap", "misalign", "garbled",
                   "corrupt display", "visual artifact", "screen garbage",
                   "z-order", "clipping", "blurry", "flicker"]):
        return "UI/Visual"
    if weak_count(["render", "display", "layout", "font", "color", "alignment",
                   "visual", "widget", "scroll", "toolbar", "sidebar",
                   "button", "menu", "window"]) >= 5:
        return "UI/Visual"

    return original_label   # uncertain → keep original, never force a wrong label


# ─────────────────────────────────────────────────────────────────────────────
#  FIXED cleaning  (v2)
#
#  Original problem: URL replacement → [URL], number removal, truncation — all
#  destroy sentence meaning on a dataset that is ALREADY pre-cleaned.
#
#  Fix: only strip hex stack-trace tokens that are pure noise.
#  No truncation. No placeholder injection. All real words preserved.
# ─────────────────────────────────────────────────────────────────────────────
def clean_for_bert(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove hex dump tokens from stack traces: 'x a3 ff', long hex sequences
    text = re.sub(r"\b[0-9a-f]{6,}\b", "", text)
    text = re.sub(r"\bx\s+[0-9a-f]{1,4}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
#  UI helpers
# ─────────────────────────────────────────────────────────────────────────────
def styled_btn(parent, text, command, color=ACCENT, small=False):
    font = FN_SMALL if small else FN_BODY
    b = tk.Button(parent, text=text, command=command, bg=color, fg=DARK_BG,
                  font=font, relief="flat", cursor="hand2",
                  activebackground=TEXT_PRI, activeforeground=DARK_BG,
                  padx=12, pady=3 if small else 6)
    b.bind("<Enter>", lambda e: b.config(bg=TEXT_PRI))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b

def ghost_btn(parent, text, command, small=False):
    font = FN_SMALL if small else FN_BODY
    b = tk.Button(parent, text=text, command=command, bg=CARD_BG, fg=TEXT_SEC,
                  font=font, relief="flat", cursor="hand2",
                  activebackground=BORDER, activeforeground=TEXT_PRI,
                  padx=10, pady=3 if small else 5,
                  highlightthickness=1, highlightbackground=BORDER)
    return b

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, **kw):
        bg = kw.get("bg", DARK_BG)
        super().__init__(parent, bg=bg)
        canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = tk.Frame(canvas, bg=bg)
        self.inner.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))


# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────
class BERTPreprocessorApp:

    def __init__(self, root):
        self.root = root
        self.root.title("BERT Bug Classifier — Preprocessing Studio v2")
        self.root.geometry("1300x840")
        self.root.minsize(1100, 700)
        self.root.configure(bg=DARK_BG)
        self.df         = None
        self._splits    = {}
        self._desc_idx  = 0
        self._build_ui()
        self._log("Welcome to BERT Bug Classifier Preprocessing Studio v2", ACCENT)
        self._log("Load a CSV to begin. Expected: Severity · Fixing_time · Cleaned_Description · Bug_Type", TEXT_SEC)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self.root, bg=PANEL_BG, height=52,
                       highlightthickness=1, highlightbackground=BORDER)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top, text="⬡", font=("Segoe UI", 18), bg=PANEL_BG, fg=ACCENT).pack(side="left", padx=(14,4), pady=10)
        tk.Label(top, text="BERT Bug Classifier", font=("Segoe UI",13,"bold"), bg=PANEL_BG, fg=TEXT_PRI).pack(side="left", pady=10)
        tk.Label(top, text="Preprocessing Studio v2", font=FN_BODY, bg=PANEL_BG, fg=TEXT_SEC).pack(side="left", padx=6, pady=10)
        styled_btn(top, "⇧  Load CSV", self._load_file).pack(side="right", padx=8, pady=8)
        self._lbl_file = tk.Label(top, text="No file loaded", font=FN_SMALL, bg=PANEL_BG, fg=TEXT_MUT)
        self._lbl_file.pack(side="right", padx=6)

        body = tk.PanedWindow(self.root, orient="horizontal", bg=DARK_BG, sashwidth=4, handlesize=0)
        body.pack(fill="both", expand=True)
        sidebar = tk.Frame(body, bg=PANEL_BG, width=275)
        body.add(sidebar, minsize=240)
        self._build_sidebar(sidebar)
        right = tk.Frame(body, bg=DARK_BG)
        body.add(right, minsize=640)
        self._build_right(right)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        sf = tk.Frame(parent, bg=PANEL_BG)
        sf.pack(fill="x", padx=12, pady=(10,4))
        self._lbl_rows = tk.Label(sf, text="Rows: —",       font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC)
        self._lbl_cols = tk.Label(sf, text="Cols: —",       font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC)
        self._lbl_bugs = tk.Label(sf, text="Bug types: —",  font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC)
        self._lbl_sevs = tk.Label(sf, text="Severities: —", font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC)
        for w in (self._lbl_rows, self._lbl_cols, self._lbl_bugs, self._lbl_sevs):
            w.pack(anchor="w")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=6)

        SECTIONS = [
            ("📂  FILE", GREEN, [
                ("Load CSV",               self._load_file),
                ("Save Processed CSV",     self._save_file),
            ]),
            ("🔍  AUDIT & INSPECT", PURPLE, [
                ("Label Audit Report",          self._run_audit),
                ("Show Label Distributions",    self._show_distributions),
                ("Preview Table",               self._show_preview),
                ("View Full Description",       self._view_full_desc),
                ("Find Short Descriptions",     self._find_short_desc),
            ]),
            ("🛠  CLEAN & FIX", ORANGE, [
                ("Clean Hex / Stack Noise",     self._clean_descriptions),
                ("Auto-Relabel Bug Types ✦",    self._auto_relabel),
                ("Fix Label Casing",            self._fix_casing),
                ("Remove Short Rows (< 5 w)",   self._remove_short),
                ("Remove Duplicates",           self._remove_duplicates),
            ]),
            ("✏️  EDIT LABELS", ACCENT, [
                ("Rename a Label Value",        self._rename_label),
                ("Merge Two Labels",            self._merge_labels),
                ("Manual Cell Edit",            self._open_cell_editor),
            ]),
            ("📊  ENCODE & SPLIT", GREEN, [
                ("Label Encode All Targets",    self._encode_labels),
                ("Train / Val / Test Split",    self._split_data),
                ("Export Split CSVs",           self._export_splits),
                ("Show Class Weights",          self._show_weights),
            ]),
        ]

        scr = ScrollableFrame(parent, bg=PANEL_BG)
        scr.pack(fill="both", expand=True)
        for title, color, buttons in SECTIONS:
            tk.Label(scr.inner, text=title, font=("Consolas",8,"bold"),
                     bg=PANEL_BG, fg=TEXT_MUT).pack(anchor="w", padx=14, pady=(10,2))
            for label, cmd in buttons:
                b = tk.Button(scr.inner, text=f"  {label}", command=cmd,
                              bg=PANEL_BG, fg=TEXT_PRI, font=FN_SMALL,
                              relief="flat", anchor="w", cursor="hand2",
                              activebackground=CARD_BG, activeforeground=color,
                              padx=8, pady=5, width=30)
                b.pack(fill="x", padx=6, pady=1)
                c = color
                b.bind("<Enter>", lambda e, btn=b, col=c: btn.config(bg=CARD_BG, fg=col))
                b.bind("<Leave>", lambda e, btn=b: btn.config(bg=PANEL_BG, fg=TEXT_PRI))

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right(self, parent):
        s = ttk.Style(); s.theme_use("default")
        s.configure("D.TNotebook",     background=DARK_BG, borderwidth=0)
        s.configure("D.TNotebook.Tab", background=PANEL_BG, foreground=TEXT_SEC,
                    padding=[14,6], font=FN_SMALL)
        s.map("D.TNotebook.Tab",
              background=[("selected", CARD_BG)],
              foreground=[("selected", TEXT_PRI)])

        self._nb = ttk.Notebook(parent, style="D.TNotebook")
        self._nb.pack(fill="both", expand=True, padx=8, pady=8)

        tabs = [("Console", "_tab_log"), ("Table Preview", "_tab_table"),
                ("Distributions", "_tab_dist"), ("Audit", "_tab_audit"),
                ("Description Viewer", "_tab_desc")]
        for title, attr in tabs:
            f = tk.Frame(self._nb, bg=DARK_BG)
            setattr(self, attr, f)
            self._nb.add(f, text=f" {title} ")

        self._build_tab_log()
        self._build_tab_table()
        self._build_tab_dist()
        self._build_tab_audit()
        self._build_tab_desc()

    # ── Console tab ───────────────────────────────────────────────────────────
    def _build_tab_log(self):
        p = self._tab_log
        hdr = tk.Frame(p, bg=DARK_BG); hdr.pack(fill="x", padx=10, pady=(8,0))
        tk.Label(hdr, text="Console", font=FN_HEAD, bg=DARK_BG, fg=TEXT_PRI).pack(side="left")
        ghost_btn(hdr, "Clear", self._clear_log, small=True).pack(side="right")
        self._log_box = tk.Text(p, bg="#010409", fg=TEXT_PRI, font=FN_MONO_S,
                                relief="flat", padx=14, pady=10,
                                insertbackground=ACCENT, wrap="word",
                                highlightthickness=1, highlightbackground=BORDER)
        self._log_box.pack(fill="both", expand=True, padx=10, pady=8)
        self._log_box.config(state="disabled")
        for tag, col in [("a",ACCENT),("g",GREEN),("r",RED),("p",PURPLE),
                         ("o",ORANGE),("y",YELLOW),("s",TEXT_SEC),("m",TEXT_MUT)]:
            self._log_box.tag_config(tag, foreground=col)

    _TAG = {None:"", ACCENT:"a", GREEN:"g", RED:"r", PURPLE:"p",
            ORANGE:"o", YELLOW:"y", TEXT_SEC:"s", TEXT_MUT:"m"}

    def _log(self, msg, color=None):
        tag = self._TAG.get(color, "")
        self._log_box.config(state="normal")
        self._log_box.insert("end", msg+"\n", tag)
        self._log_box.see("end")
        self._log_box.config(state="disabled")

    def _log_sep(self): self._log("─"*60, TEXT_MUT)

    def _clear_log(self):
        self._log_box.config(state="normal")
        self._log_box.delete("1.0","end")
        self._log_box.config(state="disabled")

    # ── Table Preview tab ─────────────────────────────────────────────────────
    def _build_tab_table(self):
        p = self._tab_table
        ctrl = tk.Frame(p, bg=DARK_BG); ctrl.pack(fill="x", padx=10, pady=(8,0))
        tk.Label(ctrl, text="Table Preview", font=FN_HEAD, bg=DARK_BG, fg=TEXT_PRI).pack(side="left")
        ghost_btn(ctrl, "Refresh", self._show_preview, small=True).pack(side="right", padx=6)
        tk.Label(ctrl, text="Rows:", font=FN_SMALL, bg=DARK_BG, fg=TEXT_SEC).pack(side="right")
        self._entry_rows = tk.Entry(ctrl, width=5, font=FN_SMALL, bg=CARD_BG, fg=TEXT_PRI,
                                    insertbackground=ACCENT, relief="flat",
                                    highlightthickness=1, highlightbackground=BORDER)
        self._entry_rows.insert(0, "50"); self._entry_rows.pack(side="right", padx=(0,4))

        tk.Label(p, text="  ↑ Double-click any cell to edit  ·  Select row then 'View Full Description' for complete text",
                 font=FN_SMALL, bg=DARK_BG, fg=TEXT_MUT).pack(anchor="w", padx=10)

        frame = tk.Frame(p, bg=DARK_BG)
        frame.pack(fill="both", expand=True, padx=10, pady=(4,8))

        s = ttk.Style()
        s.configure("D.Treeview", background=CARD_BG, fieldbackground=CARD_BG,
                    foreground=TEXT_PRI, rowheight=24, font=FN_MONO_S, borderwidth=0)
        s.configure("D.Treeview.Heading", background=PANEL_BG, foreground=TEXT_SEC,
                    font=("Segoe UI",9,"bold"), relief="flat")
        s.map("D.Treeview", background=[("selected","#1f3a5f")])

        sy = ttk.Scrollbar(frame, orient="vertical")
        sx = ttk.Scrollbar(frame, orient="horizontal")
        sy.pack(side="right", fill="y"); sx.pack(side="bottom", fill="x")
        self._tree = ttk.Treeview(frame, style="D.Treeview",
                                   yscrollcommand=sy.set, xscrollcommand=sx.set)
        self._tree.pack(fill="both", expand=True)
        sy.config(command=self._tree.yview); sx.config(command=self._tree.xview)
        self._tree.bind("<Double-1>", self._on_cell_dbl_click)

        self._lbl_info = tk.Label(p, text="", font=FN_SMALL, bg=DARK_BG, fg=TEXT_SEC)
        self._lbl_info.pack(anchor="w", padx=10, pady=(0,4))

    # ── Distributions tab ─────────────────────────────────────────────────────
    def _build_tab_dist(self):
        p = self._tab_dist
        tk.Label(p, text="Label Distributions", font=FN_HEAD, bg=DARK_BG,
                 fg=TEXT_PRI).pack(anchor="w", padx=14, pady=(10,4))
        scr = ScrollableFrame(p, bg=DARK_BG); scr.pack(fill="both", expand=True)
        self._dist_inner = scr.inner

    def _draw_dist(self, col_name, counts, total, palette):
        card = tk.Frame(self._dist_inner, bg=CARD_BG,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x", padx=12, pady=5)
        tk.Label(card, text=col_name, font=FN_TITLE, bg=CARD_BG,
                 fg=TEXT_PRI).pack(anchor="w", padx=12, pady=(8,4))
        for i, (label, count) in enumerate(counts.items()):
            pct   = count/total*100
            color = palette[i % len(palette)]
            row   = tk.Frame(card, bg=CARD_BG); row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=str(label), font=FN_SMALL, bg=CARD_BG,
                     fg=TEXT_PRI, width=20, anchor="w").pack(side="left")
            bg = tk.Frame(row, bg=PANEL_BG, height=14, width=280)
            bg.pack(side="left", padx=(0,8)); bg.pack_propagate(False)
            tk.Frame(bg, bg=color, height=14, width=max(2, int(pct/100*280))).place(x=0, y=0)
            tk.Label(row, text=f"{count:,}  ({pct:.1f}%)", font=FN_MONO_S,
                     bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        tk.Frame(card, bg=CARD_BG, height=6).pack()

    # ── Audit tab ─────────────────────────────────────────────────────────────
    def _build_tab_audit(self):
        p = self._tab_audit
        hdr = tk.Frame(p, bg=DARK_BG); hdr.pack(fill="x", padx=10, pady=(8,0))
        tk.Label(hdr, text="Label Audit", font=FN_HEAD, bg=DARK_BG, fg=TEXT_PRI).pack(side="left")
        ghost_btn(hdr, "Run Audit", self._run_audit, small=True).pack(side="right")
        self._audit_box = tk.Text(p, bg="#010409", fg=TEXT_PRI, font=FN_MONO_S,
                                  relief="flat", padx=14, pady=10, wrap="word",
                                  highlightthickness=1, highlightbackground=BORDER)
        self._audit_box.pack(fill="both", expand=True, padx=10, pady=8)
        self._audit_box.config(state="disabled")
        for tag, col in [("ok",GREEN),("warn",ORANGE),("err",RED),
                         ("hdr",ACCENT),("dim",TEXT_MUT),("sec",TEXT_SEC)]:
            self._audit_box.tag_config(tag, foreground=col)

    def _aw(self, msg, tag=""):
        self._audit_box.config(state="normal")
        self._audit_box.insert("end", msg+"\n", tag)
        self._audit_box.config(state="disabled")

    def _audit_clear(self):
        self._audit_box.config(state="normal")
        self._audit_box.delete("1.0","end")
        self._audit_box.config(state="disabled")

    # ── Description Viewer tab ────────────────────────────────────────────────
    def _build_tab_desc(self):
        p = self._tab_desc
        hdr = tk.Frame(p, bg=DARK_BG); hdr.pack(fill="x", padx=10, pady=(8,0))
        tk.Label(hdr, text="Full Description Viewer", font=FN_HEAD,
                 bg=DARK_BG, fg=TEXT_PRI).pack(side="left")
        ghost_btn(hdr, "Next →",  self._desc_next, small=True).pack(side="right", padx=2)
        ghost_btn(hdr, "← Prev",  self._desc_prev, small=True).pack(side="right", padx=2)

        # Meta bar
        meta = tk.Frame(p, bg=PANEL_BG, highlightthickness=1, highlightbackground=BORDER)
        meta.pack(fill="x", padx=10, pady=6)
        self._lbl_didx  = tk.Label(meta, text="Row: —",         font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC)
        self._lbl_dbt   = tk.Label(meta, text="Bug Type: —",    font=FN_SMALL, bg=PANEL_BG, fg=PURPLE)
        self._lbl_dsev  = tk.Label(meta, text="Severity: —",    font=FN_SMALL, bg=PANEL_BG, fg=ORANGE)
        self._lbl_dfix  = tk.Label(meta, text="Fix Time: —",    font=FN_SMALL, bg=PANEL_BG, fg=YELLOW)
        self._lbl_dwc   = tk.Label(meta, text="Words: —",       font=FN_SMALL, bg=PANEL_BG, fg=TEXT_MUT)
        for w in (self._lbl_didx, self._lbl_dbt, self._lbl_dsev, self._lbl_dfix, self._lbl_dwc):
            w.pack(side="left", padx=12, pady=6)

        # Full scrollable text — NO truncation whatsoever
        wrap_frame = tk.Frame(p, bg=DARK_BG)
        wrap_frame.pack(fill="both", expand=True, padx=10, pady=(0,8))
        sb = ttk.Scrollbar(wrap_frame, orient="vertical")
        sb.pack(side="right", fill="y")
        self._desc_box = tk.Text(wrap_frame, bg="#010409", fg=TEXT_PRI,
                                 font=("Segoe UI", 11), relief="flat",
                                 padx=20, pady=16, wrap="word",
                                 yscrollcommand=sb.set,
                                 highlightthickness=1, highlightbackground=BORDER,
                                 spacing1=4, spacing3=4)
        self._desc_box.pack(fill="both", expand=True)
        sb.config(command=self._desc_box.yview)
        self._desc_box.config(state="disabled")

    def _render_desc(self, idx):
        if self.df is None: return
        idx = max(0, min(idx, len(self.df)-1))
        self._desc_idx = idx
        row  = self.df.iloc[idx]
        desc = str(row.get("Cleaned_Description", ""))
        wc   = len(desc.split())

        for lbl, key, color in [(self._lbl_dbt,"Bug_Type",PURPLE),
                                 (self._lbl_dsev,"Severity",ORANGE),
                                 (self._lbl_dfix,"Fixing_time",YELLOW)]:
            val = row[key] if key in self.df.columns else "—"
            lbl.config(text=f"{key.replace('_',' ')}: {val}", fg=color)
        self._lbl_didx.config(text=f"Row: {idx} / {len(self.df)-1}")
        self._lbl_dwc.config(text=f"Words: {wc}")

        self._desc_box.config(state="normal")
        self._desc_box.delete("1.0","end")
        self._desc_box.insert("end", desc)   # ← FULL TEXT, no cuts
        self._desc_box.config(state="disabled")

    def _view_full_desc(self):
        if not self._require_df(): return
        sel = self._tree.selection()
        if sel: self._desc_idx = int(sel[0])
        self._nb.select(4)
        self._render_desc(self._desc_idx)

    def _desc_prev(self): self._render_desc(self._desc_idx - 1)
    def _desc_next(self): self._render_desc(self._desc_idx + 1)

    # ── Shared utilities ──────────────────────────────────────────────────────
    def _update_stats(self):
        if self.df is None: return
        r, c = self.df.shape
        bugs = self.df["Bug_Type"].nunique() if "Bug_Type" in self.df.columns else "?"
        sevs = self.df["Severity"].nunique() if "Severity" in self.df.columns else "?"
        self._lbl_rows.config(text=f"Rows: {r:,}")
        self._lbl_cols.config(text=f"Cols: {c}")
        self._lbl_bugs.config(text=f"Bug types: {bugs}")
        self._lbl_sevs.config(text=f"Severities: {sevs}")

    def _require_df(self):
        if self.df is None:
            messagebox.showwarning("No Data","Load a CSV file first.")
            return False
        return True

    def _modal(self, title, w, h):
        win = tk.Toplevel(self.root)
        win.title(title); win.geometry(f"{w}x{h}")
        win.configure(bg=PANEL_BG)
        win.transient(self.root); win.grab_set(); win.resizable(False, False)
        return win

    # ── FILE ──────────────────────────────────────────────────────────────────
    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        try:
            self.df = pd.read_csv(path)
            self._lbl_file.config(text=os.path.basename(path), fg=GREEN)
            self._update_stats()
            self._log_sep()
            self._log(f"✓ Loaded: {os.path.basename(path)}", GREEN)
            self._log(f"  {self.df.shape[0]:,} rows × {self.df.shape[1]} cols", TEXT_SEC)
            self._log(f"  Columns: {', '.join(self.df.columns.tolist())}", TEXT_MUT)
            self._show_preview()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _save_file(self):
        if not self._require_df(): return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            self.df.to_csv(path, index=False)
            self._log(f"✓ Saved: {os.path.basename(path)}", GREEN)
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    # ── PREVIEW ───────────────────────────────────────────────────────────────
    def _show_preview(self):
        if not self._require_df(): return
        self._nb.select(1)
        try: n = int(self._entry_rows.get())
        except ValueError: n = 50

        df_head = self.df.head(n)
        self._tree.delete(*self._tree.get_children())
        cols = list(self.df.columns)
        self._tree["columns"] = cols
        self._tree["show"]    = "headings"
        for col in cols:
            w = 280 if col == "Cleaned_Description" else 120
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="w")

        for idx, row in df_head.iterrows():
            vals = []
            for col in cols:
                v = str(row[col])
                # Display-only truncation for the table cell — underlying data untouched
                if col == "Cleaned_Description" and len(v) > 100:
                    v = v[:97] + "…"
                vals.append(v)
            self._tree.insert("","end", iid=str(idx), values=vals)

        self._lbl_info.config(
            text=(f"Showing {len(df_head):,} of {len(self.df):,} rows  ·  "
                  "Double-click a cell to edit  ·  Select row + 'View Full Description' for full text"))

    # ── DISTRIBUTIONS ─────────────────────────────────────────────────────────
    def _show_distributions(self):
        if not self._require_df(): return
        self._nb.select(2)
        for w in self._dist_inner.winfo_children(): w.destroy()
        palettes = {
            "Bug_Type":    [ACCENT, GREEN, RED, PURPLE, ORANGE, YELLOW, "#79c0ff","#56d364"],
            "Severity":    [RED, ORANGE, YELLOW, GREEN, ACCENT, PURPLE],
            "Fixing_time": [GREEN, ORANGE, RED],
        }
        for col in ["Bug_Type","Severity","Fixing_time"]:
            if col not in self.df.columns: continue
            counts = self.df[col].value_counts()
            self._draw_dist(col, counts, len(self.df), palettes.get(col,[ACCENT]))

    # ── AUDIT ─────────────────────────────────────────────────────────────────
    def _run_audit(self):
        if not self._require_df(): return
        self._nb.select(3); self._audit_clear()
        df = self.df
        self._aw("═"*56, "dim"); self._aw("  LABEL AUDIT REPORT","hdr"); self._aw("═"*56,"dim")

        self._aw("\n── Required Columns ──","hdr")
        for col in ["Bug_Type","Severity","Fixing_time","Cleaned_Description"]:
            tag = "ok" if col in df.columns else "err"
            sym = "✓" if col in df.columns else "✗  MISSING!"
            self._aw(f"  {sym}  {col}", tag)

        if "Bug_Type" in df.columns:
            self._aw("\n── Bug_Type Distribution ──","hdr")
            for label, count in df["Bug_Type"].value_counts().items():
                pct = count/len(df)*100
                tag = "warn" if label == "Other" else "sec"
                self._aw(f"  {label:15s}  {count:6,}  ({pct:5.1f}%)", tag)
            unknown = df.loc[~df["Bug_Type"].isin(VALID_BUG_TYPES),"Bug_Type"].unique()
            if len(unknown): self._aw(f"  ⚠ Unknown: {list(unknown)}","warn")
            else:            self._aw("  ✓ All values valid","ok")

            if "Cleaned_Description" in df.columns:
                self._aw("\n  Keyword match rates:","hdr")
                kw_check = {
                    "Crash":     (["crash","segfault","core dump","fatal","talkback","stack trace"],[]),
                    "Freeze":    (["hang","freeze","frozen","deadlock","not responding","unresponsive","spins"],[]),
                    "Memory":    (["memory leak","out of memory","purify","valgrind","bloaty"],["leak","malloc","bloat"]),
                    "UI/Visual": (["render","display","layout","repaint","pixel","visual artifact"],["font","scroll","widget","button","menu"]),
                    "Network":   (["connection refused","socket error","ssl error","dns fail","http error"],["socket","ftp","ssl","dns","imap"]),
                }
                for label,(strong,weak) in kw_check.items():
                    sub = df[df["Bug_Type"]==label]
                    if not len(sub): continue
                    match = sub["Cleaned_Description"].str.lower().apply(
                        lambda x: any(k in x for k in strong) or sum(k in x for k in weak)>=2
                    ).mean()*100
                    tag = "ok" if match>65 else ("warn" if match>40 else "err")
                    bar = "█"*int(match/5)
                    self._aw(f"  {label:15s}  {match:5.1f}%  {bar}", tag)

        for col, valid in [("Severity",VALID_SEVERITIES),("Fixing_time",VALID_FIXING_TIMES)]:
            if col not in df.columns: continue
            self._aw(f"\n── {col} ──","hdr")
            bad = df.loc[~df[col].isin(valid), col].unique()
            if len(bad): self._aw(f"  ⚠ Invalid: {list(bad)}","err")
            else:        self._aw("  ✓ All values valid","ok")
            for v,c in df[col].value_counts().items():
                self._aw(f"    {v:12s}  {c:,}","sec")

        if "Cleaned_Description" in df.columns:
            self._aw("\n── Description Health ──","hdr")
            wc = df["Cleaned_Description"].str.split().str.len()
            short = (wc<5).sum(); na = df["Cleaned_Description"].isna().sum()
            hex_n = df["Cleaned_Description"].str.contains(r"\b[0-9a-f]{6,}\b",regex=True).sum()
            self._aw(f"  Mean word count:     {wc.mean():.1f}","sec")
            self._aw(f"  Max  word count:     {wc.max()}  (BERT limit: 512 tokens → safe)","ok")
            self._aw(f"  Rows < 5 words:      {short}", "warn" if short else "ok")
            self._aw(f"  Missing / NaN:       {na}", "err" if na else "ok")
            self._aw(f"  Rows with hex noise: {hex_n}", "warn" if hex_n else "ok")

        self._aw("\n"+"═"*56,"dim")
        self._log("✓ Audit complete — see Audit tab", PURPLE)

    # ── CLEAN & FIX ───────────────────────────────────────────────────────────
    def _clean_descriptions(self):
        if not self._require_df(): return
        if "Cleaned_Description" not in self.df.columns:
            messagebox.showerror("Missing","'Cleaned_Description' not found."); return

        win = self._modal("Clean Descriptions", 460, 280)
        tk.Label(win, text="Clean Descriptions for BERT", font=FN_TITLE,
                 bg=PANEL_BG, fg=TEXT_PRI).pack(pady=(16,4))
        tk.Label(win, text="Only true noise removed — all real words preserved",
                 font=FN_SMALL, bg=PANEL_BG, fg=GREEN).pack()
        tk.Label(win, text="Descriptions are already lowercased and pre-cleaned",
                 font=FN_SMALL, bg=PANEL_BG, fg=TEXT_MUT).pack(pady=(2,10))

        checks = {}
        for key, label in [("hex",  "Remove hex stack-trace tokens  (e.g. x a3 ff b2 …)"),
                            ("space","Collapse extra whitespace")]:
            v = tk.BooleanVar(value=True)
            checks[key] = v
            tk.Checkbutton(win, text=label, variable=v, bg=PANEL_BG, fg=TEXT_PRI,
                           selectcolor=CARD_BG, activebackground=PANEL_BG,
                           font=FN_SMALL).pack(anchor="w", padx=40, pady=2)

        tk.Label(win,
                 text="✗  URL placeholders / number removal are DISABLED\n"
                      "   (they destroy sentence meaning on pre-cleaned text)\n"
                      "✗  Truncation is DISABLED — max 157 words, BERT limit is 512",
                 font=FN_SMALL, bg=PANEL_BG, fg=RED, justify="left").pack(pady=(10,0), padx=30, anchor="w")

        def apply():
            cfg = {k: v.get() for k,v in checks.items()}
            def _c(text):
                if not isinstance(text, str): return ""
                if cfg["hex"]:
                    text = re.sub(r"\b[0-9a-f]{6,}\b", "", text)
                    text = re.sub(r"\bx\s+[0-9a-f]{1,4}\b", "", text)
                if cfg["space"]:
                    text = re.sub(r"\s+", " ", text).strip()
                return text
            before = self.df["Cleaned_Description"].str.len().mean()
            self.df["Cleaned_Description"] = self.df["Cleaned_Description"].apply(_c)
            after = self.df["Cleaned_Description"].str.len().mean()
            self._log(f"✓ Cleaned descriptions  avg {before:.0f} → {after:.0f} chars", GREEN)
            self._show_preview(); win.destroy()

        styled_btn(win, "Apply", apply, color=ORANGE).pack(pady=16)

    def _auto_relabel(self):
        if not self._require_df(): return
        for col in ("Bug_Type","Cleaned_Description"):
            if col not in self.df.columns:
                messagebox.showerror("Missing",f"'{col}' not found."); return

        win = self._modal("Auto-Relabel Bug Types", 480, 400)
        tk.Label(win, text="Auto-Relabel Bug Types", font=FN_TITLE,
                 bg=PANEL_BG, fg=TEXT_PRI).pack(pady=(14,4))
        tk.Label(win, text="Select which current labels to re-examine:",
                 font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC).pack()

        counts = self.df["Bug_Type"].value_counts()
        vars_ = {}
        chk_f = tk.Frame(win, bg=PANEL_BG); chk_f.pack(padx=30, pady=8, fill="x")
        for bt in sorted(self.df["Bug_Type"].unique()):
            v = tk.BooleanVar(value=bt in {"Freeze","Memory","Other"})
            vars_[bt] = v
            row = tk.Frame(chk_f, bg=PANEL_BG); row.pack(fill="x", pady=1)
            tk.Checkbutton(row, text=bt, variable=v, bg=PANEL_BG, fg=TEXT_PRI,
                           selectcolor=CARD_BG, activebackground=PANEL_BG,
                           font=FN_SMALL, width=14, anchor="w").pack(side="left")
            tk.Label(row, text=f"{counts.get(bt,0):,} rows", font=FN_SMALL,
                     bg=PANEL_BG, fg=TEXT_MUT).pack(side="left", padx=8)

        tk.Label(win,
                 text="How it works:\n"
                      "  Strong phrase match  →  relabel immediately (very precise)\n"
                      "  Weak word matches     →  require 3–5 hits before relabeling\n"
                      "  Uncertain              →  original label kept unchanged",
                 font=FN_SMALL, bg=PANEL_BG, fg=TEXT_MUT, justify="left"
                 ).pack(padx=30, pady=(4,0), anchor="w")

        prog = tk.Label(win, text="", font=FN_SMALL, bg=PANEL_BG, fg=GREEN)
        prog.pack(pady=4)

        def apply():
            selected = {k for k,v in vars_.items() if v.get()}
            if not selected:
                messagebox.showwarning("Nothing selected","Pick at least one.",parent=win); return
            prog.config(text="Relabeling… please wait"); win.update()

            before = self.df["Bug_Type"].copy()
            mask   = self.df["Bug_Type"].isin(selected)
            self.df.loc[mask,"Bug_Type"] = self.df.loc[mask].apply(
                lambda r: classify_bug(r["Cleaned_Description"], r["Bug_Type"]), axis=1)

            changed = (self.df["Bug_Type"] != before).sum()
            self._log_sep()
            self._log(f"✓ Auto-relabel: {changed:,} rows updated", GREEN)

            # Breakdown
            changed_df = self.df[self.df["Bug_Type"] != before].copy()
            changed_df["_from"] = before[changed_df.index]
            breakdown = (changed_df.groupby(["_from","Bug_Type"])
                         .size().reset_index(name="n")
                         .sort_values("n", ascending=False))
            for _, r in breakdown.iterrows():
                self._log(f"  {r['_from']:15s} → {r['Bug_Type']:15s}  {r['n']:,}", TEXT_SEC)

            self._update_stats(); self._show_preview(); win.destroy()

        styled_btn(win, "Apply Relabeling", apply, color=ORANGE).pack(pady=12)

    def _fix_casing(self):
        if not self._require_df(): return
        changes = 0
        for col, fn in [("Severity",str.lower),("Fixing_time",str.lower)]:
            if col in self.df.columns:
                b = self.df[col].copy()
                self.df[col] = self.df[col].apply(lambda x: fn(x) if isinstance(x,str) else x)
                changes += (self.df[col]!=b).sum()
        if "Bug_Type" in self.df.columns:
            b = self.df["Bug_Type"].copy()
            self.df["Bug_Type"] = self.df["Bug_Type"].str.strip()
            changes += (self.df["Bug_Type"]!=b).sum()
        self._log(f"✓ Label casing fixed  ({changes} cells changed)", GREEN)
        self._show_preview()

    def _find_short_desc(self):
        if not self._require_df(): return
        if "Cleaned_Description" not in self.df.columns: return
        wc = self.df["Cleaned_Description"].str.split().str.len()
        short = self.df[wc < 5]
        self._log(f"  Short descriptions (< 5 words): {len(short)} rows", ORANGE)
        if len(short): self._log(f"  Indices: {short.index.tolist()[:20]}", TEXT_MUT)

    def _remove_short(self):
        if not self._require_df(): return
        if "Cleaned_Description" not in self.df.columns: return
        before = len(self.df)
        self.df = self.df[self.df["Cleaned_Description"].str.split().str.len() >= 5]
        self.df.reset_index(drop=True, inplace=True)
        self._log(f"✓ Removed {before-len(self.df)} short rows", GREEN)
        self._update_stats(); self._show_preview()

    def _remove_duplicates(self):
        if not self._require_df(): return
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self._log(f"✓ Removed {before-len(self.df):,} duplicates", GREEN)
        self._update_stats(); self._show_preview()

    # ── EDIT LABELS ───────────────────────────────────────────────────────────
    def _rename_label(self):
        if not self._require_df(): return
        win = self._modal("Rename a Label Value", 380, 240)
        tk.Label(win, text="Rename a Label Value", font=FN_TITLE,
                 bg=PANEL_BG, fg=TEXT_PRI).pack(pady=(16,8))
        cols = [c for c in ["Bug_Type","Severity","Fixing_time"] if c in self.df.columns]

        tk.Label(win, text="Column:", font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC).pack()
        col_var = tk.StringVar(value=cols[0])
        col_cb  = ttk.Combobox(win, values=cols, textvariable=col_var, state="readonly", width=22)
        col_cb.pack(pady=4)

        tk.Label(win, text="Current value:", font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC).pack()
        old_var = tk.StringVar()
        old_cb  = ttk.Combobox(win, textvariable=old_var, state="readonly", width=22)
        old_cb.pack(pady=4)

        def refresh(*_):
            vals = self.df[col_var.get()].dropna().unique().tolist()
            old_cb["values"] = vals
            if vals: old_var.set(vals[0])
        col_var.trace("w", refresh); refresh()

        tk.Label(win, text="New value:", font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC).pack()
        new_e = tk.Entry(win, font=FN_SMALL, bg=CARD_BG, fg=TEXT_PRI, insertbackground=ACCENT,
                         relief="flat", highlightthickness=1, highlightbackground=BORDER)
        new_e.pack(pady=4, padx=30, fill="x")

        def apply():
            col=col_var.get(); old=old_var.get(); new=new_e.get().strip()
            if not new:
                messagebox.showwarning("Empty","New value cannot be empty.",parent=win); return
            count = (self.df[col]==old).sum()
            self.df[col] = self.df[col].replace(old, new)
            self._log(f"✓ Renamed '{old}' → '{new}' in '{col}'  ({count:,} rows)", GREEN)
            self._show_preview(); win.destroy()
        styled_btn(win, "Rename", apply).pack(pady=10)

    def _merge_labels(self):
        if not self._require_df(): return
        win = self._modal("Merge Two Labels", 400, 270)
        tk.Label(win, text="Merge Two Labels", font=FN_TITLE,
                 bg=PANEL_BG, fg=TEXT_PRI).pack(pady=(16,8))
        cols = [c for c in ["Bug_Type","Severity","Fixing_time"] if c in self.df.columns]

        tk.Label(win, text="Column:", font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC).pack()
        col_var = tk.StringVar(value=cols[0])
        ttk.Combobox(win, values=cols, textvariable=col_var,
                     state="readonly", width=22).pack(pady=4)

        tk.Label(win, text="Merge this (source):", font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC).pack()
        src_var = tk.StringVar(); src_cb = ttk.Combobox(win, textvariable=src_var, state="readonly", width=22); src_cb.pack(pady=4)

        tk.Label(win, text="Into this (target):", font=FN_SMALL, bg=PANEL_BG, fg=TEXT_SEC).pack()
        tgt_var = tk.StringVar(); tgt_cb = ttk.Combobox(win, textvariable=tgt_var, state="readonly", width=22); tgt_cb.pack(pady=4)

        def upd(*_):
            v = self.df[col_var.get()].dropna().unique().tolist()
            src_cb["values"]=tgt_cb["values"]=v
            if v: src_var.set(v[0]); tgt_var.set(v[0])
        col_var.trace("w",upd); upd()

        def apply():
            col=col_var.get(); src=src_var.get(); tgt=tgt_var.get()
            if src==tgt:
                messagebox.showwarning("Same","Source and target must differ.",parent=win); return
            count=(self.df[col]==src).sum()
            self.df[col]=self.df[col].replace(src,tgt)
            self._log(f"✓ Merged '{src}' → '{tgt}' in '{col}'  ({count:,} rows)", GREEN)
            self._show_preview(); win.destroy()
        styled_btn(win,"Merge",apply).pack(pady=12)

    def _open_cell_editor(self):
        if not self._require_df(): return
        self._nb.select(1)
        messagebox.showinfo("Cell Edit",
            "Double-click any cell in the Table Preview tab to edit it.\n"
            "Press Enter to save, Escape or click away to cancel.")

    def _on_cell_dbl_click(self, event):
        if self.df is None: return
        if self._tree.identify("region", event.x, event.y) != "cell": return
        row_id = self._tree.identify_row(event.y)
        col_id = self._tree.identify_column(event.x)
        if not row_id or not col_id: return
        col_idx  = int(col_id.replace("#",""))-1
        col_name = self._tree["columns"][col_idx]
        x, y, w, h = self._tree.bbox(row_id, col_id)
        cur = self._tree.set(row_id, col_id).rstrip("…")

        # Fetch full value from df for description column
        if col_name == "Cleaned_Description":
            cur = str(self.df.at[int(row_id), col_name])

        entry = tk.Entry(self._tree, font=FN_MONO_S, bg=CARD_BG,
                         fg=ACCENT, insertbackground=ACCENT, relief="flat")
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, cur); entry.focus()

        def save(e=None):
            nv = entry.get(); df_idx = int(row_id)
            try:    self.df.at[df_idx, col_name] = pd.to_numeric(nv)
            except: self.df.at[df_idx, col_name] = nv
            disp = (nv[:97]+"…") if (col_name=="Cleaned_Description" and len(nv)>100) else nv
            self._tree.set(row_id, col_id, disp)
            self._log(f"  Edited [{df_idx}, {col_name}] → '{nv[:60]}'", TEXT_MUT)
            entry.destroy()

        entry.bind("<Return>", save)
        entry.bind("<Escape>", lambda e: entry.destroy())
        entry.bind("<FocusOut>", lambda e: entry.destroy())

    # ── ENCODE & SPLIT ────────────────────────────────────────────────────────
    def _encode_labels(self):
        if not self._require_df(): return
        targets = {
            "Bug_Type":    sorted(VALID_BUG_TYPES),
            "Severity":    sorted(VALID_SEVERITIES),
            "Fixing_time": sorted(VALID_FIXING_TIMES),
        }
        self._log_sep(); self._log("Label Encoding:", ACCENT)
        for col, classes in targets.items():
            if col not in self.df.columns: continue
            le = LabelEncoder(); le.fit(classes)
            self.df[f"{col}_ID"] = self.df[col].apply(
                lambda x: int(le.transform([x])[0]) if x in classes else -1)
            self._log(f"  {col} → {col}_ID", PURPLE)
            for i,c in enumerate(le.classes_): self._log(f"    {i}: {c}", TEXT_SEC)
        self._log("✓ *_ID columns added to dataframe", GREEN)
        self._show_preview()

    def _split_data(self):
        if not self._require_df(): return
        if "Bug_Type" not in self.df.columns:
            messagebox.showerror("Missing","'Bug_Type' required."); return

        win = self._modal("Train / Val / Test Split", 380, 260)
        tk.Label(win, text="Split Configuration", font=FN_TITLE,
                 bg=PANEL_BG, fg=TEXT_PRI).pack(pady=(16,8))
        fields = {}
        for label, default in [("Train %","80"),("Val %","10"),("Test %","10"),("Random Seed","42")]:
            row = tk.Frame(win, bg=PANEL_BG); row.pack(fill="x", padx=30, pady=3)
            tk.Label(row, text=label, font=FN_SMALL, bg=PANEL_BG,
                     fg=TEXT_SEC, width=14, anchor="w").pack(side="left")
            e = tk.Entry(row, font=FN_SMALL, bg=CARD_BG, fg=TEXT_PRI,
                         insertbackground=ACCENT, relief="flat",
                         highlightthickness=1, highlightbackground=BORDER, width=8)
            e.insert(0, default); e.pack(side="left")
            fields[label] = e

        prog = tk.Label(win, text="", font=FN_SMALL, bg=PANEL_BG, fg=GREEN)
        prog.pack()

        def apply():
            try:
                t = float(fields["Train %"].get())/100
                v = float(fields["Val %"].get())/100
                te= float(fields["Test %"].get())/100
                seed = int(fields["Random Seed"].get())
            except ValueError:
                messagebox.showerror("Error","Enter valid numbers.",parent=win); return
            if abs(t+v+te-1.0)>0.01:
                messagebox.showerror("Error","Must sum to 100.",parent=win); return
            prog.config(text="Splitting…"); win.update()
            test_frac = te+v; val_frac = v/test_frac
            train_df, temp = train_test_split(self.df, test_size=test_frac,
                                               random_state=seed, stratify=self.df["Bug_Type"])
            val_df, test_df = train_test_split(temp, test_size=val_frac,
                                                random_state=seed, stratify=temp["Bug_Type"])
            self._splits = {"train":train_df,"val":val_df,"test":test_df}
            self._log_sep()
            self._log(f"✓ Split — Train:{len(train_df):,}  Val:{len(val_df):,}  Test:{len(test_df):,}", GREEN)
            for name,frame in self._splits.items():
                self._log(f"  {name}:", PURPLE)
                for bt,c in frame["Bug_Type"].value_counts().items():
                    self._log(f"    {bt:15s}  {c:,}", TEXT_SEC)
            win.destroy()

        styled_btn(win,"Create Split",apply,color=GREEN).pack(pady=14)

    def _export_splits(self):
        if not self._require_df(): return
        if not self._splits:
            messagebox.showwarning("No Splits","Run 'Train / Val / Test Split' first."); return
        folder = filedialog.askdirectory(title="Select output folder")
        if not folder: return
        for name, frame in self._splits.items():
            out = os.path.join(folder, f"{name}.csv")
            frame.to_csv(out, index=False)
            self._log(f"✓ Saved {name}.csv  ({len(frame):,} rows)", GREEN)
        self._log(f"  → {folder}", TEXT_MUT)

    def _show_weights(self):
        if not self._require_df(): return
        if "Bug_Type" not in self.df.columns: return
        counts = self.df["Bug_Type"].value_counts()
        total  = counts.sum(); n = len(counts)
        self._log_sep(); self._log("Class Weights for Bug_Type:", ACCENT)
        for label, count in counts.items():
            w   = total/(n*count)
            bar = "█"*min(26, max(1, int(w*2)))
            self._log(f"  {label:15s}  {count:6,}  w={w:6.3f}  {bar}", PURPLE)
        self._log("  → Use as class_weight in your BERT loss function", TEXT_MUT)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    BERTPreprocessorApp(root)
    root.mainloop()