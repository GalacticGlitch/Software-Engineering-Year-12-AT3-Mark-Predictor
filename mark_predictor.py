# Yes, there are warnings (not errors) but the program works, so "If it ain't broke don't fix it" and yes I have tried to fix these warnings, but it actually breaks the code when I do.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import os
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_text


# ─────────────────────────────────────────────
#  ML MODEL WRAPPERS
# ─────────────────────────────────────────────

def knn_predict(train_X, train_y, query, k=5):
    """
    K-Nearest Neighbours regression via sklearn.
    Returns (prediction, neighbour indices+distances for logging).
    """
    model = KNeighborsRegressor(n_neighbors=k, weights='distance')
    model.fit(train_X, train_y)
    prediction = round(float(model.predict([query])[0]), 1)

    # Get neighbour info for the analysis log
    distances, indices = model.kneighbors([query])
    neighbours = [
        (distances[0][i], train_y[indices[0][i]], indices[0][i])
        for i in range(k)
    ]
    return prediction, neighbours


def linear_regression_predict(train_X, train_y, query):
    """
    Linear Regression via sklearn.
    Returns (prediction, coefficients, intercept).
    """
    model = LinearRegression()
    model.fit(train_X, train_y)
    prediction = round(float(model.predict([query])[0]), 1)
    prediction = max(0.0, min(100.0, prediction))
    return prediction, list(model.coef_), float(model.intercept_)


def decision_tree_predict(train_X, train_y, query, feature_cols,
                          max_depth=4, min_samples=2):
    """
    Decision Tree regression via sklearn.
    Returns (prediction, model, tree_description_lines).
    """
    model = DecisionTreeRegressor(max_depth=max_depth,
                                  min_samples_leaf=min_samples)
    model.fit(train_X, train_y)
    prediction = round(float(model.predict([query])[0]), 1)
    prediction = max(0.0, min(100.0, prediction))

    # Human-readable tree structure
    tree_text = export_text(model, feature_names=feature_cols)
    tree_lines = tree_text.strip().split('\n')

    return prediction, model, tree_lines


# ─────────────────────────────────────────────
#  DATA HANDLING
# ─────────────────────────────────────────────

def load_csv(filepath):
    """Load CSV and return headers + list of row dicts."""
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = [dict(row) for row in reader]
    return headers, rows


def find_missing(rows, headers, skip_cols=('StudentID', 'Name')):
    """Return list of (row_index, student_name, column) for every blank cell."""
    missing = []
    for i, row in enumerate(rows):
        for col in headers:
            if col in skip_cols:
                continue
            val = row[col].strip()
            if val == '' or val is None:
                missing.append((i, row.get('Name', f'Row {i}'), col))
    return missing


def validate_mark(value, col_name):
    """Validate that a mark is a number between 0 and 100."""
    try:
        v = float(value)
        if not 0 <= v <= 100:
            raise ValueError
        return True, v
    except (ValueError, TypeError):
        return False, f"'{value}' is not a valid mark for {col_name} (must be 0–100)"


def export_csv(out_path, headers, rows):
    """Write updated rows to the given path."""
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


# ─────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────

# ── Themes ───────────────────────────────────
THEMES = {
    "dark": {
        "DARK_BG":  "#0f1117",
        "PANEL_BG": "#1a1d27",
        "CARD_BG":  "#22263a",
        "ACCENT":   "#4f8ef7",
        "ACCENT2":  "#a78bfa",
        "SUCCESS":  "#34d399",
        "WARNING":  "#fbbf24",
        "TEXT":     "#e8eaf0",
        "SUBTEXT":  "#8b90a8",
        "BORDER":   "#2e3250",
    },
    "light": {
        "DARK_BG":  "#f0f2f8",
        "PANEL_BG": "#ffffff",
        "CARD_BG":  "#e8ecf7",
        "ACCENT":   "#1a5fd4",
        "ACCENT2":  "#7c3aed",
        "SUCCESS":  "#0d7f55",
        "WARNING":  "#b45309",
        "TEXT":     "#0f1117",
        "SUBTEXT":  "#4b5068",
        "BORDER":   "#c0c8e0",
    },
}

# Active theme (mutable dict, updated on toggle)
T = dict(THEMES["dark"])

# Base font size (modified by scaling)
BASE_FONT_SIZE = 10

def _fonts(base=None):
    """Return font tuple dict at the given base size."""
    b = base if base is not None else BASE_FONT_SIZE
    return {
        "FONT_HEAD": ("Calibri", b + 12, "bold"),
        "FONT_SUB":  ("Calibri", b + 1, "italic"),
        "FONT_BODY": ("Calibri", b),
        "FONT_BTN":  ("Calibri", b, "bold"),
        "FONT_LBL":  ("Calibri", b),
        "FONT_MONO": ("Calibri", b + 1),
        "FONT_CARD": ("Calibri", b - 1, "bold"),
        "FONT_STAT": ("Calibri", b - 1),
        "FONT_BIG":  ("Calibri", b + 26, "bold"),
    }

FONTS = _fonts(BASE_FONT_SIZE)


class MarkPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mark Predictor")
        self.geometry("860x680")
        self.minsize(780, 580)
        self.configure(bg=T["DARK_BG"])
        self.resizable(True, True)

        # State
        self.filepath = None
        self.headers = []
        self.rows = []
        self.missing = []
        self.selected_missing = None
        self.last_prediction = None
        self._current_theme = "dark"
        self._font_size = BASE_FONT_SIZE
        self._all_widgets = []   # (widget, role) for bulk re-theming

        self._build_ui()
        self._bind_tooltips()

    # ── UI CONSTRUCTION ──────────────────────

    def _build_ui(self):
        # Header bar
        self._header_frame = tk.Frame(self, bg=T["PANEL_BG"], height=70)
        self._header_frame.pack(fill='x', side='top')
        self._header_frame.pack_propagate(False)

        self._title_lbl = tk.Label(
            self._header_frame, text="MARK PREDICTOR",
            font=FONTS["FONT_HEAD"], bg=T["PANEL_BG"], fg=T["TEXT"])
        self._title_lbl.pack(side='left', padx=24, pady=14)

        # Accessibility controls in header (right side)
        acc_frame = tk.Frame(self._header_frame, bg=T["PANEL_BG"])
        acc_frame.pack(side='right', padx=16)
        self._acc_frame = acc_frame

        self._theme_btn = tk.Button(
            acc_frame, text="☀ Light", font=FONTS["FONT_BTN"],
            bg=T["ACCENT2"], fg=T["DARK_BG"], relief='flat',
            cursor='hand2', padx=8, pady=3,
            command=self._toggle_theme)
        self._theme_btn.pack(side='right', padx=(6, 0))

        self._font_down_btn = tk.Button(
            acc_frame, text="A−", font=FONTS["FONT_BTN"],
            bg=T["CARD_BG"], fg=T["TEXT"], relief='flat',
            cursor='hand2', padx=8, pady=3,
            command=self._font_decrease)
        self._font_down_btn.pack(side='right', padx=(4, 0))

        self._font_up_btn = tk.Button(
            acc_frame, text="A+", font=FONTS["FONT_BTN"],
            bg=T["CARD_BG"], fg=T["TEXT"], relief='flat',
            cursor='hand2', padx=8, pady=3,
            command=self._font_increase)
        self._font_up_btn.pack(side='right', padx=(4, 0))

        # Status bar at bottom
        self.status_var = tk.StringVar(value="Load a CSV file to begin.")
        self._status_frame = tk.Frame(self, bg=T["PANEL_BG"], height=28)
        self._status_frame.pack(fill='x', side='bottom')
        self._status_frame.pack_propagate(False)
        self._status_lbl = tk.Label(
            self._status_frame, textvariable=self.status_var,
            font=FONTS["FONT_STAT"], bg=T["PANEL_BG"], fg=T["SUBTEXT"], anchor='w')
        self._status_lbl.pack(side='left', padx=14)

        # Thin accent line under header
        self._accent_line = tk.Frame(self, bg=T["ACCENT"], height=2)
        self._accent_line.pack(fill='x', side='top')

        # Main body
        self._body = tk.Frame(self, bg=T["DARK_BG"])
        self._body.pack(fill='both', expand=True, padx=20, pady=16)

        # Left column — scrollable canvas so content survives large font sizes
        left_outer = tk.Frame(self._body, bg=T["DARK_BG"], width=276)
        left_outer.pack(side='left', fill='y', padx=(0, 12))
        left_outer.pack_propagate(False)
        self._left_outer = left_outer

        self._left_canvas = tk.Canvas(
            left_outer, bg=T["DARK_BG"], bd=0,
            highlightthickness=0, width=258)
        self._left_scrollbar = ttk.Scrollbar(
            left_outer, orient='vertical', command=self._left_canvas.yview)
        self._left_canvas.configure(yscrollcommand=self._left_scrollbar.set)

        self._left_canvas.pack(side='left', fill='both', expand=True)
        # scrollbar only appears when needed — managed via _update_left_scroll

        self._left = tk.Frame(self._left_canvas, bg=T["DARK_BG"], width=258)
        self._left_window = self._left_canvas.create_window(
            (0, 0), window=self._left, anchor='nw')

        # Resize canvas scroll region when inner frame changes size
        self._left.bind('<Configure>', self._on_left_configure)
        self._left_canvas.bind('<Configure>', self._on_left_canvas_configure)

        # Mouse wheel scrolling on left panel
        self._left_canvas.bind('<Enter>',
            lambda e: self._left_canvas.bind_all('<MouseWheel>', self._on_left_scroll))
        self._left_canvas.bind('<Leave>',
            lambda e: self._left_canvas.unbind_all('<MouseWheel>'))

        # Right column
        self._right = tk.Frame(self._body, bg=T["DARK_BG"])
        self._right.pack(side='left', fill='both', expand=True)

        self._build_left(self._left)
        self._build_right(self._right)

    def _card(self, parent, title):
        """Create a labelled card frame."""
        frame = tk.Frame(parent, bg=T["CARD_BG"], bd=0, relief='flat',
                         highlightbackground=T["BORDER"], highlightthickness=1)
        frame.pack(fill='x', pady=(0, 10))
        lbl = tk.Label(frame, text=title, font=FONTS["FONT_CARD"],
                       bg=T["CARD_BG"], fg=T["ACCENT2"])
        lbl.pack(anchor='w', padx=12, pady=(10, 4))
        div = tk.Frame(frame, bg=T["BORDER"], height=1)
        div.pack(fill='x', padx=12)
        self._all_widgets += [
            (frame, "card"), (lbl, "card_title"), (div, "border")
        ]
        return frame

    def _btn(self, parent, text, cmd, color=None, width=22, tooltip=None):
        c = color or T["ACCENT"]
        b = tk.Button(parent, text=text, command=cmd,
                      font=FONTS["FONT_BTN"], bg=c, fg=T["DARK_BG"],
                      activebackground=T["TEXT"], activeforeground=T["DARK_BG"],
                      relief='flat', cursor='hand2', width=width, pady=6)
        b.pack(padx=12, pady=(6, 10), fill='x')
        if tooltip:
            self._add_tooltip(b, tooltip)
        return b

    def _build_left(self, parent):
        # File card
        fc = self._card(parent, "① DATA FILE")
        self.file_label = tk.Label(fc, text="No file loaded", font=FONTS["FONT_LBL"],
                                   bg=T["CARD_BG"], fg=T["SUBTEXT"],
                                   wraplength=220, justify='left')
        self.file_label.pack(anchor='w', padx=12, pady=6)
        self._all_widgets.append((self.file_label, "subtext"))
        self._load_btn = self._btn(fc, "[ LOAD CSV ]", self._load_file,
                                   color=T["ACCENT"],
                                   tooltip="Load a student CSV data file")

        # Missing marks card
        mc = self._card(parent, "② MISSING MARKS")
        self.missing_var = tk.StringVar(value="— none loaded —")
        miss_lbl = tk.Label(mc, textvariable=self.missing_var, font=FONTS["FONT_LBL"],
                            bg=T["CARD_BG"], fg=T["SUBTEXT"])
        miss_lbl.pack(anchor='w', padx=12, pady=6)
        self._all_widgets.append((miss_lbl, "subtext"))

        self.missing_list = tk.Listbox(mc, font=FONTS["FONT_BODY"],
                                       bg=T["PANEL_BG"], fg=T["TEXT"],
                                       selectbackground=T["ACCENT"],
                                       selectforeground=T["DARK_BG"],
                                       relief='flat', bd=0, height=6,
                                       highlightthickness=0, activestyle='none')
        self.missing_list.pack(fill='x', padx=12, pady=(0, 8))
        self.missing_list.bind('<<ListboxSelect>>', self._on_select)
        self._all_widgets.append((self.missing_list, "listbox"))

        # Algorithm selector card
        ac = self._card(parent, "③ ALGORITHM")
        self.algo_var = tk.StringVar(value="Auto")
        self._algo_radios = []
        algo_frame = tk.Frame(ac, bg=T["CARD_BG"])
        algo_frame.pack(anchor='w', padx=12, pady=(4, 8))
        self._all_widgets.append((algo_frame, "card"))
        algo_tooltips = {
            "Auto": (
                "Automatically selects the best algorithm and settings "
                "based on how much data is available."
            ),
            "KNN": (
                "K-Nearest Neighbours - finds the k most similar students "
                "and averages their marks to make a prediction."
            ),
            "Linear Regression": (
                "Linear Regression - draws a line of best fit through the data "
                "and uses it to estimate the missing mark mathematically."
            ),
            "Decision Tree": (
                "Decision Tree - asks a series of yes/no questions about a "
                "student's marks to narrow down a predicted value."
            ),
        }
        for algo, tip in algo_tooltips.items():
            rb = tk.Radiobutton(algo_frame, text=algo, variable=self.algo_var,
                                value=algo, font=FONTS["FONT_LBL"],
                                bg=T["CARD_BG"], fg=T["TEXT"],
                                selectcolor=T["DARK_BG"], activebackground=T["CARD_BG"],
                                command=self._on_algo_change)
            rb.pack(anchor='w', padx=4, pady=1)
            self._add_tooltip(rb, tip)
            self._all_widgets.append((rb, "radiobutton"))
            self._algo_radios.append(rb)

        # K selector card (KNN only)
        self._knn_card = self._card(parent, "④ KNN SETTINGS")
        k_lbl = tk.Label(self._knn_card, text="Neighbours (k):", font=FONTS["FONT_LBL"],
                         bg=T["CARD_BG"], fg=T["TEXT"])
        k_lbl.pack(anchor='w', padx=12, pady=(6, 2))
        self._all_widgets.append((k_lbl, "text"))
        self.k_var = tk.IntVar(value=5)
        k_frame = tk.Frame(self._knn_card, bg=T["CARD_BG"])
        k_frame.pack(anchor='w', padx=12, pady=(0, 8))
        self._all_widgets.append((k_frame, "card"))
        for k in [3, 5, 7]:
            rb = tk.Radiobutton(k_frame, text=str(k), variable=self.k_var, value=k,
                                font=FONTS["FONT_LBL"], bg=T["CARD_BG"], fg=T["TEXT"],
                                selectcolor=T["DARK_BG"], activebackground=T["CARD_BG"])
            rb.pack(side='left', padx=4)
            self._all_widgets.append((rb, "radiobutton"))

        # Decision Tree settings card (DT only)
        self._dt_card = self._card(parent, "④ TREE SETTINGS")
        dt_depth_lbl = tk.Label(self._dt_card, text="Max depth:", font=FONTS["FONT_LBL"],
                                bg=T["CARD_BG"], fg=T["TEXT"])
        dt_depth_lbl.pack(anchor='w', padx=12, pady=(6, 2))
        self._all_widgets.append((dt_depth_lbl, "text"))
        self.dt_depth_var = tk.IntVar(value=4)
        dt_frame = tk.Frame(self._dt_card, bg=T["CARD_BG"])
        dt_frame.pack(anchor='w', padx=12, pady=(0, 8))
        self._all_widgets.append((dt_frame, "card"))
        for d in [2, 3, 4]:
            rb = tk.Radiobutton(dt_frame, text=str(d), variable=self.dt_depth_var, value=d,
                                font=FONTS["FONT_LBL"], bg=T["CARD_BG"], fg=T["TEXT"],
                                selectcolor=T["DARK_BG"], activebackground=T["CARD_BG"])
            rb.pack(side='left', padx=4)
            self._all_widgets.append((rb, "radiobutton"))
        self._dt_card.pack_forget()   # hidden by default
        self._knn_card.pack_forget()  # hidden by default (Auto selected)

        self._predict_btn  = self._btn(parent, "[ PREDICT MARK ]", self._predict,
                                       color=T["SUCCESS"],
                                       tooltip="Predict the selected missing mark using KNN")
        self._predictall_btn = self._btn(parent, "[ PREDICT ALL ]", self._predict_all,
                                         color=T["ACCENT2"],
                                         tooltip="Automatically predict every missing mark in the dataset")
        self._export_btn   = self._btn(parent, "[ EXPORT CSV ]", self._export,
                                       color=T["WARNING"],
                                       tooltip="Save the updated data with predicted marks to a new CSV file")

    def _build_right(self, parent):
        # Student info card
        si = self._card(parent, "SELECTED STUDENT")
        self.student_info_var = tk.StringVar(value="Select a missing mark from the list.")
        self._student_info_lbl = tk.Label(
            si, textvariable=self.student_info_var, font=FONTS["FONT_MONO"],
            bg=T["CARD_BG"], fg=T["TEXT"], justify='left', wraplength=520, anchor='w')
        self._student_info_lbl.pack(anchor='w', padx=12, pady=10)
        self._all_widgets.append((self._student_info_lbl, "text"))

        # Results card
        rc = self._card(parent, "PREDICTION RESULT")
        self.result_var = tk.StringVar(value="—")
        self._result_big_lbl = tk.Label(
            rc, textvariable=self.result_var,
            font=FONTS["FONT_BIG"], bg=T["CARD_BG"], fg=T["SUCCESS"])
        self._result_big_lbl.pack(pady=(10, 2))
        self.result_sub_var = tk.StringVar(value="Run a prediction to see results.")
        self._result_sub_lbl = tk.Label(
            rc, textvariable=self.result_sub_var, font=FONTS["FONT_LBL"],
            bg=T["CARD_BG"], fg=T["SUBTEXT"], wraplength=520, justify='left')
        self._result_sub_lbl.pack(padx=12, pady=(0, 10))
        self._all_widgets += [
            (self._result_big_lbl, "success"),
            (self._result_sub_lbl, "subtext"),
        ]

        # Neighbours / log card
        lc = self._card(parent, "ANALYSIS LOG")
        self.log = tk.Text(lc, font=FONTS["FONT_BODY"], bg=T["PANEL_BG"], fg=T["TEXT"],
                           relief='flat', bd=0, height=14, state='disabled',
                           highlightthickness=0, wrap='word')
        scroll = ttk.Scrollbar(lc, orient='vertical', command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side='left', fill='both', expand=True, padx=(12, 0), pady=8)
        scroll.pack(side='right', fill='y', pady=8, padx=(0, 8))
        self._all_widgets.append((self.log, "log"))

    # ── LEFT PANEL SCROLL HELPERS ────────────

    def _on_left_configure(self, event):
        """Update scroll region and show/hide scrollbar as needed."""
        self._left_canvas.configure(
            scrollregion=self._left_canvas.bbox('all'))
        self._update_left_scroll()

    def _on_left_canvas_configure(self, event):
        """Keep inner frame width in sync with canvas width."""
        self._left_canvas.itemconfig(self._left_window, width=event.width)
        self._update_left_scroll()

    def _update_left_scroll(self):
        """Show scrollbar only when content is taller than the canvas."""
        canvas_h  = self._left_canvas.winfo_height()
        content_h = self._left.winfo_reqheight()
        if content_h > canvas_h:
            self._left_scrollbar.pack(side='right', fill='y')
        else:
            self._left_scrollbar.pack_forget()

    def _on_left_scroll(self, event):
        self._left_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    # ── TOOLTIPS ─────────────────────────────

    def _add_tooltip(self, widget, text):
        """Attach a hover tooltip to a widget."""
        tip_win = [None]

        def show(event):
            if tip_win[0]:
                return
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            tw = tk.Toplevel(self)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            lbl = tk.Label(tw, text=text, font=FONTS["FONT_STAT"],
                           bg=T["PANEL_BG"], fg=T["TEXT"],
                           relief='flat', bd=0, padx=8, pady=4,
                           highlightbackground=T["BORDER"], highlightthickness=1)
            lbl.pack()
            tip_win[0] = tw

        def hide(event):
            if tip_win[0]:
                tip_win[0].destroy()
                tip_win[0] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _bind_tooltips(self):
        """Tooltips for the accessibility buttons in the header."""
        self._add_tooltip(self._font_up_btn,   "Increase font size")
        self._add_tooltip(self._font_down_btn,  "Decrease font size")
        self._add_tooltip(self._theme_btn,      "Toggle light / dark mode")

    # ── THEME & FONT SCALING ─────────────────

    def _toggle_theme(self):
        self._current_theme = "light" if self._current_theme == "dark" else "dark"
        T.update(THEMES[self._current_theme])
        self._theme_btn.config(
            text="🌙 Dark" if self._current_theme == "light" else "☀ Light",
            bg=T["ACCENT2"], fg=T["DARK_BG"]
        )
        self._apply_theme()

    def _font_increase(self):
        if self._font_size < 18:
            self._font_size += 1
            FONTS.update(_fonts(self._font_size))
            self._apply_theme()

    def _font_decrease(self):
        if self._font_size > 8:
            self._font_size -= 1
            FONTS.update(_fonts(self._font_size))
            self._apply_theme()

    def _apply_theme(self):
        """Re-apply current theme colours and fonts to every tracked widget."""
        self.configure(bg=T["DARK_BG"])
        self._header_frame.configure(bg=T["PANEL_BG"])
        self._title_lbl.configure(bg=T["PANEL_BG"], fg=T["TEXT"], font=FONTS["FONT_HEAD"])
        self._acc_frame.configure(bg=T["PANEL_BG"])
        self._accent_line.configure(bg=T["ACCENT"])
        self._body.configure(bg=T["DARK_BG"])
        self._left_outer.configure(bg=T["DARK_BG"])
        self._left_canvas.configure(bg=T["DARK_BG"])
        self._left.configure(bg=T["DARK_BG"])
        self._right.configure(bg=T["DARK_BG"])
        self._status_frame.configure(bg=T["PANEL_BG"])
        self._status_lbl.configure(bg=T["PANEL_BG"], fg=T["SUBTEXT"], font=FONTS["FONT_STAT"])

        # Accessibility buttons
        for btn in [self._font_up_btn, self._font_down_btn]:
            btn.configure(bg=T["CARD_BG"], fg=T["TEXT"], font=FONTS["FONT_BTN"])
        self._theme_btn.configure(bg=T["ACCENT2"], fg=T["DARK_BG"], font=FONTS["FONT_BTN"])

        # Main action buttons
        self._load_btn.configure(      bg=T["ACCENT"],  fg=T["DARK_BG"], font=FONTS["FONT_BTN"])
        self._predict_btn.configure(   bg=T["SUCCESS"], fg=T["DARK_BG"], font=FONTS["FONT_BTN"])
        self._predictall_btn.configure(bg=T["ACCENT2"], fg=T["DARK_BG"], font=FONTS["FONT_BTN"])
        self._export_btn.configure(    bg=T["WARNING"],  fg=T["DARK_BG"], font=FONTS["FONT_BTN"])

        # Result big label
        self._result_big_lbl.configure(bg=T["CARD_BG"], fg=T["SUCCESS"], font=FONTS["FONT_BIG"])
        self._result_sub_lbl.configure(bg=T["CARD_BG"], fg=T["SUBTEXT"], font=FONTS["FONT_LBL"])
        self._student_info_lbl.configure(bg=T["CARD_BG"], fg=T["TEXT"],  font=FONTS["FONT_MONO"])

        # Listbox
        self.missing_list.configure(
            bg=T["PANEL_BG"], fg=T["TEXT"], font=FONTS["FONT_BODY"],
            selectbackground=T["ACCENT"], selectforeground=T["DARK_BG"])

        # Log text widget
        self.log.configure(bg=T["PANEL_BG"], fg=T["TEXT"], font=FONTS["FONT_BODY"])

        # File label (preserve success colour if file loaded)
        if self.filepath:
            self.file_label.configure(bg=T["CARD_BG"], fg=T["SUCCESS"], font=FONTS["FONT_LBL"])
        else:
            self.file_label.configure(bg=T["CARD_BG"], fg=T["SUBTEXT"], font=FONTS["FONT_LBL"])

        # Bulk-tracked widgets
        role_map = {
            "card":       {"bg": T["CARD_BG"]},
            "card_title": {"bg": T["CARD_BG"], "fg": T["ACCENT2"], "font": FONTS["FONT_CARD"]},
            "border":     {"bg": T["BORDER"]},
            "text":       {"bg": T["CARD_BG"], "fg": T["TEXT"],    "font": FONTS["FONT_LBL"]},
            "subtext":    {"bg": T["CARD_BG"], "fg": T["SUBTEXT"], "font": FONTS["FONT_LBL"]},
            "success":    {"bg": T["CARD_BG"], "fg": T["SUCCESS"]},
            "listbox":    {},   # handled above
            "log":        {},   # handled above
            "radiobutton": {
                "bg": T["CARD_BG"], "fg": T["TEXT"],
                "selectcolor": T["DARK_BG"], "activebackground": T["CARD_BG"],
                "font": FONTS["FONT_LBL"]
            },
        }
        for widget, role in self._all_widgets:
            props = role_map.get(role, {})
            if props:
                try:
                    widget.configure(**props)
                except tk.TclError:
                    pass

    # ── ACTIONS ──────────────────────────────

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _log(self, msg, tag=None):
        self.log.configure(state='normal')
        self.log.insert('end', msg + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _clear_log(self):
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select student CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.headers, self.rows = load_csv(path)
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not read file:\n{e}")
            return

        if 'Name' not in self.headers:
            messagebox.showerror("Format Error", "CSV must contain a 'Name' column.")
            return

        self.filepath = path
        fname = os.path.basename(path)
        self.file_label.config(text=fname, fg=T["SUCCESS"])
        self._set_status(f"Loaded: {fname}  |  {len(self.rows)} students, {len(self.headers)} columns")

        self.missing = find_missing(self.rows, self.headers)
        self.missing_list.delete(0, 'end')

        if not self.missing:
            self.missing_var.set("✓ No missing marks found.")
            self._clear_log()
            self._log("No missing marks detected in the dataset.")
        else:
            self.missing_var.set(f"{len(self.missing)} missing mark(s) found:")
            for _, name, col in self.missing:
                self.missing_list.insert('end', f"  {name}  →  {col}")

        self._clear_log()
        self._log(f"File loaded: {fname}")
        self._log(f"Students: {len(self.rows)}   Columns: {', '.join(self.headers)}")
        self._log(f"Missing marks detected: {len(self.missing)}")
        self._log("─" * 50)

    def _on_select(self, event):
        sel = self.missing_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self.selected_missing = self.missing[idx]
        row_i, name, col = self.selected_missing
        row = self.rows[row_i]

        # Build student summary
        known = {c: row[c] for c in self.headers
                 if c not in ('StudentID', 'Name') and row[c].strip() != ''}
        known_str = "   ".join(f"{c}: {v}" for c, v in known.items())
        self.student_info_var.set(
            f"Student:  {name}\n"
            f"Missing:  {col}\n"
            f"Known marks:  {known_str or 'none'}"
        )
        self.result_var.set("—")
        self.result_sub_var.set(f"Press [ PREDICT MARK ] to run {self.algo_var.get()} prediction.")
        self._set_status(f"Selected: {name} → {col}")

    def _auto_select_algo(self):
        """
        Pick the best algorithm and settings based on dataset size.
          < 10 students  → KNN k=3
          10–20 students → KNN k=5
          21–35 students → Linear Regression
          > 35 students  → Decision Tree depth=4
        Returns (algo_name, k, dt_depth).
        """
        n = len(self.rows)
        if n < 10:
            return "KNN", 3, 4
        elif n <= 20:
            return "KNN", 5, 4
        elif n <= 35:
            return "Linear Regression", 5, 4
        else:
            return "Decision Tree", 5, 4

    def _on_algo_change(self):
        """Show/hide settings cards based on selected algorithm."""
        algo = self.algo_var.get()

        if algo == "Auto":
            self._knn_card.pack_forget()
            self._dt_card.pack_forget()
            self._add_tooltip(self._predict_btn, f"Auto: selects the best algorithm based on data size")
            return

        if algo == "KNN":
            self._knn_card.pack(fill='x', pady=(0, 10), before=self._predict_btn)
            self._dt_card.pack_forget()
        elif algo == "Decision Tree":
            self._dt_card.pack(fill='x', pady=(0, 10), before=self._predict_btn)
            self._knn_card.pack_forget()
        else:
            self._knn_card.pack_forget()
            self._dt_card.pack_forget()

        tips = {
            "KNN": "Predict using K-Nearest Neighbours",
            "Linear Regression": "Predict using Linear Regression",
            "Decision Tree": "Predict using a Decision Tree",
        }
        self._add_tooltip(self._predict_btn, tips.get(algo, "Predict the missing mark"))

    def _run_single_prediction(self, row_i, name, col, k):
        """
        Dispatch to the selected algorithm and return a unified result dict.
        Keys: prediction, algo, feature_cols, query, class_avg, extras (algo-specific)
        """
        assessment_cols = [c for c in self.headers if c not in ('StudentID', 'Name')]
        target_row = self.rows[row_i]
        feature_cols = [c for c in assessment_cols if c != col]

        # Target student's known feature values
        target_features_raw = []
        for fc in feature_cols:
            val = target_row[fc].strip()
            valid, parsed = validate_mark(val, fc) if val else (False, None)
            target_features_raw.append(parsed if valid else None)

        # Build training set
        train_X, train_y, train_names = [], [], []
        for i, row in enumerate(self.rows):
            if i == row_i:
                continue
            target_val = row[col].strip()
            ok, y_val = validate_mark(target_val, col) if target_val else (False, None)
            if not ok:
                continue
            features = []
            for j, fc in enumerate(feature_cols):
                v = row[fc].strip()
                valid, parsed = validate_mark(v, fc) if v else (False, None)
                if valid:
                    features.append(parsed)
                elif target_features_raw[j] is not None:
                    features.append(target_features_raw[j])
                else:
                    features.append(50.0)
            train_X.append(features)
            train_y.append(y_val)
            train_names.append(row.get('Name', f'Row {i}'))

        algo = self.algo_var.get()
        if algo == "Auto":
            algo, k, dt_depth = self._auto_select_algo()
        else:
            dt_depth = self.dt_depth_var.get()
        extras = {}

        min_needed = k if algo == "KNN" else 3
        if len(train_X) < min_needed:
            raise ValueError(
                f"Need at least {min_needed} students with a known '{col}' mark. "
                f"Only {len(train_X)} found."
            )

        # Query vector
        query = []
        for j in range(len(feature_cols)):
            if target_features_raw[j] is not None:
                query.append(target_features_raw[j])
            else:
                vals = [train_X[r][j] for r in range(len(train_X))]
                query.append(sum(vals) / len(vals))

        # Class average
        all_vals = []
        for i, row in enumerate(self.rows):
            if i == row_i:
                continue
            v = row[col].strip()
            ok, parsed = validate_mark(v, col) if v else (False, None)
            if ok:
                all_vals.append(parsed)
        class_avg = round(sum(all_vals) / len(all_vals), 1) if all_vals else 0

        if algo == "KNN":
            prediction, neighbours = knn_predict(train_X, train_y, query, k=k)
            prediction = max(0, min(100, prediction))
            extras = {"neighbours": neighbours, "train_names": train_names,
                      "train_X": train_X}

        elif algo == "Linear Regression":
            prediction, coefficients, intercept = linear_regression_predict(
                train_X, train_y, query)
            extras = {"coefficients": coefficients, "intercept": intercept,
                      "feature_cols": feature_cols}

        elif algo == "Decision Tree":
            prediction, tree_root, tree_desc = decision_tree_predict(
                train_X, train_y, query, feature_cols, max_depth=dt_depth)
            extras = {"tree_desc": tree_desc, "tree_root": tree_root}

        return {
            "prediction": prediction,
            "algo": algo,
            "feature_cols": feature_cols,
            "query": query,
            "class_avg": class_avg,
            "train_names": train_names,
            "extras": extras,
        }

    def _log_mark_influence(self, algo, feature_cols, query, extras):
        """
        Log a plain-English breakdown of how each of the student's
        own known marks influenced the predicted result.
        """
        self._log("")
        self._log("── HOW THIS STUDENT'S MARKS INFLUENCED THE PREDICTION ──")

        if algo == "KNN":
            neighbours = extras["neighbours"]
            train_X    = extras.get("train_X", [])
            self._log(f"{'Mark':<22} {'Your value':>12}  {'Neighbour avg':>14}  {'Difference':>11}")
            self._log("─" * 64)
            for j, fc in enumerate(feature_cols):
                sv  = query[j]
                nav = round(
                    sum(train_X[idx][j] for _, _, idx in neighbours) / len(neighbours), 1
                ) if train_X else sv
                diff  = round(sv - nav, 1)
                arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
                self._log(f"{fc:<22} {sv:>12.1f}  {nav:>14.1f}  {diff:>+10.1f} {arrow}")
            self._log("")
            self._log("↑ Your mark is above the neighbour avg → pulled prediction up.")
            self._log("↓ Your mark is below the neighbour avg → pulled prediction down.")

        elif algo == "Linear Regression":
            coefficients = extras["coefficients"]
            intercept    = extras["intercept"]
            total = intercept
            self._log(f"  {'Mark':<22} {'Your value':>10}  {'Coefficient':>12}  {'Contribution':>13}")
            self._log("  " + "─" * 62)
            self._log(f"  {'(intercept)':<22} {'':>10}  {'':>12}  {round(intercept,2):>13.2f}")
            for fc, coef, val in zip(feature_cols, coefficients, query):
                contrib = coef * val
                total  += contrib
                direction = "↑" if contrib > 0 else ("↓" if contrib < 0 else "=")
                self._log(
                    f"  {fc:<22} {val:>10.1f}  {coef:>12.4f}  "
                    f"{contrib:>+12.2f} {direction}"
                )
            self._log("  " + "─" * 62)
            self._log(f"  {'Total (= prediction)':<22} {'':>10}  {'':>12}  {round(total,2):>13.2f}")
            self._log("")
            self._log("  Contribution = your mark × coefficient.")
            self._log("  A positive coefficient means higher marks push the prediction up.")
            self._log("  A negative coefficient means higher marks push the prediction down.")

        elif algo == "Decision Tree":
            model = extras.get("tree_root")  # sklearn model stored here
            if model is None:
                self._log("  (tree path not available)")
                return
            import numpy as np
            node_indicator = model.decision_path(np.array([query]))
            node_ids = node_indicator.indices
            tree = model.tree_
            self._log("  Decision path through the tree for this student:")
            self._log("")
            depth = 0
            for node_id in node_ids[:-1]:  # exclude the leaf
                feat  = tree.feature[node_id]
                thresh = round(tree.threshold[node_id], 1)
                val   = query[feat]
                fname = feature_cols[feat] if feat < len(feature_cols) else f"F{feat}"
                went_left = val <= thresh
                direction = "YES  →  go left" if went_left else "NO   →  go right"
                self._log(
                    f"  {'  ' * depth}Q: Is {fname} ≤ {thresh}?  "
                    f"(Your value: {val:.1f})  {direction}"
                )
                depth += 1
            leaf_id = node_ids[-1]
            leaf_val = round(tree.value[leaf_id][0][0], 1)
            self._log(f"  {'  ' * depth}→ Leaf reached: predict {leaf_val}")
            self._log("")
            self._log("  Each question was answered using this student's own marks.")

    def _predict(self):
        if not self.selected_missing:
            messagebox.showwarning("No Selection", "Please select a missing mark from the list first.")
            return

        row_i, name, col = self.selected_missing
        k = self.k_var.get()

        try:
            result = self._run_single_prediction(row_i, name, col, k)
        except ValueError as e:
            messagebox.showerror("Insufficient Data", str(e))
            return
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        prediction = result["prediction"]
        algo       = result["algo"]
        class_avg  = result["class_avg"]
        query      = result["query"]
        feature_cols = result["feature_cols"]
        extras     = result["extras"]

        self.rows[row_i][col] = str(prediction)
        self.last_prediction = (row_i, col, prediction)

        self.result_var.set(f"{prediction}")
        self.result_sub_var.set(
            f"[{algo}]  Predicted {col} for {name}   |   Class average: {class_avg}"
        )

        self._clear_log()
        self._log(f"═══ {algo.upper()} PREDICTION REPORT ═══")
        self._log(f"Student     : {name}")
        self._log(f"Target col  : {col}")
        self._log(f"Algorithm   : {algo}")
        self._log(f"Feature cols: {', '.join(feature_cols)}")
        self._log(f"Query vector: {[round(q, 1) for q in query]}")
        self._log("─" * 40)

        if algo == "KNN":
            neighbours  = extras["neighbours"]
            train_names = extras["train_names"]
            self._log(f"k           : {k}")
            self._log(f"{'#':<4} {'Student':<22} {'Distance':>10}  {'Mark':>6}  {'Weight':>8}")
            self._log("─" * 40)
            total_w = sum(1 / (d + 1e-5) for d, _, _ in neighbours)
            for rank, (dist, mark, idx) in enumerate(neighbours, 1):
                w = (1 / (dist + 1e-5)) / total_w * 100
                self._log(f"{rank:<4} {train_names[idx]:<22} {dist:>10.3f}  {mark:>6.1f}  {w:>7.1f}%")

        elif algo == "Linear Regression":
            coefficients = extras["coefficients"]
            intercept    = extras["intercept"]
            self._log(f"Intercept   : {round(intercept, 4)}")
            self._log("Coefficients:")
            for fc, coef in zip(feature_cols, coefficients):
                self._log(f"  {fc:<20} : {round(coef, 4)}")
            self._log("")
            equation = f"  {round(intercept,2)}"
            for fc, coef in zip(feature_cols, coefficients):
                equation += f" + {round(coef,2)}×{fc}"
            self._log(f"Equation: ŷ = {equation}")

        elif algo == "Decision Tree":
            tree_desc = extras["tree_desc"]
            self._log(f"Max depth   : {self.dt_depth_var.get()}")
            self._log("Tree structure:")
            for line in tree_desc:
                self._log("  " + line)

        self._log_mark_influence(algo, feature_cols, query, extras)
        self._log("─" * 40)
        self._log(f"Predicted mark  : {prediction}")
        self._log(f"Class average   : {class_avg}")
        self._log(f"Difference      : {round(prediction - class_avg, 1):+}")
        self._log("─" * 40)
        self._log("✓ Prediction complete. Run [ EXPORT CSV ] to save.")
        self._set_status(f"[{algo}] Predicted {col} for {name}: {prediction}  |  Class avg: {class_avg}")

    def _predict_all(self):
        """Run predictions for every missing mark using the selected algorithm."""
        if not self.missing:
            messagebox.showwarning("No Data", "Load a CSV file with missing marks first.")
            return

        k    = self.k_var.get()
        algo = self.algo_var.get()
        succeeded, failed = [], []

        self._clear_log()
        self._log(f"═══ PREDICT ALL — {len(self.missing)} missing mark(s) ═══")
        self._log(f"Algorithm : {algo}")
        if algo == "KNN":
            self._log(f"k         : {k}")
        elif algo == "Decision Tree":
            self._log(f"Max depth : {self.dt_depth_var.get()}")
        self._log("─" * 50)

        for row_i, name, col in self.missing:
            try:
                result = self._run_single_prediction(row_i, name, col, k)
                prediction = result["prediction"]
                class_avg  = result["class_avg"]
                extras     = result["extras"]

                self.rows[row_i][col] = str(prediction)
                self.last_prediction  = (row_i, col, prediction)
                succeeded.append((name, col, prediction, class_avg))

                self._log(f"✓  {name:<22}  {col:<18}  → {prediction:>5}  (class avg: {class_avg})")

                # Algorithm-specific neighbour/detail log
                if algo == "KNN":
                    neighbours  = extras["neighbours"]
                    train_names = extras["train_names"]
                    total_w = sum(1 / (d + 1e-5) for d, _, _ in neighbours)
                    for rank, (dist, mark, idx) in enumerate(neighbours, 1):
                        w = (1 / (dist + 1e-5)) / total_w * 100
                        self._log(f"     [{rank}] {train_names[idx]:<20} dist={dist:.2f}  mark={mark:.1f}  wt={w:.1f}%")
                elif algo == "Linear Regression":
                    self._log(f"     intercept={round(extras['intercept'],2)}  "
                              f"coefs={[round(c,2) for c in extras['coefficients']]}")
                elif algo == "Decision Tree":
                    self._log(f"     tree depth={self.dt_depth_var.get()}")
                self._log("")

            except ValueError as e:
                failed.append((name, col, str(e)))
                self._log(f"✗  {name:<22}  {col:<18}  → SKIPPED: {e}")
                self._log("")

        self._log("─" * 50)
        self._log(f"Complete: {len(succeeded)} predicted, {len(failed)} skipped.")
        self._log("Run [ EXPORT CSV ] to save all results.")

        if succeeded:
            summary = "\n".join(f"  {name}  →  {col}: {pred}" for name, col, pred, _ in succeeded)
            self.result_var.set(f"{len(succeeded)}/{len(self.missing)}")
            self.result_sub_var.set(f"[{algo}] All predictions complete:\n{summary}")
            self.student_info_var.set(
                f"Predicted {len(succeeded)} mark(s) using {algo}:\n" +
                "\n".join(f"  {name}  →  {col}: {pred}  (class avg: {avg})"
                          for name, col, pred, avg in succeeded)
            )

        if failed:
            skipped = "\n".join(f"• {n} → {c}" for n, c, _ in failed)
            messagebox.showwarning("Some Skipped",
                                   f"{len(failed)} prediction(s) skipped:\n{skipped}")

        self._set_status(
            f"[{algo}] Predict All — {len(succeeded)} predicted, {len(failed)} skipped."
        )

    def _export(self):
        if not self.filepath:
            messagebox.showwarning("No File", "Load a CSV file first.")
            return
        if self.last_prediction is None:
            messagebox.showwarning("No Prediction", "Run at least one prediction before exporting.")
            return

        # Ask user where to save
        base, ext = os.path.splitext(os.path.basename(self.filepath))
        from datetime import datetime
        default_name = f"{base}_predicted_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        out_path = filedialog.asksaveasfilename(
            title="Save predicted CSV",
            initialfile=default_name,
            initialdir=os.path.dirname(self.filepath),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not out_path:
            return  # user cancelled

        try:
            out = export_csv(out_path, self.headers, self.rows)
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save file:\n{e}")
            return

        self._log(f"\n✓ Exported to: {os.path.basename(out)}")
        self._set_status(f"Exported: {os.path.basename(out)}")

        # Custom export dialog with Open File option
        dialog = tk.Toplevel(self)
        dialog.title("Export Successful")
        dialog.configure(bg=T["PANEL_BG"])
        dialog.resizable(False, False)
        dialog.grab_set()

        self.update_idletasks()
        x = self.winfo_x() + self.winfo_width() // 2 - 200
        y = self.winfo_y() + self.winfo_height() // 2 - 80
        dialog.geometry(f"400x160+{x}+{y}")

        tk.Label(dialog, text="✓  Export Successful", font=FONTS["FONT_BTN"],
                 bg=T["PANEL_BG"], fg=T["SUCCESS"]).pack(pady=(18, 4))
        tk.Label(dialog, text=os.path.basename(out), font=FONTS["FONT_LBL"],
                 bg=T["PANEL_BG"], fg=T["SUBTEXT"], wraplength=360).pack(pady=(0, 14))

        btn_frame = tk.Frame(dialog, bg=T["PANEL_BG"])
        btn_frame.pack()

        def open_file():
            import subprocess, sys
            if sys.platform == "win32":
                os.startfile(out)
            elif sys.platform == "darwin":
                subprocess.call(["open", out])
            else:
                subprocess.call(["xdg-open", out])
            dialog.destroy()

        tk.Button(btn_frame, text="Open File", font=FONTS["FONT_BTN"],
                  bg=T["ACCENT"], fg=T["DARK_BG"], relief='flat',
                  cursor='hand2', padx=16, pady=6,
                  command=open_file).pack(side='left', padx=8)

        tk.Button(btn_frame, text="Close", font=FONTS["FONT_BTN"],
                  bg=T["CARD_BG"], fg=T["TEXT"], relief='flat',
                  cursor='hand2', padx=16, pady=6,
                  command=dialog.destroy).pack(side='left', padx=8)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = MarkPredictorApp()
    app.mainloop()