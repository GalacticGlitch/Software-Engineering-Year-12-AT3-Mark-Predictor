import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import os
import math
from datetime import datetime


# ─────────────────────────────────────────────
#  KNN MODEL (no external ML libraries needed)
# ─────────────────────────────────────────────

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two feature vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_predict(train_X, train_y, query, k=5):
    """
    K-Nearest Neighbours regression prediction.
    Returns predicted value and the k nearest neighbours used.
    """
    distances = []
    for i, features in enumerate(train_X):
        dist = euclidean_distance(features, query)
        distances.append((dist, train_y[i], i))

    distances.sort(key=lambda x: x[0])
    neighbours = distances[:k]

    # Weighted average: closer neighbours have more influence
    total_weight = 0
    weighted_sum = 0
    for dist, mark, idx in neighbours:
        weight = 1 / (dist + 1e-5)  # avoid division by zero
        weighted_sum += weight * mark
        total_weight += weight

    prediction = weighted_sum / total_weight
    return round(prediction, 1), neighbours


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


def export_csv(filepath, headers, rows):
    """Write updated rows back to a new CSV file."""
    base, ext = os.path.splitext(filepath)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f"{base}_predicted_{timestamp}{ext}"
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


# ─────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────

DARK_BG   = "#0f1117"
PANEL_BG  = "#1a1d27"
CARD_BG   = "#22263a"
ACCENT    = "#4f8ef7"
ACCENT2   = "#a78bfa"
SUCCESS   = "#34d399"
WARNING   = "#fbbf24"
TEXT      = "#e8eaf0"
SUBTEXT   = "#8b90a8"
BORDER    = "#2e3250"
FONT_HEAD = ("Calibri", 22, "bold")
FONT_SUB  = ("Calibri", 11, "italic")
FONT_BODY = ("Calibri", 10)
FONT_BTN  = ("Calibri", 10, "bold")
FONT_LBL  = ("Calibri", 10)
FONT_MONO = ("Calibri", 11)


class MarkPredictorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mark Predictor")
        self.geometry("860x680")
        self.minsize(780, 580)
        self.configure(bg=DARK_BG)
        self.resizable(True, True)

        # State
        self.filepath = None
        self.headers = []
        self.rows = []
        self.missing = []
        self.selected_missing = None
        self.last_prediction = None

        self._build_ui()

    # ── UI CONSTRUCTION ──────────────────────

    def _build_ui(self):
        # Header bar
        header = tk.Frame(self, bg=PANEL_BG, height=70)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)

        tk.Label(header, text="MARK PREDICTOR", font=FONT_HEAD,
                 bg=PANEL_BG, fg=TEXT).pack(side='left', padx=24, pady=14)
        tk.Label(header, text="K-Nearest Neighbors (KNN)",
                 font=FONT_SUB, bg=PANEL_BG, fg=SUBTEXT).pack(side='left', padx=4, pady=20)

        # Status bar at bottom
        self.status_var = tk.StringVar(value="Load a CSV file to begin.")
        status_bar = tk.Frame(self, bg=PANEL_BG, height=28)
        status_bar.pack(fill='x', side='bottom')
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self.status_var, font=("Calibri", 9),
                 bg=PANEL_BG, fg=SUBTEXT, anchor='w').pack(side='left', padx=14)

        # Thin accent line under header
        tk.Frame(self, bg=ACCENT, height=2).pack(fill='x', side='top')

        # Main body
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill='both', expand=True, padx=20, pady=16)

        # Left column
        left = tk.Frame(body, bg=DARK_BG, width=260)
        left.pack(side='left', fill='y', padx=(0, 12))
        left.pack_propagate(False)

        # Right column
        right = tk.Frame(body, bg=DARK_BG)
        right.pack(side='left', fill='both', expand=True)

        self._build_left(left)
        self._build_right(right)

    def _card(self, parent, title):
        """Create a labelled card frame."""
        frame = tk.Frame(parent, bg=CARD_BG, bd=0, relief='flat',
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill='x', pady=(0, 10))
        tk.Label(frame, text=title, font=("Calibri", 9, "bold"),
                 bg=CARD_BG, fg=ACCENT2).pack(anchor='w', padx=12, pady=(10, 4))
        tk.Frame(frame, bg=BORDER, height=1).pack(fill='x', padx=12)
        return frame

    def _btn(self, parent, text, cmd, color=ACCENT, width=22):
        b = tk.Button(parent, text=text, command=cmd,
                      font=FONT_BTN, bg=color, fg=DARK_BG,
                      activebackground=TEXT, activeforeground=DARK_BG,
                      relief='flat', cursor='hand2', width=width, pady=6)
        b.pack(padx=12, pady=(6, 10), fill='x')
        return b

    def _build_left(self, parent):
        # File card
        fc = self._card(parent, "① DATA FILE")
        tk.Label(fc, text="No file loaded", font=FONT_LBL,
                 bg=CARD_BG, fg=SUBTEXT, wraplength=220, justify='left',
                 name='file_label').pack(anchor='w', padx=12, pady=6)
        self.file_label = fc.children['file_label']
        self._btn(fc, "[ LOAD CSV ]", self._load_file)

        # Missing marks card
        mc = self._card(parent, "② MISSING MARKS")
        self.missing_var = tk.StringVar(value="— none loaded —")
        tk.Label(mc, textvariable=self.missing_var, font=FONT_LBL,
                 bg=CARD_BG, fg=SUBTEXT).pack(anchor='w', padx=12, pady=6)

        self.missing_list = tk.Listbox(mc, font=FONT_BODY, bg=PANEL_BG, fg=TEXT,
                                       selectbackground=ACCENT, selectforeground=DARK_BG,
                                       relief='flat', bd=0, height=6,
                                       highlightthickness=0, activestyle='none')
        self.missing_list.pack(fill='x', padx=12, pady=(0, 8))
        self.missing_list.bind('<<ListboxSelect>>', self._on_select)

        # K selector card
        kc = self._card(parent, "③ KNN SETTINGS")
        tk.Label(kc, text="Neighbours (k):", font=FONT_LBL,
                 bg=CARD_BG, fg=TEXT).pack(anchor='w', padx=12, pady=(6, 2))
        self.k_var = tk.IntVar(value=5)
        k_frame = tk.Frame(kc, bg=CARD_BG)
        k_frame.pack(anchor='w', padx=12, pady=(0, 8))
        for k in [3, 5, 7]:
            tk.Radiobutton(k_frame, text=str(k), variable=self.k_var, value=k,
                           font=FONT_LBL, bg=CARD_BG, fg=TEXT,
                           selectcolor=DARK_BG, activebackground=CARD_BG).pack(side='left', padx=4)

        self._btn(parent, "[ PREDICT MARK ]", self._predict, color=SUCCESS, width=22)
        self._btn(parent, "[ PREDICT ALL ]", self._predict_all, color=ACCENT2, width=22)
        self._btn(parent, "[ EXPORT CSV ]", self._export, color=WARNING, width=22)

    def _build_right(self, parent):
        # Student info card
        si = self._card(parent, "SELECTED STUDENT")
        self.student_info_var = tk.StringVar(value="Select a missing mark from the list.")
        tk.Label(si, textvariable=self.student_info_var, font=FONT_MONO,
                 bg=CARD_BG, fg=TEXT, justify='left', wraplength=520,
                 anchor='w').pack(anchor='w', padx=12, pady=10)

        # Results card
        rc = self._card(parent, "PREDICTION RESULT")
        self.result_var = tk.StringVar(value="—")
        tk.Label(rc, textvariable=self.result_var, font=("Calibri", 36, "bold"),
                 bg=CARD_BG, fg=SUCCESS).pack(pady=(10, 2))
        self.result_sub_var = tk.StringVar(value="Run a prediction to see results.")
        tk.Label(rc, textvariable=self.result_sub_var, font=FONT_LBL,
                 bg=CARD_BG, fg=SUBTEXT, wraplength=520, justify='left').pack(padx=12, pady=(0, 10))

        # Neighbours / log card
        lc = self._card(parent, "ANALYSIS LOG")
        self.log = tk.Text(lc, font=FONT_BODY, bg=PANEL_BG, fg=TEXT,
                           relief='flat', bd=0, height=14, state='disabled',
                           highlightthickness=0, wrap='word')
        scroll = ttk.Scrollbar(lc, orient='vertical', command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side='left', fill='both', expand=True, padx=(12, 0), pady=8)
        scroll.pack(side='right', fill='y', pady=8, padx=(0, 8))

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
        self.file_label.config(text=fname, fg=SUCCESS)
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
        self.result_sub_var.set("Press [ PREDICT MARK ] to run KNN prediction.")
        self._set_status(f"Selected: {name} → {col}")

    def _run_single_prediction(self, row_i, name, col, k):
        """
        Core KNN prediction logic for one missing mark.
        Returns (prediction, neighbours, train_names, feature_cols, query, class_avg)
        or raises ValueError with a user-facing message.
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

        # Build training set from students who have the target col filled in
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

        if len(train_X) < k:
            raise ValueError(
                f"Need at least {k} students with a known '{col}' mark. "
                f"Only {len(train_X)} found. Try reducing k."
            )

        # Query vector for the target student
        query = []
        for j, fc in enumerate(feature_cols):
            if target_features_raw[j] is not None:
                query.append(target_features_raw[j])
            else:
                vals = [train_X[r][j] for r in range(len(train_X))]
                query.append(sum(vals) / len(vals))

        prediction, neighbours = knn_predict(train_X, train_y, query, k=k)
        prediction = max(0, min(100, prediction))

        # Class average for the target column (excluding target student)
        all_vals = []
        for i, row in enumerate(self.rows):
            if i == row_i:
                continue
            v = row[col].strip()
            ok, parsed = validate_mark(v, col) if v else (False, None)
            if ok:
                all_vals.append(parsed)
        class_avg = round(sum(all_vals) / len(all_vals), 1) if all_vals else 0

        return prediction, neighbours, train_names, feature_cols, query, class_avg

    def _predict(self):
        if not self.selected_missing:
            messagebox.showwarning("No Selection", "Please select a missing mark from the list first.")
            return

        row_i, name, col = self.selected_missing
        k = self.k_var.get()

        try:
            prediction, neighbours, train_names, feature_cols, query, class_avg = \
                self._run_single_prediction(row_i, name, col, k)
        except ValueError as e:
            messagebox.showerror("Insufficient Data", str(e))
            return
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            return

        # Store prediction
        self.rows[row_i][col] = str(prediction)
        self.last_prediction = (row_i, col, prediction)

        # Update UI
        self.result_var.set(f"{prediction}")
        self.result_sub_var.set(
            f"Predicted {col} for {name}   |   "
            f"Class average: {class_avg}   |   "
            f"k = {k} neighbours used"
        )

        # Log
        self._clear_log()
        self._log(f"═══ KNN PREDICTION REPORT ═══")
        self._log(f"Student     : {name}")
        self._log(f"Target col  : {col}")
        self._log(f"k           : {k}")
        self._log(f"Feature cols: {', '.join(feature_cols)}")
        self._log(f"Query vector: {[round(q,1) for q in query]}")
        self._log("─" * 40)
        self._log(f"{'#':<4} {'Student':<22} {'Distance':>10}  {'Mark':>6}  {'Weight':>8}")
        self._log("─" * 40)
        total_w = sum(1 / (d + 1e-5) for d, _, _ in neighbours)
        for rank, (dist, mark, idx) in enumerate(neighbours, 1):
            w = (1 / (dist + 1e-5)) / total_w * 100
            self._log(f"{rank:<4} {train_names[idx]:<22} {dist:>10.3f}  {mark:>6.1f}  {w:>7.1f}%")
        self._log("─" * 40)
        self._log(f"Predicted mark  : {prediction}")
        self._log(f"Class average   : {class_avg}")
        self._log(f"Difference      : {round(prediction - class_avg, 1):+}")
        self._log("─" * 40)
        self._log("✓ Prediction complete. Run [ EXPORT CSV ] to save.")
        self._set_status(f"Predicted {col} for {name}: {prediction}  |  Class avg: {class_avg}")

    def _predict_all(self):
        """Run KNN predictions for every missing mark in the dataset."""
        if not self.missing:
            messagebox.showwarning("No Data", "Load a CSV file with missing marks first.")
            return

        k = self.k_var.get()
        succeeded, failed = [], []

        self._clear_log()
        self._log(f"═══ PREDICT ALL — {len(self.missing)} missing mark(s) ═══")
        self._log(f"k = {k}")
        self._log("─" * 50)

        for row_i, name, col in self.missing:
            try:
                prediction, neighbours, train_names, feature_cols, query, class_avg = \
                    self._run_single_prediction(row_i, name, col, k)

                # Write prediction into data
                self.rows[row_i][col] = str(prediction)
                self.last_prediction = (row_i, col, prediction)
                succeeded.append((name, col, prediction, class_avg))

                # Log summary for this prediction
                self._log(f"✓  {name:<22}  {col:<18}  → {prediction:>5}  (class avg: {class_avg})")

                # Log neighbour detail
                total_w = sum(1 / (d + 1e-5) for d, _, _ in neighbours)
                for rank, (dist, mark, idx) in enumerate(neighbours, 1):
                    w = (1 / (dist + 1e-5)) / total_w * 100
                    self._log(f"     [{rank}] {train_names[idx]:<20} dist={dist:.2f}  mark={mark:.1f}  wt={w:.1f}%")
                self._log("")

            except ValueError as e:
                failed.append((name, col, str(e)))
                self._log(f"✗  {name:<22}  {col:<18}  → SKIPPED: {e}")
                self._log("")

        # Summary
        self._log("─" * 50)
        self._log(f"Complete: {len(succeeded)} predicted, {len(failed)} skipped.")
        self._log("Run [ EXPORT CSV ] to save all results.")

        # Update result display with summary
        if succeeded:
            summary = "\n".join(f"  {name}  →  {col}: {pred}" for name, col, pred, _ in succeeded)
            self.result_var.set(f"{len(succeeded)}/{len(self.missing)}")
            self.result_sub_var.set(f"All predictions complete:\n{summary}")
            self.student_info_var.set(
                f"Predicted {len(succeeded)} mark(s):\n" +
                "\n".join(f"  {name}  →  {col}: {pred}  (class avg: {avg})"
                          for name, col, pred, avg in succeeded)
            )

        if failed:
            skipped = "\n".join(f"• {n} → {c}" for n, c, _ in failed)
            messagebox.showwarning("Some Skipped",
                                   f"{len(failed)} prediction(s) skipped (insufficient data):\n{skipped}")

        self._set_status(
            f"Predict All complete — {len(succeeded)} predicted, {len(failed)} skipped."
        )

    def _export(self):
        if not self.filepath:
            messagebox.showwarning("No File", "Load a CSV file first.")
            return
        if self.last_prediction is None:
            messagebox.showwarning("No Prediction", "Run at least one prediction before exporting.")
            return
        try:
            out = export_csv(self.filepath, self.headers, self.rows)
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save file:\n{e}")
            return

        self._log(f"\n✓ Exported to: {os.path.basename(out)}")
        self._set_status(f"Exported: {os.path.basename(out)}")
        messagebox.showinfo("Export Successful",
                            f"Updated CSV saved to:\n{out}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = MarkPredictorApp()
    app.mainloop()