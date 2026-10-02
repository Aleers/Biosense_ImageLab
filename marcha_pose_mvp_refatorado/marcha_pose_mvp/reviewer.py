import tkinter as tk
import threading
from tkinter import messagebox, simpledialog, ttk
import cv2, numpy as np
from PIL import Image, ImageTk
from i18n import tr
from processor import ProcessingOptions, draw_manual_segments, draw_overlay, ear_center
COLORS = [('#24d6ff', (255, 214, 36)), ('#ff9f2f', (47, 159, 255)), ('#ff42c8', (200, 66, 255)), ('#35d66f', (111, 214, 53)), ('#4287ff', (255, 135, 66)), ('#ffe342', (66, 227, 255)), ('#ff4545', (69, 69, 255)), ('#a855f7', (247, 85, 168)), ('#ffffff', (255, 255, 255))]

def choose_color(parent, lang):
    win = tk.Toplevel(parent)
    win.title(tr(lang, 'choose_color'))
    win.transient(parent)
    win.grab_set()
    result = [COLORS[0][1]]
    ttk.Label(win, text=tr(lang, 'choose_color'), font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, columnspan=3, padx=16, pady=12)

    def select(color):
        result[0] = color
        win.destroy()
    for i, (hex_color, bgr) in enumerate(COLORS):
        tk.Button(win, bg=hex_color, width=7, height=2, command=lambda c=bgr: select(c)).grid(row=1 + i // 3, column=i % 3, padx=6, pady=6)
    win.wait_window()
    return result[0]

class ReviewWindow(tk.Toplevel):
    """Tela de revisao quadro a quadro da analise automatica.

    Reune somente as interacoes posteriores a deteccao: reproducao, calibracao,
    medidas, edicao de landmarks, rastreamento da correcao e desfazer.
    """

    def __init__(self, parent, data, opts, on_export):
        super().__init__(parent)
        self.data, self.opts, self.on_export = (data, opts, on_export)
        if opts.side == 'Automatico':
            self.opts = ProcessingOptions(opts.view, 'Esquerdo' if data.predominant_side == 'left' else 'Direito', opts.visibility_threshold, opts.language)
        self.lang = self.opts.language
        self.title(tr(self.lang, 'review'))
        self.geometry('1280x820')
        self.minsize(1050, 700)
        self.index = 0
        self.selected = None
        self.calibration_clicks = []
        self.measurement_clicks = []
        self.measurement_landmarks = []
        self.playing = False
        self.mode = tk.StringVar(value='Editar')
        self.internal_slide = False
        self.tracking = False
        self.undo_stack = []
        self.cap = cv2.VideoCapture(str(data.input_path))
        self.cap_next = 0
        self.cache = {}
        self.pending_show = None
        self.ear_centers = [ear_center(p) for p in data.landmarks] if self.opts.view == 'Lateral' else None
        self.protocol('WM_DELETE_WINDOW', self.close)
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill='x')
        ttk.Button(bar, text=tr(self.lang, 'play'), command=self.toggle).pack(side='left')
        self.undo_button = ttk.Button(bar, text=tr(self.lang, 'undo'), command=self.undo, state='disabled')
        self.undo_button.pack(side='left', padx=8)
        ttk.Radiobutton(bar, text=tr(self.lang, 'edit'), variable=self.mode, value='Editar').pack(side='left', padx=12)
        ttk.Radiobutton(bar, text=tr(self.lang, 'calibrate'), variable=self.mode, value='Calibrar').pack(side='left')
        self.measure_button = ttk.Button(bar, text=tr(self.lang, 'measure'), command=lambda: self.mode.set('Medir'), state='normal' if data.px_per_cm else 'disabled')
        self.measure_button.pack(side='left', padx=12)
        self.landmark_measure_button = ttk.Button(bar, text=tr(self.lang, 'measure_landmarks'), command=lambda: self.mode.set('MedirPontos'), state='normal' if data.px_per_cm else 'disabled')
        self.landmark_measure_button.pack(side='left')
        self.export_button = ttk.Button(bar, text=tr(self.lang, 'export'), command=lambda: on_export(self))
        self.export_button.pack(side='right')
        self.info = ttk.Label(self, text=tr(self.lang, 'review_help'))
        self.info.pack(fill='x', padx=10)
        self.canvas = tk.Canvas(self, bg='black', cursor='crosshair')
        self.canvas.pack(fill='both', expand=True, padx=10, pady=8)
        self.canvas.bind('<Button-1>', self.click)
        self.slider = ttk.Scale(self, from_=0, to=len(data.landmarks) - 1, command=self.slide)
        self.slider.pack(fill='x', padx=12, pady=(0, 10))
        self.after(100, self.show)

    def slide(self, v):
        if self.internal_slide:
            return
        self.index = int(float(v))
        if self.pending_show:
            self.after_cancel(self.pending_show)
        self.pending_show = self.after(35, self.show)

    def push_action(self, action):
        self.undo_stack.append(action)
        self.undo_button.config(state='normal')

    def undo(self):
        if not self.undo_stack or self.tracking:
            return
        action = self.undo_stack.pop()
        kind = action['type']
        if kind == 'manual' and self.data.manual_segments:
            self.data.manual_segments.pop()
        elif kind == 'tracked' and self.data.tracked_segments:
            self.data.tracked_segments.pop()
        elif kind == 'calibration':
            self.data.calibration_points = action['points']
            self.data.calibration_cm = action['cm']
        elif kind == 'point':
            for index, value in action['snapshot']:
                if self.data.landmarks[index] is not None:
                    self.data.landmarks[index][action['landmark']] = value
        enabled = 'normal' if self.data.px_per_cm else 'disabled'
        self.measure_button.config(state=enabled)
        self.landmark_measure_button.config(state=enabled)
        if not self.undo_stack:
            self.undo_button.config(state='disabled')
        if self.opts.view == 'Lateral':
            self.ear_centers = [ear_center(p) for p in self.data.landmarks]
        self.show()

    def get_frame(self, index):
        if index in self.cache:
            return self.cache[index].copy()
        if index != self.cap_next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        self.cap_next = index + 1
        if not ok:
            raise ValueError(f'Nao foi possivel ler o quadro {index}.')
        self.cache[index] = frame
        if len(self.cache) > 40:
            self.cache.pop(next(iter(self.cache)))
        return frame.copy()

    def show(self):
        self.pending_show = None
        frame = self.get_frame(self.index)
        trail = self.ear_centers[:self.index + 1] if self.ear_centers is not None else None
        frame, _ = draw_overlay(frame, self.data.landmarks[self.index], self.opts, self.data.calibration_points, True, trail)
        current_points = self.data.landmarks[self.index]
        draw_manual_segments(frame, self.data, self.lang, current_points)
        cw, ch = (max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height()))
        self.scale = min(cw / self.data.width, ch / self.data.height)
        self.ox = (cw - self.data.width * self.scale) / 2
        self.oy = (ch - self.data.height * self.scale) / 2
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((int(self.data.width * self.scale), int(self.data.height * self.scale)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(im)
        mode_label = {'Editar': tr(self.lang, 'mode_edit'), 'Calibrar': tr(self.lang, 'mode_calibrate'), 'Medir': tr(self.lang, 'mode_measure'), 'MedirPontos': tr(self.lang, 'mode_landmarks')}.get(self.mode.get(), self.mode.get())
        self.canvas.delete('all')
        self.canvas.create_image(cw / 2, ch / 2, image=self.photo)
        self.info.config(text=f"{tr(self.lang, 'frame')} {self.index + 1}/{len(self.data.landmarks)} | {self.index / self.data.fps:.2f}s | {mode_label}")

    def click(self, e):
        x, y = ((e.x - self.ox) / self.scale, (e.y - self.oy) / self.scale)
        if not (0 <= x < self.data.width and 0 <= y < self.data.height):
            return
        if self.mode.get() == 'Calibrar':
            self.calibration_clicks.append((x, y))
            if len(self.calibration_clicks) == 2:
                cm = simpledialog.askfloat(tr(self.lang, 'distance'), tr(self.lang, 'distance_prompt'), minvalue=0.01, parent=self)
                if cm:
                    self.push_action({'type': 'calibration', 'points': self.data.calibration_points, 'cm': self.data.calibration_cm})
                    self.data.calibration_points = tuple(self.calibration_clicks)
                    self.data.calibration_cm = cm
                    self.measure_button.config(state='normal')
                    self.landmark_measure_button.config(state='normal')
                    messagebox.showinfo(tr(self.lang, 'calibrated'), f'{self.data.px_per_cm:.2f} pixels/cm', parent=self)
                self.calibration_clicks = []
            self.show()
            return
        if self.mode.get() == 'Medir':
            if not self.data.px_per_cm:
                return
            self.measurement_clicks.append((x, y))
            if len(self.measurement_clicks) == 2:
                name = simpledialog.askstring(tr(self.lang, 'segment_name'), tr(self.lang, 'segment_prompt'), parent=self)
                if name:
                    color = choose_color(self, self.lang)
                    self.data.manual_segments.append({'name': name, 'p1': self.measurement_clicks[0], 'p2': self.measurement_clicks[1], 'color': color})
                    self.push_action({'type': 'manual'})
                self.measurement_clicks = []
                self.mode.set('Editar')
                self.show()
            return
        if self.mode.get() == 'MedirPontos':
            if not self.data.px_per_cm:
                return
            points = self.data.landmarks[self.index]
            if points is None:
                return
            distances = np.linalg.norm(points[:, :2] - np.array([x, y]), axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > 35:
                return
            self.measurement_landmarks.append(nearest)
            if len(self.measurement_landmarks) == 2:
                name = simpledialog.askstring(tr(self.lang, 'segment_name'), tr(self.lang, 'segment_prompt'), parent=self)
                if name:
                    color = choose_color(self, self.lang)
                    self.data.tracked_segments.append({'name': name, 'i1': self.measurement_landmarks[0], 'i2': self.measurement_landmarks[1], 'color': color})
                    self.push_action({'type': 'tracked'})
                self.measurement_landmarks = []
                self.mode.set('Editar')
                self.show()
            return
        points = self.data.landmarks[self.index]
        if points is None:
            return
        if self.selected is None:
            d = np.linalg.norm(points[:, :2] - np.array([x, y]), axis=1)
            nearest = int(np.argmin(d))
            if d[nearest] <= 35:
                self.selected = nearest
                self.info.config(text='Ponto selecionado. Clique na posicao correta.')
        else:
            landmark = self.selected
            snapshot = [(k, self.data.landmarks[k][landmark].copy()) for k in range(self.index, len(self.data.landmarks)) if self.data.landmarks[k] is not None]
            old = points[landmark, :2].copy()
            new = np.array([x, y], dtype=float)
            delta = new - old
            points[landmark, :2] = new
            points[landmark, 2] = 1.0
            self.selected = None
            self.show()
            self.tracking = True
            self.export_button.config(state='disabled')
            self.undo_button.config(state='disabled')
            self.info.config(text='Procurando marca branca e propagando a correcao...')
            threading.Thread(target=self.propagate, args=(self.index, landmark, new, delta, snapshot), daemon=True).start()

    def propagate(self, start, landmark, position, delta, snapshot):
        """CSRT segue a aparencia da bolinha; a mascara branca refina seu centro."""
        cap = cv2.VideoCapture(str(self.data.input_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, initial = cap.read()
        previous = position.copy()
        tracker = None
        if ok:
            factory = getattr(cv2, 'TrackerCSRT_create', None) or getattr(getattr(cv2, 'legacy', None), 'TrackerCSRT_create', None)
            if factory:
                tracker = factory()
                size = 36
                tracker.init(initial, (int(position[0] - size / 2), int(position[1] - size / 2), size, size))
        limit = len(self.data.landmarks)
        for k in range(start + 1, limit):
            ok, frame = cap.read()
            if not ok:
                break
            future = self.data.landmarks[k]
            if future is None:
                continue
            tracked_ok, box = tracker.update(frame) if tracker else (False, None)
            if tracked_ok:
                bx, by, bw, bh = box
                prediction = np.array([bx + bw / 2, by + bh / 2])
                radius = int(max(24, min(65, max(bw, bh) * 1.4)))
            else:
                prediction = previous
                radius = 65
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cx, cy = map(int, prediction)
            x0, x1 = (max(0, cx - radius), min(frame.shape[1], cx + radius))
            y0, y1 = (max(0, cy - radius), min(frame.shape[0], cy + radius))
            roi = hsv[y0:y1, x0:x1]
            mask = cv2.inRange(roi, np.array([0, 0, 205]), np.array([180, 75, 255]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            candidates = []
            for n in range(1, count):
                area, w, h = (stats[n, cv2.CC_STAT_AREA], stats[n, cv2.CC_STAT_WIDTH], stats[n, cv2.CC_STAT_HEIGHT])
                if 8 <= area <= 1200 and 0.55 <= w / max(h, 1) <= 1.8:
                    p = centroids[n] + np.array([x0, y0])
                    candidates.append((np.linalg.norm(p - prediction), p))
            if candidates:
                _, tracked = min(candidates, key=lambda item: item[0])
                future[landmark, :2] = tracked
                future[landmark, 2] = 1.0
                previous = tracked
            elif tracked_ok:
                future[landmark, :2] = prediction
                future[landmark, 2] = 1.0
                previous = prediction
            else:
                weight = float(np.exp(-(k - start) / 45.0))
                future[landmark, :2] += delta * weight
                future[landmark, 2] = max(future[landmark, 2], 0.75 * weight)
                previous = future[landmark, :2].copy()
        cap.release()
        self.after(0, lambda: self.tracking_done(landmark, snapshot))

    def tracking_done(self, landmark, snapshot):
        self.tracking = False
        self.export_button.config(state='normal')
        self.push_action({'type': 'point', 'landmark': landmark, 'snapshot': snapshot})
        if self.opts.view == 'Lateral':
            self.ear_centers = [ear_center(p) for p in self.data.landmarks]
        self.info.config(text='Rastreamento concluido: marca branca priorizada; estimativa automatica usada quando necessario.')
        self.show()

    def toggle(self):
        self.playing = not self.playing
        self.step()

    def step(self):
        if not self.playing:
            return
        self.index = (self.index + 1) % len(self.data.landmarks)
        self.internal_slide = True
        self.slider.set(self.index)
        self.internal_slide = False
        self.show()
        self.after(max(15, int(1000 / self.data.fps)), self.step)

    def close(self):
        self.playing = False
        self.cap.release()
        self.destroy()
