from __future__ import annotations
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from i18n import LANGUAGES, canonical_side, canonical_view, side_values, tr, view_values
from processor import ProcessingOptions, analyze_video, render_video
from reviewer import ReviewWindow

class MarchaApp(tk.Tk):
    """Janela principal e coordenadora do fluxo analisar, revisar e exportar.

    O processamento pesado roda em threads para nao bloquear a interface. A
    fila ``events`` e o unico canal usado para atualizar widgets pela thread
    principal do Tkinter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.language_code = 'pt'
        self.language_display = tk.StringVar(value=LANGUAGES['pt'])
        self.title(f"{tr(self.language_code, 'title')} - MVP")
        self.geometry('760x500')
        self.minsize(700, 460)
        self.events: queue.Queue[tuple] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.view = tk.StringVar(value=tr('pt', 'lateral'))
        self.side = tk.StringVar(value=tr('pt', 'auto'))
        self.status = tk.StringVar(value=tr('pt', 'ready'))
        self._build()
        self.after(100, self._poll_events)

    def _build(self) -> None:
        if hasattr(self, 'root'):
            self.root.destroy()
        lang = self.language_code
        self.title(f"{tr(lang, 'title')} - MVP")
        root = self.root = ttk.Frame(self, padding=22)
        root.pack(fill='both', expand=True)
        ttk.Label(root, text=tr(lang, 'title'), font=('Segoe UI', 18, 'bold')).pack(anchor='w')
        ttk.Label(root, text=tr(lang, 'subtitle'), wraplength=690).pack(anchor='w', pady=(4, 20))
        files = ttk.LabelFrame(root, text=tr(lang, 'files'), padding=12)
        files.pack(fill='x')
        self._file_row(files, tr(lang, 'input'), self.input_path, self._choose_input, 0)
        self._file_row(files, tr(lang, 'output'), self.output_path, self._choose_output, 1)
        options = ttk.LabelFrame(root, text=tr(lang, 'settings'), padding=12)
        options.pack(fill='x', pady=14)
        ttk.Label(options, text=tr(lang, 'view')).grid(row=0, column=0, sticky='w')
        ttk.Combobox(options, textvariable=self.view, values=view_values(lang), state='readonly', width=18).grid(row=1, column=0, sticky='w', padx=(0, 30))
        ttk.Label(options, text=tr(lang, 'side')).grid(row=0, column=1, sticky='w')
        ttk.Combobox(options, textvariable=self.side, values=side_values(lang), state='readonly', width=18).grid(row=1, column=1, sticky='w', padx=(0, 30))
        ttk.Label(options, text=tr(lang, 'language')).grid(row=0, column=2, sticky='w')
        language_box = ttk.Combobox(options, textvariable=self.language_display, values=tuple(LANGUAGES.values()), state='readonly', width=16)
        language_box.grid(row=1, column=2, sticky='w')
        language_box.bind('<<ComboboxSelected>>', self._change_language)
        self.progress = ttk.Progressbar(root, mode='determinate', maximum=100)
        self.progress.pack(fill='x', pady=(8, 5))
        ttk.Label(root, textvariable=self.status, wraplength=700).pack(anchor='w')
        self.start_button = ttk.Button(root, text=tr(lang, 'process'), command=self._start)
        self.start_button.pack(anchor='e', pady=18)
        ttk.Separator(root).pack(fill='x')
        ttk.Label(root, text=tr(lang, 'warning'), foreground='#8a4b08', wraplength=700).pack(anchor='w', pady=12)

    def _file_row(self, parent, label, variable, command, row) -> None:
        ttk.Label(parent, text=label, width=16).grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky='ew', padx=8)
        ttk.Button(parent, text=tr(self.language_code, 'select'), command=command).grid(row=row, column=2)
        parent.columnconfigure(1, weight=1)

    def _change_language(self, _event=None):
        old = self.language_code
        canonical_v = canonical_view(old, self.view.get())
        canonical_s = canonical_side(old, self.side.get())
        self.language_code = next((code for code, name in LANGUAGES.items() if name == self.language_display.get()), 'pt')
        self.view.set(tr(self.language_code, 'lateral' if canonical_v == 'Lateral' else 'frontal'))
        side_key = {'Automatico': 'auto', 'Esquerdo': 'left', 'Direito': 'right', 'Ambos': 'both'}[canonical_s]
        self.side.set(tr(self.language_code, side_key))
        self.status.set(tr(self.language_code, 'ready'))
        self._build()

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(filetypes=[('Videos', '*.mp4 *.avi *.mov *.mkv'), ('Todos', '*.*')])
        if path:
            self.input_path.set(path)
            p = Path(path)
            self.output_path.set(str(p.with_name(f'{p.stem}_analisado.mp4')))

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension='.mp4', filetypes=[('MP4', '*.mp4')])
        if path:
            self.output_path.set(path)

    def _start(self) -> None:
        src, dst = (Path(self.input_path.get()), Path(self.output_path.get()))
        if not src.is_file():
            messagebox.showerror('Entrada invalida', 'Selecione um video existente.')
            return
        if not dst.name:
            messagebox.showerror('Saida invalida', 'Escolha o arquivo de saida.')
            return
        suffix = {'pt': 'pt', 'en': 'en', 'es': 'esp'}[self.language_code]
        base = dst.stem
        for old in ('_pt', '_en', '_esp'):
            if base.endswith(old):
                base = base[:-len(old)]
                break
        dst = dst.with_name(f"{base}_{suffix}{dst.suffix or '.mp4'}")
        self.output_path.set(str(dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        opts = ProcessingOptions(view=canonical_view(self.language_code, self.view.get()), side=canonical_side(self.language_code, self.side.get()), language=self.language_code)
        self.progress['value'] = 0
        self.status.set(tr(self.language_code, 'loading'))
        self.start_button.config(state='disabled')
        self.current_output, self.current_options = (dst, opts)
        self.worker = threading.Thread(target=self._run_analysis, args=(src, opts), daemon=True)
        self.worker.start()

    def _run_analysis(self, src: Path, opts: ProcessingOptions) -> None:
        try:
            data = analyze_video(src, lambda p, s: self.events.put(('progress', p, s)), opts.language)
            self.events.put(('review', data, opts))
        except Exception as exc:
            self.events.put(('error', str(exc)))

    def _export(self, review_window) -> None:
        review_window.playing = False
        review_window.cap.release()
        review_window.destroy()
        self.status.set(tr(self.language_code, 'exporting'))
        self.worker = threading.Thread(target=self._run_export, args=(review_window.data,), daemon=True)
        self.worker.start()

    def _run_export(self, data) -> None:
        try:
            csv_path = render_video(data, self.current_output, self.current_options, lambda p, s: self.events.put(('progress', p, s)))
            self.events.put(('done', self.current_output, csv_path))
        except Exception as exc:
            self.events.put(('error', str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == 'progress':
                    self.progress['value'] = event[1]
                    self.status.set(event[2])
                elif event[0] == 'review':
                    self.progress['value'] = 100
                    self.status.set(tr(self.language_code, 'detected'))
                    ReviewWindow(self, event[1], event[2], self._export)
                elif event[0] == 'done':
                    self.start_button.config(state='normal')
                    self.progress['value'] = 100
                    self.status.set(tr(self.language_code, 'done'))
                    messagebox.showinfo(tr(self.language_code, 'complete'), f'Video: {event[1]}\nCSV: {event[2]}')
                elif event[0] == 'error':
                    self.start_button.config(state='normal')
                    self.status.set(tr(self.language_code, 'failed'))
                    messagebox.showerror('Erro', event[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_events)
if __name__ == '__main__':
    MarchaApp().mainloop()
