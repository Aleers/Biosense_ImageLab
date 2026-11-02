
"""Janela de histórico de análises do BioSense+ ImageLab."""

import queue
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from cloud_history import listar_analises, baixar_arquivo


UUID_REGEX = re.compile(
    r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$",
    re.IGNORECASE
)


class HistoryWindow(tk.Toplevel):
    """Consulta e permite baixar análises armazenadas na nuvem."""

    def __init__(self, parent):
        super().__init__(parent)

        self.title("BioSense+ - Histórico de Análises")
        self.geometry("950x550")
        self.minsize(780, 430)

        self.events = queue.Queue()
        self.busy = False
        self.analises = {}

        self._build()

        self.after(100, self._poll_events)
        self._carregar_historico()

    def _build(self):
        root = ttk.Frame(self, padding=20)
        root.pack(fill="both", expand=True)

        ttk.Label(
            root,
            text="Histórico de Análises",
            font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")

        ttk.Label(
            root,
            text="Selecione uma análise para baixar seus arquivos.",
        ).pack(anchor="w", pady=(5, 15))

        # Área que contém a tabela e a barra de rolagem
        area = ttk.Frame(root)
        area.pack(fill="both", expand=True)

        colunas = (
            "data",
            "nome",
            "tipo",
            "status",
            "arquivos"
        )

        # Tabela criada diretamente dentro da área
        self.tabela = ttk.Treeview(
            area,
            columns=colunas,
            show="headings",
            selectmode="browse",
            height=14
        )

        titulos = {
            "data": "Data",
            "nome": "Vídeo",
            "tipo": "Tipo",
            "status": "Status",
            "arquivos": "Arquivos disponíveis"
        }

        larguras = {
            "data": 170,
            "nome": 210,
            "tipo": 100,
            "status": 120,
            "arquivos": 150
        }

        for coluna in colunas:
            self.tabela.heading(
                coluna,
                text=titulos[coluna]
            )

            self.tabela.column(
                coluna,
                width=larguras[coluna],
                minwidth=90,
                anchor="center" if coluna != "nome" else "w"
            )

        self.tabela.pack(
            side="left",
            fill="both",
            expand=True
        )

        barra = ttk.Scrollbar(
            area,
            orient="vertical",
            command=self.tabela.yview
        )
        barra.pack(side="right", fill="y")

        self.tabela.configure(
            yscrollcommand=barra.set
        )

        # Área dos botões
        botoes = ttk.Frame(root)
        botoes.pack(fill="x", pady=(15, 10))

        self.btn_atualizar = ttk.Button(
            botoes,
            text="Atualizar histórico",
            command=self._carregar_historico
        )
        self.btn_atualizar.pack(side="left", padx=(0, 8))

        self.btn_video = ttk.Button(
            botoes,
            text="Baixar MP4",
            command=lambda: self._iniciar_download("video")
        )
        self.btn_video.pack(side="left", padx=8)

        self.btn_csv = ttk.Button(
            botoes,
            text="Baixar CSV",
            command=lambda: self._iniciar_download("csv")
        )
        self.btn_csv.pack(side="left", padx=8)

        self.status = tk.StringVar(
            value="Consultando histórico..."
        )

        ttk.Label(
            root,
            textvariable=self.status
        ).pack(anchor="w")

    def _set_busy(self, busy):
        self.busy = busy
        estado = "disabled" if busy else "normal"

        for botao in (
            self.btn_atualizar,
            self.btn_video,
            self.btn_csv
        ):
            botao.configure(state=estado)

    def _carregar_historico(self):
        if self.busy:
            return

        self._set_busy(True)
        self.status.set("Consultando análises na nuvem...")

        def tarefa():
            try:
                dados = listar_analises()
                self.events.put(("lista", dados))
            except Exception as exc:
                self.events.put(("erro", str(exc)))

        threading.Thread(
            target=tarefa,
            daemon=True
        ).start()

    def _mostrar_analises(self, dados):
        # Limpa a tabela antes de atualizar
        itens = self.tabela.get_children()

        if itens:
            self.tabela.delete(*itens)

        self.analises.clear()

        for analise in dados:
            identificador = analise.get("id", "")

            if not isinstance(identificador, str):
                continue

            if not UUID_REGEX.fullmatch(identificador):
                continue

            self.analises[identificador] = analise

            arquivos = []

            if analise.get("tem_video"):
                arquivos.append("MP4")

            if analise.get("tem_csv"):
                arquivos.append("CSV")

            data = analise.get("data_criacao") or ""
            nome = analise.get("nome_video") or ""
            tipo = analise.get("tipo_analise") or ""
            status = analise.get("status") or ""

            self.tabela.insert(
                "",
                "end",
                iid=identificador,
                values=(
                    data,
                    nome,
                    tipo,
                    status,
                    " + ".join(arquivos)
                )
            )

        self.status.set(
            f"{len(self.analises)} análises encontradas. "
            "Selecione uma para baixar."
        )

    def _iniciar_download(self, tipo):
        if self.busy:
            return

        selecao = self.tabela.selection()

        if not selecao:
            messagebox.showwarning(
                "Seleção necessária",
                "Selecione uma análise na tabela.",
                parent=self
            )
            return

        identificador = selecao[0]

        analise = self.analises.get(identificador)

        if not analise:
            return

        campo = (
            "tem_video" if tipo == "video"
            else "tem_csv"
        )

        if not analise.get(campo):
            messagebox.showwarning(
                "Arquivo indisponível",
                "Esta análise não possui esse arquivo.",
                parent=self
            )
            return

        pasta = filedialog.askdirectory(
            parent=self,
            title="Escolha onde salvar o arquivo"
        )

        if not pasta:
            return

        self._set_busy(True)
        self.status.set("Baixando arquivo da Cloudflare...")

        def tarefa():
            try:
                caminho = baixar_arquivo(
                    identificador,
                    tipo,
                    Path(pasta)
                )

                self.events.put((
                    "download",
                    str(caminho)
                ))

            except Exception as exc:
                self.events.put((
                    "erro",
                    str(exc)
                ))

        threading.Thread(
            target=tarefa,
            daemon=True
        ).start()

    def _poll_events(self):
        if not self.winfo_exists():
            return

        try:
            while True:
                tipo, dados = self.events.get_nowait()

                self._set_busy(False)

                if tipo == "lista":
                    self._mostrar_analises(dados)

                elif tipo == "download":
                    self.status.set("Download concluído.")

                    messagebox.showinfo(
                        "Download concluído",
                        f"Arquivo salvo em:\n{dados}",
                        parent=self
                    )

                elif tipo == "erro":
                    self.status.set("Operação não concluída.")

                    messagebox.showerror(
                        "Erro no histórico",
                        dados,
                        parent=self
                    )

        except queue.Empty:
            pass

        if self.winfo_exists():
            self.after(100, self._poll_events)
