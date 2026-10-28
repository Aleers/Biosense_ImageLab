
"""Sincronização dos resultados do BioSense+ com a Cloudflare."""

import os
from pathlib import Path
from urllib.parse import quote

import requests


API_URL = (
    "https://biosense-api."
    "andreximenescapel.workers.dev"
)

LIMITE_VIDEO = 80 * 1024 * 1024
LIMITE_CSV = 2 * 1024 * 1024


class CloudSyncError(Exception):
    """Erro durante a sincronização em nuvem."""


def enviar_arquivo(url, arquivo, token, tipo):
    """Envia um arquivo usando PUT e autenticação Bearer."""

    arquivo = Path(arquivo)

    if not arquivo.is_file():
        raise CloudSyncError(
            f"Arquivo não encontrado: {arquivo.name}"
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": tipo,
        "Content-Length": str(arquivo.stat().st_size),
    }

    try:
        with arquivo.open("rb") as conteudo:
            resposta = requests.put(
                url,
                headers=headers,
                data=conteudo,
                timeout=(15, 300),
            )

        resposta.raise_for_status()
        return resposta.json()

    except requests.RequestException as exc:
        raise CloudSyncError(
            "Falha ao enviar arquivo para a Cloudflare."
        ) from exc

    except ValueError as exc:
        raise CloudSyncError(
            "A API retornou uma resposta inválida."
        ) from exc


def sincronizar_analise(video_path, csv_path):
    """Envia o MP4 e o CSV da mesma análise."""

    token = os.environ.get("BIOSENSE_API_TOKEN", "").strip()

    if not token:
        raise CloudSyncError(
            "BIOSENSE_API_TOKEN não configurado neste computador."
        )

    video = Path(video_path)
    csv = Path(csv_path)

    for arquivo in (video, csv):
        if not arquivo.is_file():
            raise CloudSyncError(
                f"Arquivo não encontrado: {arquivo.name}"
            )

    if not 0 < video.stat().st_size <= LIMITE_VIDEO:
        raise CloudSyncError(
            "Vídeo vazio ou acima do limite de 80 MB."
        )

    if not 0 < csv.stat().st_size <= LIMITE_CSV:
        raise CloudSyncError(
            "CSV vazio ou acima do limite de 2 MB."
        )

    # Primeiro envia o vídeo processado.
    resposta_video = enviar_arquivo(
        f"{API_URL}/upload/video",
        video,
        token,
        "video/mp4",
    )

    if not resposta_video.get("sucesso"):
        raise CloudSyncError(
            "A API não confirmou o envio do vídeo."
        )

    id_analise = resposta_video["id"]

    # Depois envia o CSV para a mesma análise.
    try:
        resposta_csv = enviar_arquivo(
            f"{API_URL}/upload/csv?id={quote(id_analise)}",
            csv,
            token,
            "text/csv",
        )

    except CloudSyncError as exc:
        raise CloudSyncError(
            f"Vídeo registrado como {id_analise}, "
            "mas o CSV não foi enviado. "
            "Guarde esse ID para recuperação."
        ) from exc

    if not resposta_csv.get("sucesso"):
        raise CloudSyncError(
            f"CSV não confirmado para a análise {id_analise}."
        )

    return {
        "id": id_analise,
        "video": resposta_video["arquivo"],
        "csv": resposta_csv["csv"],
        "status": "concluido",
    }
