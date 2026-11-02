
"""Consulta e download de análises do BioSense+ ImageLab."""

import os
from pathlib import Path

import requests

from cloud_sync import API_URL


class CloudHistoryError(Exception):
    """Erro ao consultar ou recuperar uma análise."""


def obter_headers():
    """Obtém a autenticação configurada no computador."""
    token = os.environ.get("BIOSENSE_API_TOKEN", "").strip()

    if not token:
        raise CloudHistoryError(
            "BIOSENSE_API_TOKEN não configurado."
        )

    return {
        "Authorization": f"Bearer {token}"
    }


def listar_analises():
    """Consulta as 50 análises mais recentes no D1."""

    try:
        resposta = requests.get(
            f"{API_URL}/analises",
            headers=obter_headers(),
            timeout=30
        )

        resposta.raise_for_status()
        dados = resposta.json()

        if not dados.get("sucesso"):
            raise CloudHistoryError(
                "A API não confirmou a consulta."
            )

        return dados.get("analises", [])

    except requests.RequestException as exc:
        raise CloudHistoryError(
            "Não foi possível consultar o histórico."
        ) from exc

    except ValueError as exc:
        raise CloudHistoryError(
            "Resposta inválida da API."
        ) from exc


def baixar_arquivo(id_analise, tipo, pasta_destino):
    """Baixa um vídeo ou CSV associado a uma análise."""

    if tipo not in ("video", "csv"):
        raise CloudHistoryError(
            "Tipo de arquivo inválido."
        )

    destino = Path(pasta_destino)
    destino.mkdir(parents=True, exist_ok=True)

    nome = "processado.mp4" if tipo == "video" else "resultados.csv"

    # Cada análise tem sua própria pasta local.
    pasta_analise = destino / id_analise
    pasta_analise.mkdir(parents=True, exist_ok=True)

    caminho = pasta_analise / nome
    temporario = pasta_analise / (nome + ".part")

    url = f"{API_URL}/analises/{id_analise}/arquivo"

    try:
        with requests.get(
            url,
            params={"tipo": tipo},
            headers=obter_headers(),
            stream=True,
            timeout=(15, 120)
        ) as resposta:

            resposta.raise_for_status()

            with temporario.open("wb") as arquivo:
                for bloco in resposta.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if bloco:
                        arquivo.write(bloco)

        # Só substitui o arquivo final após o download.
        temporario.replace(caminho)

        return caminho

    except requests.RequestException as exc:
        raise CloudHistoryError(
            "Não foi possível baixar o arquivo."
        ) from exc

    except OSError as exc:
        raise CloudHistoryError(
            "Não foi possível salvar o arquivo no computador."
        ) from exc

    finally:
        if temporario.exists():
            temporario.unlink()
