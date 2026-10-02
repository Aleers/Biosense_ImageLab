"""Carregamento dos textos da interface e do video exportado."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


LANGUAGES = {"pt": "Portugues", "en": "English", "es": "Espanol"}
LOCALES_DIR = Path(__file__).with_name("locales")


class Translator:
    """Fornece os textos de um idioma e converte escolhas para valores internos.

    Cada idioma fica em um JSON independente. Isso deixa este modulo pequeno e
    permite revisar as traducoes sem alterar a logica do programa.
    """

    def __init__(self, language: str = "pt") -> None:
        self.language = language if language in LANGUAGES else "pt"
        self.texts = _load_language(self.language)

    def text(self, key: str) -> str:
        return self.texts.get(key, key)


@lru_cache(maxsize=None)
def _load_language(language: str) -> dict[str, str]:
    """Carrega cada JSON uma unica vez durante a execucao."""
    path = LOCALES_DIR / f"{language}.json"
    with path.open(encoding="ascii") as file:
        return json.load(file)


@lru_cache(maxsize=None)
def tr(language: str, key: str) -> str:
    """Retorna um texto ASCII pronto para ser desenhado pelo OpenCV."""
    return Translator(language).text(key)


def view_values(language: str) -> tuple[str, str]:
    return tr(language, "lateral"), tr(language, "frontal")


def side_values(language: str) -> tuple[str, str, str, str]:
    keys = ("auto", "left", "right", "both")
    return tuple(tr(language, key) for key in keys)


def canonical_view(language: str, value: str) -> str:
    return "Lateral" if value == tr(language, "lateral") else "Frontal"


def canonical_side(language: str, value: str) -> str:
    """Converte o texto exibido para o valor interno usado nos calculos."""
    displayed_values = side_values(language)
    internal_values = ("Automatico", "Esquerdo", "Direito", "Ambos")
    return dict(zip(displayed_values, internal_values)).get(value, "Automatico")
