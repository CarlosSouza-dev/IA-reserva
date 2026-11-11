"""Gera docs/Relatorio_Projeto_IA.pdf a partir de docs/relatorio.md.

Uso (na raiz do repositório):  python docs/gerar_pdf.py
Requer: pip install markdown  e o Google Chrome (ou Microsoft Edge) instalado.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

PASTA = Path(__file__).resolve().parent
ORIGEM = PASTA / "relatorio.md"
HTML = PASTA / "relatorio.html"
PDF = PASTA / "Relatorio_Projeto_IA.pdf"

CSS = """
@page { size: A4; margin: 16mm 17mm 16mm 17mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: #1F2933; }
h1 { font-size: 17pt; margin: 4px 0 2px; color: #C8102E; }
h2 { font-size: 12.5pt; margin: 14px 0 5px; border-bottom: 1.5px solid #C8102E; padding-bottom: 2px; }
.capa h2 { border: none; font-size: 11.5pt; margin: 0 0 6px; color: #1F2933; font-weight: 600; }
.capa { text-align: center; margin-bottom: 8px; }
.capa p { margin: 3px 0; }
p { margin: 4px 0 6px; text-align: justify; }
ul, ol { margin: 3px 0 6px; padding-left: 20px; }
li { margin: 1px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 8px; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border: 1px solid #CBD2D9; padding: 3px 5px; text-align: left; vertical-align: top; }
th { background: #F1F3F5; }
blockquote { margin: 5px 18px; padding: 4px 10px; background: #F7F8FA; border-left: 3px solid #C8102E; font-family: Consolas, monospace; font-size: 8.8pt; }
code { font-family: Consolas, monospace; font-size: 8.8pt; background: #F1F3F5; padding: 0 2px; }
figure { margin: 6px 0 8px; text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.5pt; color: #52606D; margin-top: 2px; }
a { color: #1D4ED8; text-decoration: none; }
"""


def navegador():
    candidatos = [
        shutil.which("chrome"), shutil.which("google-chrome"), shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidatos:
        if c and Path(c).exists():
            return c
    sys.exit("Chrome/Edge não encontrado: abra docs/relatorio.html e imprima em PDF.")


LISTA = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def preparar(texto):
    """O GitHub aceita lista logo abaixo de um parágrafo; o Python-Markdown exige
    uma linha em branco antes. Insere essa linha para o PDF sair igual."""
    linhas, saida = texto.splitlines(), []
    for linha in linhas:
        anterior = saida[-1] if saida else ""
        continua_lista = LISTA.match(anterior) or anterior.startswith((" ", "\t"))
        if LISTA.match(linha) and anterior.strip() and not continua_lista:
            saida.append("")
        saida.append(linha)
    return "\n".join(saida)


def main():
    corpo = markdown.markdown(preparar(ORIGEM.read_text(encoding="utf-8")),
                              extensions=["tables", "md_in_html", "sane_lists"])
    HTML.write_text(f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
                    f"<title>Relatório - Projeto de IA</title><style>{CSS}</style></head>"
                    f"<body>{corpo}</body></html>", encoding="utf-8")
    subprocess.run([navegador(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True,
                   capture_output=True)
    print("PDF gerado em", PDF)


if __name__ == "__main__":
    main()
