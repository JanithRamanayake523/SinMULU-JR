"""Turn transcription_prompt.txt into appendix_prompt.tex: verbatim text, Sinhala runs in the Sinhala font.
python make_prompt_tex.py"""
import re

SINHALA = re.compile("([\u0d80-\u0dff\u200d]+)")
ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "%": r"\%", "#": r"\#", "&": r"\&",
           "$": r"\$", "^": r"\^{}", "~": r"\~{}"}


def escape(s):
    return "".join(ESCAPES.get(c, c) for c in s)


with open("transcription_prompt.txt", encoding="utf-8") as f, open("appendix_prompt.tex", "w", encoding="utf-8") as g:
    for line in f.read().splitlines():
        if not line.strip():
            g.write("\\medskip\n\n")
            continue
        body = line.lstrip(" ")
        indent = len(line) - len(body)
        parts = [r"\sinhalatext{%s}" % p if SINHALA.fullmatch(p) else escape(p) for p in SINHALA.split(body) if p]
        g.write((r"\hspace*{%.1fem}" % (indent * 0.5) if indent else "") + "".join(parts) + "\n\n")
