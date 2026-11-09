#!/usr/bin/env python3
"""Собрать `recon/pubagents/` из опубликованных ноутбуков Kaggle.

Каталог намеренно в gitignore: это чужой код (Apache-2.0), и перевыкладывать
его в наш репозиторий не нужно. Скрипт восстанавливает его из первоисточника.

Что достаётся:
  * ячейки `%%writefile <имя>` -- так авторы кладут `main.py`;
  * ячейки, которые распаковывают приложенные данные (`actions.json` с 13
    лентами у yhay81) -- исполняются в каталоге агента.

Ячейки исполняются, поэтому запускать только на доверенных ноутбуках из списка
`KERNELS` и понимая, что это чужой код.

    python3 recon/fetch_pubagents.py            # все
    python3 recon/fetch_pubagents.py yhay81     # по подстроке
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "recon" / "pubagents"
KERNELS = (
    # Срез от 9 сентября. К 12-му он протух: живое поле на 2350+ бьёт нашу
    # ленту в 85% партий, тогда как против этой четвёрки она даёт 96%.
    "yhay81/shop-router-0909",
    "aurax7/kaggriculture-reactive-router",
    "thomastschinkel/kaggriculture-public-state-router-74-5-win-rate",
    # Срез от 12 сентября: то, что опубликовано и набрало голоса за эти трое
    # суток. Панель без них меряет силу против поля, которого уже нет.
    "pilkwang/kaggriculture-structured-economic-policy",
    "tetsutani/market-smart-farming-kaggriculture",
    "reyhanksatria/kaggriculture-dynamic-route-agent",
    "guruprasaathas111/kaggriculture-master-engine-v3",
    "municef1/lb-2448-single-file-kaggriculture-agent",
    # Срез от 15 сентября: панель к оценке снимков BC expansion. За трое суток
    # вышли новые версии тех же авторов и несколько новых с голосами.
    "ahmedberatozer/kaggriculture-v43-recovering-lost-harvests",
    "ahmedberatozer/kaggriculture-v41-review-candidate",
    "aurax7/kaggriculture-shop-router-reactive-v5",
    "lynnsakurai/farming-score-v5-timing-optimized",
    "lynnsakurai/farming-score-v4-a-better-shop",
    "prvsiyan/kaggriculture-frontier-the-moon-counts-melons",
    "prvsiyan/kaggriculture-frontier-the-soil-remembers-rain",
    "kunaldesale2408/kaggriculture-2026-v1",
    "alperen5252525/kaggriculture-metacounter-r1-scored-agent",
    # Первоисточник прежнего «СИЛЬНОГО» 94c1c2c0 (V37). reyhanksatria,
    # guruprasaathas111 и tetsutani к 15.09 перезалили свои копии на V41 и
    # reactive-v5, так что этот ноутбук -- единственный живой. Хеш сходится
    # только после замены CRLF на LF, как делает сам ноутбук.
    "ahmedberatozer/more-yield-smarter-labor",
    # Срез от 21 сентября, вечер: опубликовано ПОСЛЕ утреннего сбора (13:51).
    # Кандидат лидировал в пуле 13:51, но панель стареет за сутки -- это и есть
    # проверка, не появилось ли шасси сильнее.
    "prvsiyan/kaggriculture-frontier-the-soil-remembers-rain",
    "ahmedberatozer/kaggriculture-v56-smarter-seeds-and-fertilizer",
    "flexonafft/kaggriculture-multi-route-farming-agent",
    "hakdevelopment/kaggriculture-2887-score-fieldcraft-agent",
    "romantamrazov/kaggriculture-yummers",
    "kenanzhang9/kaggriculture-a1-t31-guard",
    "xuanzhang001/kaggriculture-auto-top1",
    "lynnsakurai/farmer-john-and-the-wheat-seller",
    "dmitriigluzdov/kaggriculture-more-wheat-smarter-sales",
    # Ниже -- не через %%writefile: V43 кладёт исходник байтовыми литералами,
    # prvsiyan строкой AGENT_SOURCE, reyhanksatria/guruprasaathas111 base85+zlib,
    # alperen5252525 публикует сам main.py скриптом. Разобраны вручную 15.09
    # через ast.literal_eval со сверкой объявленных хешей, без исполнения ячеек.
    # lime0001/score2710 -- побайтовая копия metacounter плюс одна строка.
)
# Ячейки, которые пишут файлы не через %%writefile, а кодом. Ключ -- подстрока
# ноутбука, значение -- признак нужной ячейки.
UNPACK = {"yhay81": "data_payload"}


def fetch(ref):
    d = OUT / ref.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    raw = REPO / "recon" / "kernels" / ref.replace("/", "_")
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "pull", ref, "-p", str(raw)], check=True)
    nb_path = next(raw.glob("*.ipynb"))
    nb = json.loads(nb_path.read_text())
    wrote = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        m = re.match(r"\s*%%writefile\s+(\S+)\s*\n", src)
        if m:
            (d / m.group(1)).write_text(src[m.end():])
            wrote.append(m.group(1))
            continue
        mark = next((v for k, v in UNPACK.items() if k in ref), None)
        if mark and mark in src:
            # Ячейка пишет файлы в текущий каталог -- исполняем в каталоге агента.
            cwd = Path.cwd()
            try:
                import os
                os.chdir(d)
                exec(compile(src, f"{ref}:unpack", "exec"), {})
            finally:
                os.chdir(cwd)
            wrote.append("(распаковка)")
    print(f"{ref} -> {d.relative_to(REPO)}: {', '.join(wrote) or 'ничего'}")
    for f in sorted(d.iterdir()):
        print(f"    {f.name:<16}{f.stat().st_size:>12,} байт")


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    todo = [k for k in KERNELS if pat in k]
    if not todo:
        raise SystemExit(f"ничего не подошло под {pat!r}; есть: {KERNELS}")
    for ref in todo:
        fetch(ref)


if __name__ == "__main__":
    main()
