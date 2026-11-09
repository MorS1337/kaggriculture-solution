#!/usr/bin/env python3
"""Приёмка заявки ТАК, КАК ЕЁ ЗАПУСТИТ KAGGLE. Обязательный шаг перед отправкой.

Три заявки подряд ушли в ERROR, потому что я проверял агента импортом у себя, а
Kaggle грузит его иначе:

    exec(compile(raw, path, "exec"), {})        # globals ПУСТОЙ

Из этого следуют две ловушки, обе тихие:

1. `__file__` в модуле НЕ ОПРЕДЕЛЁН. Любое `Path(__file__)` на верхнем уровне
   валит загрузку с NameError ещё до создания агента. Настоящий путь лежит в
   code object -- `func.__code__.co_filename`.
2. Загрузчик возвращает ПОСЛЕДНИЙ вызываемый объект модуля:
   `[v for v in env.values() if callable(v)][-1]`. Значит `agent` должен быть
   определён последним; импортированный ниже класс или функция уведут точку
   входа.

Ни то, ни другое не видно при обычном `import main`. Поэтому проверяем ровно
то, что уедет: распаковываем АРХИВ, грузим штатным загрузчиком, играем полный
эпизод штатным раннером.

    python3 accept_submission.py submissions/entity_bc_966/submission.tar.gz
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def accept(tar, opponent="starter"):
    from kaggle_environments import make
    from kaggle_environments.agent import get_last_callable

    tar = Path(tar)
    size = tar.stat().st_size
    listing = subprocess.run(["tar", "tzf", str(tar)], capture_output=True,
                             text=True, check=True).stdout.split()
    pyc = [n for n in listing if "pycache" in n or n.endswith(".pyc")]
    d = Path(tempfile.mkdtemp(prefix="accept_"))
    ok = True
    try:
        subprocess.run(["tar", "xzf", str(tar), "-C", str(d)], check=True)
        main = d / "main.py"
        if not main.exists():
            print("  ПРОВАЛ: в архиве нет main.py")
            return False
        print(f"  архив         : {size:,} байт, {len(listing)} записей, "
              f"pycache {len(pyc)}")
        try:
            fn = get_last_callable(main.read_text(), path=str(main))
        except Exception as e:
            print(f"  ПРОВАЛ загрузки: {e!r}")
            return False
        print(f"  точка входа   : {fn.__name__}")
        # Kaggle uses the *last callable* returned by its loader, not its name.
        # A wrapper may legitimately be named e.g. `shadow_roundtrip_agent`.
        if fn.__name__ not in ("agent", "kaggle_submission_agent"):
            print("  note: entry point is not named `agent`; Kaggle uses the last callable")
        t0 = time.time()
        env = make("kaggriculture", debug=True)
        res = env.run([str(main), opponent])
        el = time.time() - t0
        errs = [s["info"] for st in res for s in st if s.get("info")]
        last = res[-1]
        print(f"  эпизод        : {len(res)} ходов за {el:.0f} с "
              f"(предел 1200 с на партию)")
        print(f"  статусы       : {[s['status'] for s in last]}")
        print(f"  награды       : {[s['reward'] for s in last]}")
        print(f"  ошибок        : {len(errs)}")
        for e in errs[:3]:
            print(f"     {str(e)[:160]}")
        if len(res) < 719 or any(s["status"] != "DONE" for s in last) or errs:
            ok = False
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tarballs", nargs="+")
    ap.add_argument("--opponent", default="starter")
    a = ap.parse_args()
    bad = []
    for t in a.tarballs:
        print(f"\n=== {t}")
        if not accept(t, a.opponent):
            bad.append(t)
    print()
    if bad:
        print("НЕ ПРОШЛИ ПРИЁМКУ:", *bad, sep="\n  ")
        sys.exit(1)
    print("ВСЕ ПРОШЛИ. Отправлять может пользователь.")


if __name__ == "__main__":
    main()
