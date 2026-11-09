#!/usr/bin/env python3
"""Где именно клон сходит с ленты: разбор ОДНОЙ партии по ходам.

Зачем отдельный инструмент. `vs_agent.py` отвечает «проиграл на 40 тысяч», но не
говорит, на каком ходу и что было сделано не так. E91 показал, что расширение
корпуса чинит девять катастроф из одиннадцати, а две (посевы 990017 и 990028)
переживают и узкий, и широкий корпус -- значит там причина другая, и её надо
увидеть глазами.

Конструкция, которая ТУТ важна. Нельзя прогнать два эпизода -- клон против
соперника и учитель против соперника -- и сравнивать их по номеру хода:
состояния разойдутся с первого же различия, и дальше сравниваются две разные
партии, а не два решения. Поэтому эпизод ОДИН: играет клон, а учитель
опрашивается тенью на тех же самых состояниях и никак на игру не влияет.
Учитель -- лента, он отвечает `tape[t]` плюс мелкие реактивные починки, и
воспроизводить эту ленту -- ровно то, чему клон обучался.

    python3 divergence.py --agent weights/agents/bc-wide-ep8 --seed 990017
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from vs_agent import _load  # noqa: E402


def _cmd(c, head=""):
    """Одна команда к смысловому виду.

    Клон пишет ['PICKUP', 'COW', 1] там, где лента пишет ['PICKUP', 'COW'] --
    явная единица вместо подразумеваемой. Для фермера и рук это ОДНО И ТО ЖЕ:
    `n = int(action[2]) if len(action) >= 3 else 1`. Сравнение по repr без этой
    поправки раздувало счётчик до 18% на партии, которую клон ВЫИГРАЛ и в
    которой банки шли ход в ход.

    Для рынка снимать единицу НЕЛЬЗЯ. Там разбор требует трёх элементов:
    `if len(order) < 3: return None`, то есть ['SELL','X',1] -- настоящая
    заявка, а ['SELL','X'] -- мёртвый слот. Приравняв их, я приравнивал сделку
    к пустышке. И по той же причине не выбрасывается 'PASS': он тоже занимает
    слот, а заявки исполняются по позициям в лок-шаге с соперником.
    """
    if isinstance(c, (list, tuple)):
        c = list(c)
        if head != "market":
            while len(c) > 1 and c[-1] in (1, "1"):
                c = c[:-1]
        return tuple(map(repr, c))
    return repr(c)


def _parts(a):
    """Разложить по головам: поле, руки, рынок."""
    if not isinstance(a, dict):
        return {}
    out = {}
    for k, v in a.items():
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
            out[k] = tuple(_cmd(c, k) for c in v)   # список команд (руки, рынок)
        else:
            out[k] = _cmd(v, k)                     # одна команда (фермер)
    return out


def _norm(a):
    """Действие целиком к сравнимому виду."""
    if not isinstance(a, dict):
        return ("НЕ_СЛОВАРЬ", repr(a)[:80])
    return tuple(sorted(_parts(a).items()))


def run(agent_spec, opp_spec, teacher_spec, seed, seat, steps=719):
    import kaggle_environments.envs.kaggriculture.kaggriculture as K
    from recon.fastenv import new_episode, observation_for

    me = _load(agent_spec, None, "me")
    opp = _load(opp_spec, None, "opp")
    teacher = _load(teacher_spec, None, "tea")

    env, state = new_episode(seed)
    rows = []
    for step in range(steps):
        obs = observation_for(state, seat, step)
        a_me = me(obs)
        a_te = teacher(obs)          # тень: на игру не влияет
        farms = state[0].observation.farms
        rows.append({
            "step": step,
            "same": _norm(a_me) == _norm(a_te),
            "nme": _norm(a_me),
            "nte": _norm(a_te),
            "me": _parts(a_me),
            "teacher": _parts(a_te),
            "bank_me": float(farms[seat]["money"]),
            "bank_opp": float(farms[1 - seat]["money"]),
        })
        state[seat].action = a_me
        state[1 - seat].action = opp(observation_for(state, 1 - seat, step))
        state[0].observation.step = step
        K.interpreter(state, env)

    farms = state[0].observation.farms
    return rows, float(farms[seat]["money"]), float(farms[1 - seat]["money"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="weights/agents/bc-wide-ep8")
    ap.add_argument("--opp", default="recon/pubagents/yhay81_shop-router-0909")
    ap.add_argument("--teacher", default="build/market_agent")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--show", type=int, default=12,
                    help="сколько первых расхождений напечатать подробно")
    a = ap.parse_args()

    rows, us, them = run(a.agent, a.opp, a.teacher, a.seed, a.seat)
    diff = [r for r in rows if not r["same"]]
    print(f"посев {a.seed}, место {a.seat}: наш банк {us:,.0f}, "
          f"соперник {them:,.0f}, разрыв {us - them:+,.0f}")
    print(f"ходов {len(rows)}, расхождений с лентой {len(diff)} "
          f"({100 * len(diff) / len(rows):.1f}%)")
    if not diff:
        print("клон повторил ленту ход в ход -- причина не в действиях")
        return

    # По каким головам расходится
    by_head = {}
    for r in diff:
        for k in set(r["me"]) | set(r["teacher"]):
            if r["me"].get(k) != r["teacher"].get(k):
                by_head[k] = by_head.get(k, 0) + 1
    print("расхождения по головам: " +
          ", ".join(f"{k} {v}" for k, v in sorted(by_head.items(),
                                                  key=lambda x: -x[1])))

    # Причина или следствие. Лента индексируется номером хода и не реагирует:
    # стоит клону оказаться в другом состоянии, как её действия перестают быть
    # применимы, и расхождение возникает САМО, как следствие проигрыша. Отличить
    # можно только порядком во времени -- что было раньше, расхождение или
    # разъезд банков. Так же был разоблачён недокорм в E88.
    gap = next((r["step"] for r in rows
                if abs(r["bank_me"] - r["bank_opp"]) > 1000), None)

    # По головам отдельно. Общее «первое расхождение» у 30 партий из 32 попадало
    # на один и тот же ход 121 -- это первая рыночная заявка ленты, и она
    # забивала сигнал. Обе подлинные катастрофы расходятся на ходах 30 и 33,
    # то есть ЕЩЁ ДО рынка: у них ломается полевая игра.
    # Подстановка-пустышка событием НЕ считается. На выигранной партии клон
    # пишет ('PASS',) там, где лента пишет () -- обе записи движок разбирает в
    # None и обе занимают ту же позицию в очереди, то есть в исполнении разницы
    # нет. На катастрофе же лента ПОКУПАЕТ пшеницу, а клон не делает ничего.
    # Без этой фильтрации оба случая считаются одинаково, и аггрегат по всем
    # партиям говорит «рынок первый» независимо от исхода.
    QTY = ("SELL", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL")

    def alive(cmds):
        """Оставить только заявки, которые движок вообще исполнит."""
        out = []
        for c in (cmds or ()):
            if not c:
                continue
            op = c[0].strip("'")
            if op in ("HIRE", "BUY_LAND"):
                out.append(c)
            elif op in QTY and len(c) >= 3 and c[2].strip("'") not in ("0",):
                out.append(c)
        return tuple(out)

    firsts = {}
    for head in ("farmer", "hands", "market"):
        firsts[head] = next((r["step"] for r in rows
                             if r["me"].get(head) != r["teacher"].get(head)), None)
    firsts["market_живые"] = next(
        (r["step"] for r in rows
         if alive(r["me"].get("market")) != alive(r["teacher"].get("market"))), None)
    print("\nПОРЯДОК: первое расхождение "
          + ", ".join(f"{h} {v if v is not None else '-'}"
                      for h, v in firsts.items())
          + f", разрыв банков свыше 1000 ход {gap if gap is not None else 'не было'}")

    print(f"первое расхождение на ходу {diff[0]['step']} "
          f"(день {diff[0]['step'] // 24}, час {diff[0]['step'] % 24})")
    for r in diff[:a.show]:
        print(f"\n  ход {r['step']:3d} день {r['step']//24:2d} час {r['step']%24:2d}"
              f"  банк {r['bank_me']:,.0f} против {r['bank_opp']:,.0f}")
        for k in sorted(set(r["me"]) | set(r["teacher"])):
            if r["me"].get(k) != r["teacher"].get(k):
                print(f"    {k}:")
                print(f"      клон:    {r['me'].get(k)}")
                print(f"      учитель: {r['teacher'].get(k)}")

    # Отстал ли клон по фазе. Две болезни выглядят похоже в распечатке ходов:
    # «делает не то» и «делает то же, но с опозданием». Различает их только
    # сравнение потоков со сдвигом: если при лаге k совпадение резко выше, чем
    # при нуле, значит клон исполняет ту же ленту, потеряв такт.
    nme = [r["nme"] for r in rows]
    nte = [r["nte"] for r in rows]
    print("\nсовпадение потоков при сдвиге (лаг: доля совпавших ходов):")
    best = None
    for lag in range(-5, 6):
        pairs = [(nme[t], nte[t + lag]) for t in range(len(rows))
                 if 0 <= t + lag < len(rows)]
        hit = sum(1 for x, y in pairs if x == y) / max(1, len(pairs))
        mark = ""
        if best is None or hit > best[1]:
            best = (lag, hit)
        print(f"  лаг {lag:+d}: {100*hit:5.1f}%{mark}")
    print(f"  лучший сдвиг {best[0]:+d} при {100*best[1]:.1f}%")

    # Рыночные слоты. Заявки режутся по maxMarketOrdersPerTurn = 10 и
    # исполняются ПО ПОЗИЦИЯМ в лок-шаге с соперником, поэтому лишняя запись в
    # начале списка сдвигает все последующие, а на пределе -- выбивает
    # последнюю настоящую заявку.
    def mlen(key):
        return [len(r[key].get("market", ())) for r in rows]
    lm, lt = mlen("me"), mlen("teacher")
    cap_me = sum(1 for x in lm if x >= 10)
    cap_te = sum(1 for x in lt if x >= 10)
    print(f"\nрыночных заявок за ход: клон в среднем {sum(lm)/len(lm):.2f}, "
          f"максимум {max(lm)}, на пределе (10) {cap_me} ходов")
    print(f"                        лента в среднем {sum(lt)/len(lt):.2f}, "
          f"максимум {max(lt)}, на пределе (10) {cap_te} ходов")

    # Выдуманный словарь: что клон произносит, а лента за всю партию -- ни разу.
    def vocab(key):
        out = set()
        for r in rows:
            for head, v in r[key].items():
                cmds = v if v and isinstance(v[0], tuple) else (v,)
                for c in cmds:
                    if c:
                        out.add((head,) + tuple(c[:2]))
        return out

    tv = vocab("teacher")
    mine = {}
    for r in rows:
        for head, v in r["me"].items():
            cmds = v if v and isinstance(v[0], tuple) else (v,)
            for c in cmds:
                if c and (head,) + tuple(c[:2]) not in tv:
                    k = (head,) + tuple(c[:2])
                    mine[k] = mine.get(k, 0) + 1
    if mine:
        print("\nдействия, которых лента не делает НИ РАЗУ за партию:")
        for k, n in sorted(mine.items(), key=lambda x: -x[1])[:12]:
            print(f"  {n:4d} x {' '.join(map(str, k))}")
    else:
        print("\nвыдуманных действий нет: словарь клона -- подмножество ленты")

    # Где разъехались банки
    print("\nбанк по дням (наш / соперник / разрыв):")
    for d in range(0, 30, 2):
        r = rows[min(d * 24, len(rows) - 1)]
        print(f"  день {d:2d}  {r['bank_me']:9,.0f}  {r['bank_opp']:9,.0f}  "
              f"{r['bank_me'] - r['bank_opp']:+9,.0f}")


if __name__ == "__main__":
    main()
