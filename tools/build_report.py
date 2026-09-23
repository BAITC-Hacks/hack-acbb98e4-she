"""Build a local, self-contained demo report from an actual agent run."""

import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fmt(value):
    return f"{value:,.0f}".replace(",", " ")


def main():
    from agent import Agent
    from mock_environment import make_mock_env
    from plan_validation import audit_plan

    env, _ = make_mock_env(seed=42)
    agent = Agent()
    campaigns = agent.act(env)
    audit = audit_plan(campaigns, env.customer_profile, env.tariffs, env.channels,
                       env.remaining_budget, env.remaining_contacts)
    if not audit["valid"]:
        raise RuntimeError("Invalid final plan: " + "; ".join(audit["errors"]))
    pilot_cost = sum(float(p.get("cost", 0)) for p in env.pilot_history)
    pilot_contacts = sum(int(p.get("n_customers", 0)) for p in env.pilot_history)
    final_cost = audit["communication_cost"]
    total_cost = pilot_cost + final_cost
    total_contacts = pilot_contacts + audit["total_contacts"]
    # Public environment balances already include pilot spending. Subtract
    # only the final plan here, so pilots are not charged twice.
    remaining_budget = env.remaining_budget - final_cost
    remaining_contacts = env.remaining_contacts - audit["total_contacts"]
    rows = []
    for campaign, detail in zip(campaigns, audit["campaigns"]):
        filters = ", ".join(f"{k.removeprefix('filter_')}: {v}" for k, v in campaign.items()
                            if k.startswith("filter_") and v is not None)
        row = [campaign.get("campaign_name", ""), filters, campaign["target_tariff"],
               campaign["channel"], fmt(detail["contacts"]), fmt(detail["communication_cost"])]
        rows.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in row) + "</tr>")
    benchmark_file = ROOT / "reports/benchmark.json"
    benchmark_html = "<p>Сравнение пока не запускалось.</p>"
    benchmark = None
    if benchmark_file.is_file():
        benchmark = json.loads(benchmark_file.read_text())
        benchmark_html = '<div class="compare">'
        for key, label in (("template", "Шаблон организаторов"), ("adaptive", "Адаптивный агент")):
            values = benchmark["agents"][key]
            benchmark_html += (f'<article><h3>{label}</h3><p class="number">{fmt(values["mean"])}</p>'
                               '<p>средний чистый прирост ARPU</p>'
                               f'<p>Минимум: {fmt(values["min"])} · Медиана: {fmt(values["median"])}</p>'
                               f'<p>Положительных прогонов: {values["positive_runs"]} / '
                               f'{len(values["runs"])}</p></article>')
        benchmark_html += '</div>'
    strategy = getattr(agent, "last_report", {})
    report = {"seed": 42, "campaigns": campaigns, "audit": audit,
              "pilot_contacts": pilot_contacts, "pilot_cost": pilot_cost,
              "total_contacts": total_contacts, "total_cost": total_cost,
              "remaining_budget_after_plan": remaining_budget,
              "remaining_contacts_after_plan": remaining_contacts,
              "strategy": strategy}
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    (output / "plan_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    page = """<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>She++ · План тарифных кампаний</title>
<style>
:root{color-scheme:light;--ink:#142027;--muted:#56606b;--line:#dfe4e5;--yellow:#ffd328}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 system-ui,-apple-system,sans-serif;background:#f4f6f5;color:var(--ink)}
main{max-width:1220px;margin:auto;padding:42px 28px 60px}header{background:var(--ink);color:white;border-radius:24px;padding:36px;margin-bottom:24px}
.tag{font-size:13px;letter-spacing:2px;text-transform:uppercase;color:var(--yellow)}h1{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:16px 0}
header p{color:#d8e0e2;max-width:800px}section{background:white;border:1px solid var(--line);border-radius:18px;padding:26px;margin-top:22px}
h2{font-size:22px;margin:0 0 14px}h3{font-size:16px}.stats,.compare{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}
.stats article{background:white;border:1px solid var(--line);padding:20px;border-radius:16px}.number{font-size:28px;font-weight:750;line-height:1.2;margin:6px 0}
.sub{color:var(--muted);font-size:14px}.compare{grid-template-columns:1fr 1fr}.compare article{background:#f5f7f6;padding:20px;border-radius:12px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;text-align:left;font-size:14px}th{background:#f3f5f4;color:var(--muted)}th,td{padding:12px;border-bottom:1px solid var(--line);vertical-align:top}
.ok{color:#166445;font-weight:650}.notice{background:#fff6c9;padding:14px 18px;border-radius:12px}input,button{font:inherit;border:1px solid #9ca9a9;border-radius:8px;padding:9px 12px;background:white}input{min-width:260px;margin-bottom:16px}button{cursor:pointer;float:right}a{color:#075e77}pre{overflow:auto;max-height:460px;padding:16px;background:#f5f7f6;font-size:12px}
footer{color:var(--muted);font-size:13px;margin-top:24px}@media(max-width:760px){main{padding:20px 12px}.stats{grid-template-columns:1fr 1fr}.compare{grid-template-columns:1fr}header,section{padding:20px}.number{font-size:23px}}
@media print{button,input{display:none}body{background:white}main{padding:0}section{break-inside:avoid}header{color:black;background:white;border:1px solid #ccc}header p{color:black}}
</style><main>
<header><div class="tag">She++ / Campaign Lab / локальная демонстрация</div>
<h1>План кампаний,<br>подтверждённый пилотами</h1>
<p>Сегменты → гипотезы → пилоты → оценка неопределённости → кампании в пределах ресурсов.</p>
<p>Синтетический кейс Beeline. Результаты относятся к выданной мок-среде; это не прогноз оценки жюри.</p></header>
<div class="stats">
<article><div class="sub">Финальные кампании</div><div class="number">__COUNT__ / 10</div><div class="sub">корректность плана проверена</div></article>
<article><div class="sub">Расходы, у.е.</div><div class="number">__COST__</div><div class="sub">из 100 000, включая пилоты</div><p class="sub">Остаток после всех кампаний: <strong>__REMAINING_BUDGET__ у.е.</strong></p></article>
<article><div class="sub">Контакты</div><div class="number">__CONTACTS__</div><div class="sub">из 15 000, включая повторы и пилоты</div><p class="sub">Остаток после всех кампаний: <strong>__REMAINING_CONTACTS__</strong></p></article>
<article><div class="sub">Пилоты</div><div class="number">__PILOTS__ / 20</div><div class="sub">фактически выполнены, seed 42</div></article>
</div>
<section><h2>План для аналитика <button onclick="window.print()">Печать / PDF</button></h2>
<p class="ok">Независимый аудит пройден: схемы, фильтры, размеры аудиторий и остатки ресурсов.</p>
<input id="search" aria-label="Поиск по кампаниям" placeholder="Найти тариф, сегмент или канал…">
<div class="scroll"><table><thead><tr><th>Кампания</th><th>Сегмент</th><th>Целевой тариф</th><th>Канал</th><th>Контакты</th><th>Расходы, у.е.</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="sub">Уникальных абонентов финального плана: __UNIQUE__. Пилотные пересечения не вычитаются из контактов; дедупликацию выручки выполняет официальный оценщик.</p></section>
<section><h2>Сравнение на одинаковых seed</h2><p>Фактический чистый прирост в локальном оценщике после затрат, включая пилоты.</p>__BENCHMARK__
<p class="notice">На судействе эффекты другие. Положительные результаты здесь проверяют механику и устойчивость на мок-среде, но не гарантируют прибыль в иной среде.</p></section>
<section><h2>Почему приняты эти решения</h2><p>Журнал агента содержит наблюдения пилотов и оценки риска. Эти оценки не являются истинными эффектами среды.</p>
<details><summary>Показать журнал принятия решений</summary><pre>__STRATEGY__</pre></details>
<p><a href="plan_audit.json">Открыть машинный отчёт JSON</a> · <a href="../submission.csv">submission.csv</a> · <a href="benchmark.json">Результаты сравнения</a></p></section>
<footer>Отчёт создан командой <code>python tools/build_report.py</code> на исходных синтетических данных. Внешние сервисы и отправка маркетинговых сообщений не используются.</footer>
</main><script>document.getElementById('search').addEventListener('input',function(){const q=this.value.toLocaleLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLocaleLowerCase().includes(q));});</script></html>"""
    values = {"__COUNT__": str(len(campaigns)), "__COST__": fmt(total_cost),
              "__CONTACTS__": fmt(total_contacts), "__PILOTS__": str(len(env.pilot_history)),
              "__REMAINING_BUDGET__": fmt(remaining_budget),
              "__REMAINING_CONTACTS__": fmt(remaining_contacts),
              "__ROWS__": "\n".join(rows), "__UNIQUE__": fmt(audit["unique_contacts"]),
              "__BENCHMARK__": benchmark_html,
              "__STRATEGY__": html.escape(json.dumps(strategy, ensure_ascii=False, indent=2))}
    for key, value in values.items():
        page = page.replace(key, value)
    (output / "campaign_report.html").write_text(page)
    print("Created reports/campaign_report.html and reports/plan_audit.json")


if __name__ == "__main__":
    main()
