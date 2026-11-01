#!/usr/bin/env python3
"""Build a local HTML gallery from existing experiment artifacts; stdlib only.

No model loading, training, server connection, or web dependencies.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


def build_gallery(run, out=None, diagnostics=None):
    run = Path(run).resolve()
    if not (run/"summary.json").is_file():
        raise FileNotFoundError(f"Expected an experiment directory with summary.json: {run}")
    out = Path(out).resolve() if out else run/"gallery.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run/"summary.json").read_text())
    def href(path):
        return quote(os.path.relpath(path, out.parent), safe="/")
    def media(path, caption):
        url = html.escape(href(path), quote=True)
        tag = (f'<video controls loop muted playsinline preload="metadata" src="{url}"></video>'
               if path.suffix == ".mp4" else f'<a href="{url}"><img loading="lazy" src="{url}"></a>')
        return f'<figure>{tag}<figcaption>{html.escape(caption)}</figcaption></figure>'
    sections = []
    corrected = summary.get("task_family") == "single_goal"
    sections.append('<p><b>'+('Одна цель, несколько примитивных переходов; чистый рендерер.' if corrected else
        'Старая отладочная постановка возле цели. Этот SR не проверяет подход на несколько клеток. '
        '«!» в старых кадрах обозначает пустой карман/поле рендерера.')+'</b></p>')
    rows = []
    for model in summary["models"]:
        sr = model["sr"]
        rows.append(f"<tr><td>{html.escape(model['mode'])}</td><td>{model['horizon']}</td>"+
                    "".join(f"<td>{sr[s]:.1%}</td>" for s in ("train","val","test"))+"</tr>")
    sections.append('<h2>Результат управления</h2><table><tr><th>Режим</th><th>n</th><th>Train</th><th>Val</th><th>Test</th></tr>'+"".join(rows)+"</table>")
    comparison = run/"policy_comparison.gif"
    if comparison.exists():
        sections.append('<h2>Одна исходная сцена: три способа управления</h2>'
                        '<p>PPO, video и joint выполняют действие из одинаковой validation-сцены. '
                        'Анимация собрана из сохранённых настоящих траекторий, без нового запуска моделей.</p>'+
                        media(comparison,"PPO: успех / video: ошибка / joint: успех"))
    dataset = sorted((run/"dataset/val").glob("episode_*.mp4"))
    sections.append('<h2>Настоящая среда: демонстрации PPO</h2><p>Сначала исходная сцена, затем последствия последовательных действий. '
                    'В старой задаче с одним переходом ролик содержит два кадра. Красная стрелка — агент; нижняя панель показывает цель и карман.</p><div class="grid">'+
                    "".join(media(p,f"PPO, validation: {p.stem}") for p in dataset)+"</div>")
    reconstructed = sorted((run/"latents").glob("val_reconstruction_*.png"))
    sections.append('<h2>Что сохраняет и теряет VAE</h2><p>Четыре панели слева направо: настоящий текущий кадр, его VAE-реконструкция, '
                    'настоящий последний кадр эпизода, его VAE-реконструкция. Смотрите на форму предметов, направление стрелки и карман.</p><div class="wide">'+
                    "".join(media(p,p.name) for p in reconstructed)+"</div>")
    for mode in ("policy","video","joint"):
        directories = sorted((run/"models").glob(f"{mode}_n*_seed*"))
        for directory in directories:
            split = directory/"val"
            if not (split/"evaluation.json").exists():
                continue
            evaluation = json.loads((split/"evaluation.json").read_text())
            control = evaluation["closed_loop"]
            text = ("Модель действует через action head." if mode != "video" else
                    "Действие выбирает планировщик по предсказанным кадрам. Это отдельная проверка от качества картинок.")
            clips = sorted(split.glob("task_*_*.mp4"), key=lambda p:("failure" not in p.name,p.name))
            forecasts = sorted(split.glob("forecast_*_truth_left.mp4"))
            sections.append(f'<section data-mode="{mode}"><h2>{html.escape(directory.name)}</h2>'+
                f'<p>Validation SR: <b>{control["successes"]}/{control["episodes"]} ({control["sr"]:.1%})</b>. {text}</p>'+
                '<h3>Исполнение в настоящей среде</h3><div class="grid">'+
                "".join(media(p,"Ошибка" if "failure" in p.name else "Успех") for p in clips)+"</div>"+
                ('<h3>Прогноз: слева настоящий будущий кадр, справа предсказанный</h3>'
                 '<p>Прогноз сделан по записанным действиям, которые могут быть неуспешными. Новые ролики начинают с контекста и '
                 'показывают только настоящие переходы до terminal. В старых файлах n=1 только один будущий кадр, '
                 'поэтому такой ролик статичен.</p><div class="wide">'+
                 "".join(media(p,p.name) for p in forecasts)+"</div>" if forecasts else "")+"</section>")
    if diagnostics:
        diagnostics = Path(diagnostics).resolve()
        report = json.loads((diagnostics/"diagnosis.json").read_text())
        sections.append('<h2>Проверка шести действий на тех же весах</h2><p>Только validation; без обучения. '
                        'Для каждой сцены показаны настоящие последствия, VAE-реконструкции и прогнозы NanoWM.</p>'+
                        '<pre>'+html.escape(json.dumps(report["aggregate"],ensure_ascii=False,indent=2))+'</pre><div class="wide">'+
                        "".join(media(p,p.stem) for p in sorted(diagnostics.glob("task_[0-9][0-9][0-9].png")))+"</div>")
        sections.append('<h3>Анимации: настоящий кадр / VAE / прогноз</h3><p>Сначала исходное наблюдение, затем три варианта будущего. '
                        'Показаны действия старого и нового планировщика на первых шести validation-сценах.</p><div class="wide">'+
            "".join(media(diagnostics/f"task_{r['task_id']:03d}_action_{action}.gif",f"task {r['task_id']}, action {action}")
                for r in report["records"][:6] for action in sorted({r["choices"]["model_raw_cost"],r["choices"]["model_goal_cost"]}))+"</div>")
    document = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-WAM — просмотр результатов</title><style>
*{box-sizing:border-box}body{margin:0;background:#111827;color:#e5e7eb;font:16px/1.6 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:36px 24px}h1,h2,h3{line-height:1.3}h2{margin-top:42px}
p{max-width:1000px;color:#cbd5e1}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}
.wide{display:grid;gap:18px}figure{margin:0;padding:14px;background:#1f2937;border:1px solid #374151;border-radius:12px}
video,img{display:block;width:100%;height:auto;max-height:640px;object-fit:contain;image-rendering:pixelated;background:#0b1020}
.grid video{max-height:300px}figcaption{font-size:14px;margin-top:8px;color:#cbd5e1;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;max-width:760px}th,td{text-align:left;padding:10px 16px;border-bottom:1px solid #374151}
button,select{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid #64748b;background:#1f2937;color:#fff;margin:4px}
.toolbar{position:sticky;top:0;z-index:2;background:#111827ed;padding:10px 0;border-bottom:1px solid #374151}pre{white-space:pre-wrap}
</style><main><h1>mini-WAM: смотрим реальные результаты</h1>
<p>Эта страница читает уже сохранённые картинки и видео. Нового обучения и подключения к серверу нет.</p>
<div class="toolbar"><button onclick="document.querySelectorAll('video').forEach(v=>{if(v.getBoundingClientRect().top<innerHeight&&v.getBoundingClientRect().bottom>0)v.play().catch(()=>{})})">▶ Видимые видео</button>
<button onclick="document.querySelectorAll('video').forEach(v=>v.pause())">Ⅱ Пауза</button>
<label>Скорость <select onchange="document.querySelectorAll('video').forEach(v=>v.playbackRate=Number(this.value))"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label></div>
'''+"\n".join(sections)+"</main></html>\n"
    out.write_text(document)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="runs/from-server/one_step")
    parser.add_argument("--out")
    parser.add_argument("--diagnostics")
    args = parser.parse_args()
    path = build_gallery(args.run,args.out,args.diagnostics)
    print(f"Gallery: {path}\nOpen on macOS: open '{path}'")


if __name__ == "__main__":
    main()
