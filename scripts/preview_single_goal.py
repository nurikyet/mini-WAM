#!/usr/bin/env python3
"""Render native construction witnesses, explicitly not learned-model results."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from miniwam.server.common import read_json, write_json
from miniwam.server.protocol import episode_limit, renderer_version
from miniwam.server.xland import load_bank, reference_plan, render_ts, simulator, single_ts
import jax
import numpy as np
from PIL import Image, ImageDraw


def preview(banks, out, split="val"):
    banks,out=Path(banks),Path(out)
    config=read_json(banks/"manifest.json")["config"]
    if config.get("family") != "single_goal":
        raise ValueError("Use corrected single_goal banks, not the old near-terminal results")
    bank=load_bank(banks,split)
    env,params=simulator(episode_limit(config));step=jax.jit(env.step)
    out.mkdir(parents=True,exist_ok=True)
    report={"source":"native construction witness; NO learned PPO/WAM predictions",
            "renderer":renderer_version(config),"episodes":[]}
    rows=[]
    for kind,name in ((3,"approach"),(1,"pick_up"),(4,"put_near")):
        ids=np.flatnonzero((bank["goal"][:,0]==kind)&(bank["travel_moves"]==3))
        if not len(ids):ids=np.flatnonzero(bank["goal"][:,0]==kind)
        aligned=ids[bank["direction"][ids]==bank["approach_direction"][ids]]
        task=int((aligned if len(aligned) else ids)[0])
        record={k:v[task] for k,v in bank.items()}
        actions=reference_plan(record)
        ts=single_ts(bank,task)
        frames=[render_ts(ts,renderer_version(config))]
        for action in actions:
            ts=step(params,ts,action);frames.append(render_ts(ts,renderer_version(config)))
        if not float(ts.reward)>0:raise ValueError("Reference plan failed")
        tiles=[]
        for t,frame in enumerate(frames):
            panel=Image.new("RGB",(384,430),(24,24,24));draw=ImageDraw.Draw(panel)
            draw.text((8,8),"REFERENCE SIMULATOR / NOT A TRAINED MODEL",fill="white")
            draw.text((8,27),f"{name}: transition {t}/{len(actions)}",fill="white")
            panel.paste(Image.fromarray(frame).resize((384,384),Image.Resampling.NEAREST),(0,46))
            tiles.append(panel)
        tiles[0].save(out/f"{name}.gif",save_all=True,append_images=tiles[1:],
                      duration=[1000]+[650]*(len(tiles)-2)+[1600],loop=0)
        selected=[tiles[0],tiles[min(3,len(tiles)-2)],tiles[-1]]
        row=Image.new("RGB",(1152,430));[row.paste(tile,(i*384,0)) for i,tile in enumerate(selected)]
        rows.append(row)
        report["episodes"].append({"task_id":task,"goal_type":kind,"travel_moves":int(record["travel_moves"]),
                                  "actions":actions,"success":True,"file":f"{name}.gif"})
    sheet=Image.new("RGB",(1152,430*len(rows)))
    for i,row in enumerate(rows):sheet.paste(row,(0,i*430))
    sheet.save(out/"reference_examples.png")
    write_json(out/"reference.json",report)
    (out/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Одна цель, несколько переходов</title>'
        '<style>body{background:#181818;color:#eee;font:18px system-ui}img{width:384px;max-width:100%}</style>'
        '<h1>Одна цель, несколько переходов</h1><p>Эталонные действия в настоящем симуляторе. '
        'Это иллюстрация постановки, не результат обученной PPO или NanoWM. Пустые поля без восклицательных знаков.</p>'+
        ''.join(f'<img src="{r["file"]}" alt="goal {r["goal_type"]}">' for r in report["episodes"]))
    return out/"index.html"


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--banks",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--split",choices=("train","val","test"),default="val")
    args=parser.parse_args();print(preview(args.banks,args.out,args.split))
