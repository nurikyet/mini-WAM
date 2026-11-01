"""Evaluate frozen video weights and two task costs; never train or overwrite a run."""
from __future__ import annotations

from pathlib import Path

from .common import (completed, file_hash, finish, read_json, stage, verify_files, write_json)
from .evaluation import visual_score
from .goal_images import goal_images, goal_scores, goal_weights
from .model import autocast
from .progress import Progress, log, logged_stage
from .training import load_model
from .vae import decode, encode, load_vae, vae_identity
from .xland import ACTION_NAMES, load_bank, render_ts, simulator, single_ts
import jax
import numpy as np
from PIL import Image, ImageDraw
import torch


def comparison_image(path, current, true_frames, reconstructed, predicted, rows):
    scale, top, row_height = 2, 36, 286
    canvas = Image.new("RGB",(4*128*scale,top+6*row_height),(239,243,248))
    draw = ImageDraw.Draw(canvas)
    headings = ("Current RGB", "Real next RGB", "VAE reconstruction", "NanoWM prediction")
    for c,title in enumerate(headings):
        draw.text((c*256+8,8),title,fill="black")
    for a in range(6):
        label=f"a={a} {ACTION_NAMES[a]} | success={rows[a]['success']} | raw={rows[a]['model_raw_score']:.2f} | latent={rows[a]['model_goal_score']:.5f}"
        draw.text((8,top+a*row_height),label,fill="black")
        for c,frame in enumerate((current,true_frames[a],reconstructed[a],predicted[a])):
            img=Image.fromarray(frame).resize((256,256),Image.Resampling.NEAREST)
            canvas.paste(img,(c*256,top+a*row_height+22))
    canvas.save(path)


@torch.no_grad()
def diagnose_video(run, out, device, limit=None, ddim_steps=20, resume=False):
    run=Path(run)
    checkpoint=run/"models/video_n1_seed3407/best.pt"
    if not checkpoint.is_file():
        matches=sorted((run/"models").glob("video_n1_seed*/best.pt"))
        if len(matches) != 1:
            raise ValueError("Expected one video n=1 checkpoint in --run/models")
        checkpoint=matches[0]
    original=read_json(checkpoint.parent/"complete.json")
    verify_files(checkpoint.parent,{"best.pt":original["files_sha256"]["best.pt"]})
    from .protocol import episode_limit
    depth=episode_limit(read_json(run/"banks/manifest.json")["config"])
    if depth != 1:
        raise ValueError("This diagnostic explicitly supports only the one-step curriculum")
    out=stage(out,{"checkpoint_sha256":file_hash(checkpoint),"bank_sha256":file_hash(run/"banks/manifest.json"),
        "split":"val","limit":limit,"ddim_steps":ddim_steps,"vae":vae_identity()},resume)
    if completed(out):
        log("ПРОПУСК: диагностика уже сохранена",path=out)
        return out
    model,saved=load_model(checkpoint,device)
    vae=load_vae(device)
    bank=load_bank(run/"banks","val")
    env,params=simulator(1)
    step=jax.jit(env.step)
    count=len(bank["goal"]) if limit is None else min(limit,len(bank["goal"]))
    if count < 1 or not 2 <= ddim_steps <= 1000:
        raise ValueError("limit must be positive and ddim_steps in [2,1000]")
    records=[]
    files=["diagnosis.json"]
    wins={name:0 for name in ("perfect_rgb_raw_cost","vae_reconstruction_raw_cost","true_latents_goal_cost","model_raw_cost","model_goal_cost")}
    progress=Progress("Диагностика video: validation-сцены",count)
    moving=0
    with logged_stage("Прогноз шести действий на замороженных весах",checkpoint_step=saved["step"]):
        for task in range(count):
            ts=single_ts(bank,task)
            current=render_ts(ts)
            # These futures are ground truth for DIAGNOSTICS ONLY. Neither task
            # cost nor NanoWM is passed these states or their success labels.
            real=[step(params,ts,a) for a in range(6)]
            true_frames=np.stack([render_ts(s) for s in real])
            success=np.array([float(s.reward)>0 for s in real])
            moving += sum(not np.array_equal(np.asarray(s.state.agent.position),np.asarray(ts.state.agent.position)) for s in real)
            z=encode(vae,current[None],device)
            true_z=encode(vae,true_frames,device)
            reconstructed=decode(vae,true_z)
            actions=torch.arange(6,device=device)[:,None]
            gen=torch.Generator(device=device).manual_seed(2027+task)
            noise=torch.randn((1,1,4,16,16),generator=gen,device=device).expand(6,-1,-1,-1,-1)
            with autocast(device):
                predicted_z=model.sample(z.expand(6,-1,-1,-1),actions,ddim_steps,noise)[:,0]
            predicted=decode(vae,predicted_z)
            current_tensor=torch.as_tensor(current.copy(),device=device).permute(2,0,1).float()[None]/255
            true_tensor=torch.as_tensor(true_frames.copy(),device=device).permute(0,3,1,2).float()/255
            goals=goal_images(current)
            encoded_goals=encode(vae,goals,device)
            weights=goal_weights(current,goals,device)
            scores={"perfect_rgb_raw_cost":visual_score(true_tensor,current_tensor.expand(6,-1,-1,-1)),
                "vae_reconstruction_raw_cost":visual_score(reconstructed,current_tensor.expand(6,-1,-1,-1)),
                "true_latents_goal_cost":goal_scores(true_z,encoded_goals,weights),
                "model_raw_cost":visual_score(predicted,current_tensor.expand(6,-1,-1,-1)),
                "model_goal_cost":goal_scores(predicted_z,encoded_goals,weights)}
            choices={name:int(values.argmax()) for name,values in scores.items()}
            for name,choice in choices.items():wins[name] += int(success[choice])
            rows=[{"action":a,"name":ACTION_NAMES[a],"success":bool(success[a]),
                "model_raw_score":float(scores["model_raw_cost"][a]),"model_goal_score":float(scores["model_goal_cost"][a])} for a in range(6)]
            record={"task_id":task,"goal":bank["goal"][task].tolist(),"choices":choices,
                "choice_success":{name:bool(success[a]) for name,a in choices.items()},
                "scores":{name:value.float().cpu().tolist() for name,value in scores.items()},"actions":rows}
            records.append(record)
            # Save all validation cases so failed ties and successful changes are reviewable.
            reconstructed_rgb=(reconstructed.permute(0,2,3,1).cpu().numpy().clip(0,1)*255).astype(np.uint8)
            predicted_rgb=(predicted.permute(0,2,3,1).cpu().numpy().clip(0,1)*255).astype(np.uint8)
            name=f"task_{task:03d}.png"
            comparison_image(out/name,current,true_frames,reconstructed_rgb,predicted_rgb,rows)
            files.append(name)
            for a in range(6):
                # Context + future, three panels: real, VAE, model. Unlike the
                # old n=1 forecast clips this is an actual two-frame animation.
                before=np.concatenate([current,current,current],axis=1)
                after=np.concatenate([true_frames[a],reconstructed_rgb[a],predicted_rgb[a]],axis=1)
                # PIL animation avoids forking ffmpeg from a multithreaded JAX process.
                name=f"task_{task:03d}_action_{a}.gif"
                frames=[Image.fromarray(before),Image.fromarray(after)]
                frames[0].save(out/name,save_all=True,append_images=frames[1:],duration=[1200,1600],loop=0)
                files.append(name)
            Image.fromarray(np.concatenate(list(goals),axis=1)).save(out/f"task_{task:03d}_goals.png")
            files.append(f"task_{task:03d}_goals.png")
            progress.update(task+1,old_sr=f"{wins['model_raw_cost']/(task+1):.1%}",goal_sr=f"{wins['model_goal_cost']/(task+1):.1%}")
    aggregate={name:{"successes":k,"episodes":count,"sr":k/count} for name,k in wins.items()}
    report={"split":"val","training":False,"checkpoint":str(checkpoint),"checkpoint_sha256":file_hash(checkpoint),
        "selected_step":saved["step"],"new_cost":"nearest public goal image in VAE latent space; fixed weights 1+4*ROI",
        "aggregate":aggregate,"moving_counterfactuals":int(moving),"records":records,
        "caveat":"Validation diagnostic, not a new test result; original run and weights are unchanged"}
    write_json(out/"diagnosis.json",report)
    finish(out,files)
    log("ДИАГНОСТИКА ГОТОВА",report=out/"diagnosis.json",results=aggregate)
    return out
