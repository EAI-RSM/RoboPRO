#!/usr/bin/env python3
"""Build a read-only researcher-facing pilot from one validated episode."""
from __future__ import annotations
import argparse, importlib.util, json, shutil
from pathlib import Path
import cv2, h5py, numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
EPISODE=ROOT/"customized_robotwin/data/action_validation_suite_v6/chain_heat_hamburger_ks/relation_validation_d14"
OUTPUT=ROOT/"docs/research_feature_showcase/hamburger_in_microwave"
MOMENTS=[(122,"Contained Before Contact","The hamburger is inside the microwave while transport remains active."),(123,"Container Contact Established","collides_with appears between the hamburger and microwave."),(128,"Contact Re-established","The transient object-container contact appears again during transport."),(142,"Release Phase","The hamburger remains contained as the left gripper begins release.")]
C={"target":(40,80,230),"collides_with":(20,20,220),"destination":(20,160,90),"effector":(170,0,170),"camera":(220,120,20),"action":(0,150,210),"text":(38,38,38),"in":(20,140,20),"contains":(20,160,90),"held_by":(170,0,170),"reachable_by":(0,150,210),"visible_to":(220,120,20)}

def dec(x): return x.decode("utf-8",errors="replace") if isinstance(x,(bytes,np.bytes_)) else (x.item() if isinstance(x,np.generic) else x)
def renderer():
 p=ROOT/"benchmark/bench_script/visualize_relation_frame.py"; s=importlib.util.spec_from_file_location("relation_renderer",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def image(ds,frame,flags=cv2.IMREAD_COLOR):
 raw=ds[frame]; raw=raw.rstrip(b"\0") if isinstance(raw,bytes) else bytes(raw); out=cv2.imdecode(np.frombuffer(raw,np.uint8),flags)
 if out is None: raise ValueError(f"Cannot decode frame {frame}")
 return out
def action(root,frame):
 a=root["benchmark_support/action_nodes"]; active=np.flatnonzero(a["active"][frame])
 if not len(active): return None
 i=int(active[-1]); row={"id":int(a["action_ids"][i]),"type":dec(a["action_types"][i]),"phase":dec(a["execution_phases"][i]),"arm":dec(a["arms"][i]),"start_frame":int(a["start_frame"][i]),"end_frame":int(a["end_frame"][i]),"status":dec(a["statuses"][i])}
 for src,dst in (("tool_calls_json","tool_call"),("observed_effects_json","observed_effects")):
  value=dec(a[src][i])
  try: value=json.loads(value)
  except (TypeError,json.JSONDecodeError): pass
  row[dst]=value
 return row
def edges(root,frame):
 s=root["benchmark_support"]; rs=s["relation_state"]; ids=[int(x) for x in s["object_catalog/object_ids"][()]]; names=[dec(x) for x in s["object_catalog/names"][()]]; labels=dict(zip(ids,names)); state_ids=[int(x) for x in s["object_state/object_ids"][()]]; target=state_ids.index(ids[names.index("006_hamburg")]); microwave=state_ids.index(ids[names.index("microwave")]); out=[]
 for rel in ("in","contains","collides_with"):
  for src,dst in np.argwhere(rs[rel][frame]):
   if {int(src),int(dst)}=={target,microwave}: out.append({"relation":rel,"source":labels[state_ids[src]],"destination":labels[state_ids[dst]]})
 for rel,key in (("held_by","held_by_effector_names"),("reachable_by","reachable_by_effector_names")):
  endpoints=[dec(x) for x in rs[key][()]]
  for src,dst in np.argwhere(rs[rel][frame]):
   if int(src)==target: out.append({"relation":rel,"source":"006_hamburg","destination":endpoints[dst]})
 cameras=[dec(x) for x in rs["visible_to_camera_names"][()]]
 for src,dst in np.argwhere(rs["visible_to"][frame]):
  if int(src)==target and cameras[dst]=="countertop_camera": out.append({"relation":"visible_to","source":"006_hamburg","destination":"countertop_camera"})
 return out
def scene(root,frame):
 rgb=image(root["observation/countertop_camera/rgb"],frame); seg=image(root["observation/countertop_camera/actor_segmentation"],frame,cv2.IMREAD_UNCHANGED); out=cv2.resize(rgb,(960,720),interpolation=cv2.INTER_CUBIC); sx,sy=960/rgb.shape[1],720/rgb.shape[0]
 for actor,label,color in ((112,"target: hamburger",C["target"]),(90,"destination: microwave",C["destination"])):
  ys,xs=np.where(seg==actor)
  if not len(xs): continue
  x0,x1=int(xs.min()*sx),int((xs.max()+1)*sx); y0,y1=int(ys.min()*sy),int((ys.max()+1)*sy); cv2.rectangle(out,(x0,y0),(x1,y1),color,3,cv2.LINE_AA); cv2.rectangle(out,(x0,max(0,y0-28)),(min(959,x0+235),y0),color,-1); cv2.putText(out,label,(x0+5,max(18,y0-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
 return out
def graph(root,frame,draw):
 out=np.full((720,960,3),255,np.uint8); p={"006_hamburg":np.array([240,340]),"microwave":np.array([710,340]),"left_ee":np.array([230,580]),"countertop_camera":np.array([220,110]),"action":np.array([710,110])}
 for e in edges(root,frame):
  if e["source"] in p and e["destination"] in p: draw._draw_edge(out,p[e["source"]],p[e["destination"]],C[e["relation"]],e["relation"],dashed=e["relation"] in {"reachable_by","visible_to"})
 a=action(root,frame)
 if a: draw._draw_edge(out,p["action"],p["006_hamburg"],C["target"],"target"); draw._draw_edge(out,p["action"],p["left_ee"],C["effector"],"agent")
 for name,kind,shape in (("006_hamburg","target","circle"),("microwave","destination","square"),("left_ee","effector","diamond"),("countertop_camera","camera","triangle")):
  draw._draw_node(out,p[name],C[kind],shape,size=11); cv2.putText(out,name,tuple(p[name]+[16,6]),cv2.FONT_HERSHEY_SIMPLEX,.62,C["text"],2,cv2.LINE_AA)
 if a: draw._draw_node(out,p["action"],C["action"],"diamond",size=13); cv2.putText(out,f"A{a['id']} {a['type']}",tuple(p["action"]+[18,0]),cv2.FONT_HERSHEY_SIMPLEX,.62,C["text"],2,cv2.LINE_AA); cv2.putText(out,a["phase"],tuple(p["action"]+[18,24]),cv2.FONT_HERSHEY_SIMPLEX,.48,(90,90,90),1,cv2.LINE_AA)
 cv2.putText(out,"ACTION-AWARE SCENE GRAPH",(28,38),cv2.FONT_HERSHEY_SIMPLEX,.78,C["text"],2,cv2.LINE_AA); return out
def panel(left,right,frame,title,caption):
 out=np.full((1080,1920,3),248,np.uint8); out[150:870,:960]=left; out[150:870,960:]=right; cv2.putText(out,f"{title}  |  frame {frame}",(44,66),cv2.FONT_HERSHEY_SIMPLEX,1.18,C["text"],3,cv2.LINE_AA); cv2.putText(out,"Scene",(32,130),cv2.FONT_HERSHEY_SIMPLEX,.72,C["text"],2,cv2.LINE_AA); cv2.putText(out,"Graph",(992,130),cv2.FONT_HERSHEY_SIMPLEX,.72,C["text"],2,cv2.LINE_AA); cv2.putText(out,caption,(44,940),cv2.FONT_HERSHEY_SIMPLEX,.66,C["text"],2,cv2.LINE_AA); cv2.putText(out,"Exported evidence: schema 1.6.0 | rule_based expert | countertop_camera",(44,1000),cv2.FONT_HERSHEY_SIMPLEX,.52,(95,95,95),1,cv2.LINE_AA); return out
def edge_text(e): return f"{e['source']} --{e['relation']}--> {e['destination']}"
def delta(before,after):
 out=np.full((720,1280,3),255,np.uint8); b={edge_text(e) for e in before["relations"]}; a={edge_text(e) for e in after["relations"]}; cv2.putText(out,f"GRAPH DELTA: frame {before['frame']} -> {after['frame']}",(42,62),cv2.FONT_HERSHEY_SIMPLEX,1.05,C["text"],3,cv2.LINE_AA); y=130
 for heading,items,color,prefix in (("ADDED",sorted(a-b),(25,145,55),"+"),("REMOVED",sorted(b-a),(45,60,205),"-")):
  cv2.putText(out,heading,(48,y),cv2.FONT_HERSHEY_SIMPLEX,.76,color,2,cv2.LINE_AA); y+=42
  for item in items or ["(none in focused graph)"]: cv2.putText(out,(prefix+" "+item) if not item.startswith("(") else item,(70,y),cv2.FONT_HERSHEY_SIMPLEX,.62,color,2,cv2.LINE_AA); y+=40
  y+=24
 return out
def write_report(out,records,tool):
 rows="\n".join(f"| {r['frame']} | {r['title']} | {r['caption']} |" for r in records); text=f'''# Hamburger-microwave physics-aware graph: pilot

This pilot pairs each exact exported camera frame with two graph views at the same HDF5 index: the authoritative `full_scene_graph` and the policy-facing `action_relevant_subgraph`.

| Frame | Evidence stage | Interpretation |
|---:|---|---|
{rows}

## Tool-calling transition

The graph grounds the target, destination, and acting arm. The exported provider-neutral call is:

```json
{json.dumps(tool,indent=2)}
```

The graph distinguishes persistent containment from transient object-container contact during transport.

## Scope

- `visible_to` means at least one segmentation pixel, not full visibility.
- `reachable_by` is collision-aware IK, not full grasp or trajectory feasibility.
- `occludes` and `blocks` are not claimed as canonical relations.
'''; (out/"README.md").write_text(text)
def main():
 ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--episode-root",type=Path,default=EPISODE); ap.add_argument("--output",type=Path,default=OUTPUT); args=ap.parse_args(); episode=args.episode_root.resolve(); out=args.output.resolve(); h5=episode/"data/episode0.hdf5"; video=episode/"video/episode0.mp4"
 if not h5.is_file() or not video.is_file(): raise SystemExit(f"Missing HDF5/video pair below {episode}")
 for d in (out/"keyframes",out/"graphs"/"full_scene_graph",out/"graphs"/"action_relevant_subgraph",out/"deltas"): d.mkdir(parents=True,exist_ok=True)
 shutil.copy2(video,out/"episode.mp4"); draw=renderer(); records=[]
 with h5py.File(h5,"r") as root:
  for frame,title,caption in MOMENTS:
   s=scene(root,frame); g=graph(root,frame,draw); record={"frame":frame,"title":title,"caption":caption,"active_action":action(root,frame),"relations":edges(root,frame),"graph_outputs":{"full_scene_graph":f"graphs/full_scene_graph/graph_{frame:04d}.png","action_relevant_subgraph":f"graphs/action_relevant_subgraph/graph_{frame:04d}.png"}}; records.append(record); cv2.imwrite(str(out/"keyframes"/f"frame_{frame:04d}.png"),s); cv2.imwrite(str(out/"graphs"/"action_relevant_subgraph"/f"graph_{frame:04d}.png"),g); draw.render_relation_frame(h5,frame,out/"graphs"/"full_scene_graph"/f"graph_{frame:04d}.png",1600,1000,show_edge_labels=True,excluded_edges=set(),abstract_layout=False); cv2.imwrite(str(out/f"panel_{frame:04d}.png"),panel(s,g,frame,title,caption))
  for before,after in zip(records,records[1:]): cv2.imwrite(str(out/"deltas"/f"delta_{before['frame']:04d}_{after['frame']:04d}.png"),delta(before,after))
 tool=next((r["active_action"].get("tool_call") for r in records if r["active_action"] and r["active_action"].get("type")=="transport"),None); evidence={"claim":"Hamburger-microwave physics-aware graph","source_hdf5":str(h5.relative_to(ROOT)),"source_video":str(video.relative_to(ROOT)),"camera":"countertop_camera","keyframes":records,"tool_call_example":tool,"selection_method":"Exported action boundaries and relation transition frames","graph_views":{"full_scene_graph":{"definition":"All present entities and all implemented relations at the selected frame, positioned from exported world coordinates"},"action_relevant_subgraph":{"definition":"One-hop action-centered projection over target, destination, effector, and selected camera","seed_nodes":["active_action","006_hamburg","microwave","left_ee","countertop_camera"],"relations":["target","agent","in","contains","collides_with","held_by","reachable_by","visible_to"]}}}; (out/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n"); write_report(out,records,tool); print(f"Built pilot showcase at {out}")
if __name__=="__main__": main()
