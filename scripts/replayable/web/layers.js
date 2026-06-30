/* Layered viewer engine. Per-view (main + optional twin), driven by shared frame/layers.
   Everything is guarded: a failing layer skips, it never blanks the canvas. */
'use strict';
const $ = id => document.getElementById(id);

let IDX, frame = 0, playing = false, baseMode = 'rgb', selId = null;
let needRender = true, lastT = performance.now(), acc = 0;
const layers = { pcd:true, bbox:true, ids:true, traj:false, vel:false, grip:true, skel:false,
                 occ:false, relview:true, minimap:true, graph:true };
const success = { eps: 4, h: 2 };
const objVis = {};                          // global -> applies to both twin views
const off = document.createElement('canvas');
const offctx = off.getContext('2d', { willReadFrequently: true });
const GRIP_COL = { left:'#5cc8ff', right:'#ffb86b' };
const EDGES = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]];

// ---- math ----
function quatMat(q){let[x,y,z,w]=q;const n=Math.hypot(x,y,z,w)||1;x/=n;y/=n;z/=n;w/=n;
  return[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];}
const mv=(M,v)=>[M[0][0]*v[0]+M[0][1]*v[1]+M[0][2]*v[2],M[1][0]*v[0]+M[1][1]*v[1]+M[1][2]*v[2],M[2][0]*v[0]+M[2][1]*v[1]+M[2][2]*v[2]];
const add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]], sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], norm=a=>Math.hypot(a[0],a[1],a[2]);
const camRT=(c,f)=>{const e=c.extrinsic_cv[f];return{R:[[e[0],e[1],e[2]],[e[4],e[5],e[6]],[e[8],e[9],e[10]]],t:[e[3],e[7],e[11]]};};
function project(P,RT,K){const Pc=add(mv(RT.R,P),RT.t);if(Pc[2]<=1e-4)return null;
  return[(K[0][0]*Pc[0]+K[0][1]*Pc[1]+K[0][2]*Pc[2])/Pc[2],(K[1][0]*Pc[0]+K[1][1]*Pc[1]+K[1][2]*Pc[2])/Pc[2],Pc[2]];}
const css=c=>`rgb(${c[0]*255|0},${c[1]*255|0},${c[2]*255|0})`;
const heat=f=>`hsl(${8+Math.max(0,Math.min(1,f))*202},85%,58%)`;
const DT=0.03, AB={left:'L',right:'R',front:'F',back:'B',up:'U',down:'D'};
function dir3d(d){const p=[];if(d[0]>DT)p.push('right');else if(d[0]<-DT)p.push('left');
  if(d[1]>DT)p.push('back');else if(d[1]<-DT)p.push('front');if(d[2]>DT)p.push('up');else if(d[2]<-DT)p.push('down');return p;}
const dirShort=d=>dir3d(d).map(x=>AB[x]).join('·');
function heatOf(objs,V,sf,g){const ds=objs.map(o=>norm(sub(oCenter(V,sf,o),g))*100);const mn=Math.min(...ds),mx=Math.max(...ds);
  return{ds,frac:ds.map(d=>mx>mn?(d-mn)/(mx-mn):0),closest:ds.indexOf(mn)};}

// ---- views ----
function makeView(cvId,tagId){const cv=$(cvId);return{cv,ctx:cv.getContext('2d'),tag:$(tagId),S:null,CAMS:[],camName:null,imgC:new Map(),depthC:new Map(),lastReady:null,nf:0,bounds:null};}
const V=makeView('cv','camtag'), CMP=makeView('cv2','camtag2');
let cmpOn=false;
const camOf=v=>v.CAMS.find(c=>c.name===v.camName)||v.CAMS[0];
const oPos=(v,f,i)=>v.S.frames.object_pos[f].slice(i*3,i*3+3);
const oQuat=(v,f,i)=>v.S.frames.object_quat[f].slice(i*4,i*4+4);
const oCenter=(v,f,o)=>add(oPos(v,f,o.id),mv(quatMat(oQuat(v,f,o.id)),o.bbox?o.bbox.center:[0,0,0]));
const gPos=(v,side,f)=>v.S.grippers[side].pose[f].slice(0,3);
const trackedV=v=>v.S.objects.filter(o=>o.tracked&&objVis[o.id]);

// ---- image / depth ----
const fnum=f=>String(f).padStart(4,'0');
function imgFor(v,c,f){const u=`${c.rgb}/f${fnum(f)}.jpg`;let im=v.imgC.get(u);if(!im){im=new Image();im.decoding='async';im.onload=()=>needRender=true;im.onerror=()=>{needRender=true;};im.src=u;v.imgC.set(u,im);}return im;}
function depthImgFor(v,c,f){if(!c.depth)return null;const u=`${c.depth}/f${fnum(f)}.png`;let im=v.imgC.get(u);if(!im){im=new Image();im.onload=()=>needRender=true;im.src=u;v.imgC.set(u,im);}return im;}
const ready=im=>im&&im.complete&&im.naturalWidth>0;
function depthData(v,c,f){if(!c.depth)return null;const key=c.name+':'+f;let d=v.depthC.get(key);if(d)return d;
  const im=depthImgFor(v,c,f);if(!ready(im))return null;off.width=c.width;off.height=c.height;offctx.drawImage(im,0,0,c.width,c.height);
  d={data:offctx.getImageData(0,0,c.width,c.height).data,w:c.width,h:c.height,near:c.depth_range[0],far:c.depth_range[1]};v.depthC.set(key,d);return d;}
const surfDepth=(dd,u,v)=>{const x=u|0,y=v|0;if(x<0||y<0||x>=dd.w||y>=dd.h)return Infinity;return dd.near+(dd.data[(y*dd.w+x)*4]/255)*(dd.far-dd.near);};
function preloadCam(v){const c=camOf(v);if(!c)return;const cur=Math.min(frame,v.nf-1);  // load outward from the visible frame first
  const wantDepth=c.depth&&(layers.occ||baseMode==='depth');
  for(let d=0;d<v.nf;d++){const a=cur+d,b=cur-d;
    if(a<v.nf){imgFor(v,c,a);if(wantDepth)depthImgFor(v,c,a);}
    if(d&&b>=0){imgFor(v,c,b);if(wantDepth)depthImgFor(v,c,b);}}}
function loadStat(v){const c=camOf(v);if(!c)return[0,0];let n=0;for(let f=0;f<v.nf;f++)if(ready(v.imgC.get(`${c.rgb}/f${fnum(f)}.jpg`)))n++;return[n,v.nf];}

// ---- overlay drawers (V, sf, RT, K, lw, occ) ----
function drawPcd(v,sf,RT,K,lw,occ,big){const sz=Math.max(1,lw*(big?3:1.1));const ctx=v.ctx;
  for(const o of trackedV(v)){if(!o.points)continue;const p=oPos(v,sf,o.id),R=quatMat(oQuat(v,sf,o.id)),P=o.points;
    ctx.fillStyle=css(o.color);ctx.globalAlpha=big?0.95:0.7;
    for(let k=0;k<P.length;k+=3){const q=project(add(p,mv(R,[P[k],P[k+1],P[k+2]])),RT,K);if(q&&!(occ&&occ(q[0],q[1],q[2])))ctx.fillRect(q[0]-sz/2,q[1]-sz/2,sz,sz);}}
  ctx.globalAlpha=1;}
function drawObjects(v,sf,RT,K,lw,occ){const ctx=v.ctx;ctx.font=`${12*lw}px -apple-system,sans-serif`;ctx.textBaseline='bottom';
  for(const o of trackedV(v)){const sel=selId!=null&&o.id===selId, dim=selId!=null&&!sel;
    const color=css(o.color);ctx.globalAlpha=dim?0.35:1;const p=oPos(v,sf,o.id),R=quatMat(oQuat(v,sf,o.id));
    if(layers.bbox&&o.bbox){const c0=o.bbox.center,hf=o.bbox.half,C=[];
      for(const sx of[-1,1])for(const sy of[-1,1])for(const sz of[-1,1])C.push(project(add(p,mv(R,[c0[0]+sx*hf[0],c0[1]+sy*hf[1],c0[2]+sz*hf[2]])),RT,K));
      ctx.strokeStyle=color;ctx.lineWidth=(sel?2.2:1)*lw;ctx.beginPath();
      for(const[a,b]of EDGES)if(C[a]&&C[b]){ctx.moveTo(C[a][0],C[a][1]);ctx.lineTo(C[b][0],C[b][1]);}ctx.stroke();}
    if(layers.traj){ctx.strokeStyle=color;ctx.globalAlpha=(dim?0.2:0.85);ctx.lineWidth=lw;ctx.beginPath();let st=false;
      for(let f=0;f<=sf;f+=2){const q=project(oPos(v,f,o.id),RT,K);if(!q){st=false;continue;}if(!st){ctx.moveTo(q[0],q[1]);st=true;}else ctx.lineTo(q[0],q[1]);}ctx.stroke();ctx.globalAlpha=dim?0.35:1;}
    const cc=project(oCenter(v,sf,o),RT,K);
    if(cc&&!(occ&&occ(cc[0],cc[1],cc[2]))){ctx.fillStyle=color;ctx.beginPath();ctx.arc(cc[0],cc[1],2.4*lw,0,7);ctx.fill();
      if(layers.vel&&sf>0){const pv=oPos(v,sf-1,o.id),vw=sub(oPos(v,sf,o.id),pv);const tip=project(add(oCenter(v,sf,o),vw.map(x=>x*8)),RT,K);
        if(tip&&norm(vw)*100>0.3){ctx.strokeStyle=color;ctx.lineWidth=2*lw;ctx.beginPath();ctx.moveTo(cc[0],cc[1]);ctx.lineTo(tip[0],tip[1]);ctx.stroke();}}
      if(layers.ids){const t=`${o.id} ${o.label}`,tw=ctx.measureText(t).width;ctx.fillStyle='rgba(0,0,0,.55)';ctx.fillRect(cc[0]+4*lw,cc[1]-15*lw,tw+6*lw,15*lw);ctx.fillStyle=color;ctx.fillText(t,cc[0]+7*lw,cc[1]-3*lw);}}
  }ctx.globalAlpha=1;ctx.lineWidth=lw;}
function drawRel(v,sf,RT,K,lw){const ctx=v.ctx,objs=trackedV(v);
  for(const side of['left','right']){const g3=gPos(v,side,sf),gp=project(g3,RT,K);if(!gp)continue;const H=heatOf(objs,v,sf,g3);
    objs.forEach((o,i)=>{const op=project(oCenter(v,sf,o),RT,K);if(!op)return;const fr=H.frac[i],col=heat(fr),far=fr>0.66;
      ctx.strokeStyle=col;ctx.globalAlpha=far?0.5:0.95;ctx.lineWidth=(1+(1-fr)*2.4)*lw;ctx.setLineDash(far?[4*lw,3*lw]:[]);
      ctx.beginPath();ctx.moveTo(gp[0],gp[1]);ctx.lineTo(op[0],op[1]);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;
      const lab=i===H.closest?`${H.ds[i].toFixed(0)}cm ${dirShort(sub(oCenter(v,sf,o),g3))}`:`${H.ds[i].toFixed(0)}`;
      ctx.fillStyle=col;ctx.font=`${11*lw}px monospace`;ctx.fillText(lab,(gp[0]+op[0])/2+3,(gp[1]+op[1])/2);});}
  ctx.lineWidth=lw;}
function drawSkel(v,sf,RT,K,lw,occ){const r=v.S.robot;if(!r||!r.link_pose||!r.bones)return;const ctx=v.ctx,n=r.n_links,LP=r.link_pose[sf];
  const pos=[];for(let j=0;j<n;j++)pos.push(project([LP[j*7],LP[j*7+1],LP[j*7+2]],RT,K));
  ctx.strokeStyle='#9bd1ff';ctx.lineWidth=2.4*lw;ctx.beginPath();
  for(const[a,b]of r.bones)if(pos[a]&&pos[b]){ctx.moveTo(pos[a][0],pos[a][1]);ctx.lineTo(pos[b][0],pos[b][1]);}ctx.stroke();
  ctx.fillStyle='#d6ecff';for(const q of pos)if(q&&!(occ&&occ(q[0],q[1],q[2]))){ctx.beginPath();ctx.arc(q[0],q[1],2.2*lw,0,7);ctx.fill();}
  ctx.lineWidth=lw;}
function drawGrip(v,sf,RT,K,lw){const ctx=v.ctx;ctx.font=`${12*lw}px -apple-system,sans-serif`;
  for(const side of['left','right']){const q=project(gPos(v,side,sf),RT,K);if(!q)continue;const s=7*lw;ctx.strokeStyle=GRIP_COL[side];ctx.lineWidth=2*lw;
    ctx.beginPath();ctx.moveTo(q[0]-s,q[1]-s);ctx.lineTo(q[0]+s,q[1]+s);ctx.moveTo(q[0]+s,q[1]-s);ctx.lineTo(q[0]-s,q[1]+s);ctx.stroke();
    ctx.fillStyle=GRIP_COL[side];ctx.fillText(side==='left'?'L':'R',q[0]+s+2,q[1]+s*0.7);}ctx.lineWidth=lw;}

// ---- main render of a view ----
function drawBuffering(ctx,W,H,n,t){const fs=Math.max(12,Math.round(W/30)),bw=Math.min(W*0.72,300),bh=fs+30,x=(W-bw)/2,y=(H-bh)/2;
  ctx.save();ctx.fillStyle='rgba(13,17,23,.84)';ctx.fillRect(x,y,bw,bh);ctx.strokeStyle='#2b3340';ctx.lineWidth=1;ctx.strokeRect(x,y,bw,bh);
  ctx.fillStyle='#e6edf3';ctx.font=`${fs}px -apple-system,sans-serif`;ctx.textAlign='center';ctx.textBaseline='alphabetic';
  ctx.fillText(`buffering frames  ${n} / ${t}`,W/2,y+fs+6);
  const pw=bw-24,px=x+12,py=y+bh-16;ctx.fillStyle='#1c2330';ctx.fillRect(px,py,pw,9);ctx.fillStyle='#58a6ff';ctx.fillRect(px,py,pw*n/Math.max(t,1),9);
  ctx.restore();}
function renderView(v,isMain){const c=camOf(v);if(!c)return;const W=c.width,H=c.height,ctx=v.ctx;
  if(v.cv.width!==W)v.cv.width=W;if(v.cv.height!==H)v.cv.height=H;
  const f=Math.min(frame,v.nf-1),cur=imgFor(v,c,f),curReady=ready(cur);let sf=f,base=cur;
  if(curReady)v.lastReady={f,img:cur};else if(v.lastReady){sf=v.lastReady.f;base=v.lastReady.img;}
  else{ctx.fillStyle='#05070a';ctx.fillRect(0,0,W,H);const[bn,bt]=loadStat(v);drawBuffering(ctx,W,H,bn,bt);return sf;}
  if(baseMode==='depth'&&c.depth){const di=depthImgFor(v,c,sf);ctx.fillStyle='#05070a';ctx.fillRect(0,0,W,H);if(ready(di))ctx.drawImage(di,0,0,W,H);}
  else if(baseMode==='seg'){ctx.fillStyle='#05070a';ctx.fillRect(0,0,W,H);}
  else ctx.drawImage(base,0,0,W,H);
  const RT=camRT(c,sf),K=c.K,lw=Math.max(1,W/640);
  const dd=(layers.occ&&c.depth)?depthData(v,c,sf):null;
  const occ=dd?((u,vv,z)=>z>surfDepth(dd,u,vv)+0.03):null;
  if(baseMode==='seg')drawPcd(v,sf,RT,K,lw,occ,true);
  else if(layers.pcd)drawPcd(v,sf,RT,K,lw,occ,false);
  if(layers.relview)drawRel(v,sf,RT,K,lw);
  if(layers.skel)drawSkel(v,sf,RT,K,lw,occ);
  drawObjects(v,sf,RT,K,lw,occ);
  if(layers.grip)drawGrip(v,sf,RT,K,lw);
  const gid=isMain?'graph':'graph2',mid=isMain?'minimap':'minimap2';   // per-view (twin-specific) panels
  if(layers.graph)renderGraph(v,sf,gid);else{const g=$(gid);if(g)g.innerHTML='';}
  if(layers.minimap)renderMinimap(v,sf,mid);
  v.tag.textContent=`${c.name} ${W}×${H} · f${sf}`+(baseMode==='depth'&&!c.depth?' (no depth)':'');
  if(!isMain){const d=divergence(sf);if(d!=null)v.tag.textContent+=`  ·  Δmouse ${d.toFixed(1)}cm`;}
  if(!curReady){const[bn,bt]=loadStat(v);drawBuffering(ctx,W,H,bn,bt);}   // only when the frame you're viewing isn't ready
  return sf;}
function divergence(sf){if(!cmpOn||!CMP.S)return null;const a=V.S.objects.find(o=>o.role==='target'),b=CMP.S.objects.find(o=>o.role==='target');
  if(!a||!b)return null;const f=Math.min(sf,V.nf-1,CMP.nf-1);return norm(sub(oPos(V,f,a.id),oPos(CMP,f,b.id)))*100;}

// ---- panels (main view) ----
function renderGraph(v,sf,gid){const objs=trackedV(v);const gx={left:{x:62,y:90},right:{x:62,y:215}},OX=250;
  const n=objs.length,oy=i=>n<=1?150:42+i*(228/(n-1));let svg='';
  for(const side of['left','right']){const g=gPos(v,side,sf),G=gx[side],H=heatOf(objs,v,sf,g);
    objs.forEach((o,i)=>{const fr=H.frac[i],col=heat(fr),far=fr>0.66,x2=OX-47,y2=oy(i),mx=(G.x+20+x2)/2,my=(G.y+y2)/2;
      const lab=`${H.ds[i].toFixed(0)}cm ${dirShort(sub(oCenter(v,sf,o),g))}`.trim(),ww=lab.length*5.4+8;
      svg+=`<line x1="${G.x+20}" y1="${G.y}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${(1+(1-fr)*2.6).toFixed(1)}" ${far?'stroke-dasharray="4 3"':''} opacity="${far?0.6:1}" stroke-linecap="round"/>`
        +(i===H.closest?`<circle cx="${x2}" cy="${y2}" r="3.5" fill="${col}"/>`:'')
        +`<rect x="${mx-ww/2}" y="${my-7}" width="${ww}" height="13" rx="3" fill="#0d1117" opacity="${far?0.72:0.92}"/>`
        +`<text class="gx-elabel" x="${mx}" y="${my+3}" text-anchor="middle" fill="${col}" opacity="${far?0.75:1}">${lab}</text>`;});}
  for(const side of['left','right']){const G=gx[side];svg+=`<g class="gx-node"><circle cx="${G.x}" cy="${G.y}" r="19" fill="#1c2330" stroke="${GRIP_COL[side]}" stroke-width="2"/><text x="${G.x}" y="${G.y+4}" text-anchor="middle" fill="#e6edf3">${side==='left'?'L':'R'}</text></g>`;}
  objs.forEach((o,i)=>{const y=oy(i);svg+=`<g class="gx-node"><rect x="${OX-45}" y="${y-11}" width="100" height="22" rx="5" fill="#1c2330" stroke="${css(o.color)}" stroke-width="1.5"/><rect x="${OX-38}" y="${y-4}" width="8" height="8" rx="2" fill="${css(o.color)}"/><text x="${OX-25}" y="${y+4}" fill="#e6edf3">${o.id} ${o.label}</text></g>`;});
  $(gid).innerHTML=svg;}
function fillHeader(v,sfx){const m=v.S.meta,sg=v.S.signals,oc=m.outcome||'';
  $('h'+sfx+'-id').textContent=m.scene_id;
  const b=$('h'+sfx+'-oc');b.textContent=oc;
  b.className='badge '+(oc.indexOf('success')>=0&&oc.indexOf('collision')<0?'good':oc.indexOf('collision')>=0?'warn':'bad');
  $('h'+sfx+'-pert').textContent=m.perturbation_type?('pert: '+m.perturbation_type):'baseline';
  $('h'+sfx+'-metrics').textContent=`place ${sg.place_error_cm}cm · moved ${sg.object_moved_cm}cm`;}
function headSparks(v,sfx,sf){const sg=v.S.signals;spark('h'+sfx+'-prog',sg.progress,sf,'#3fb950',0,1);
  spark('h'+sfx+'-safe',sg.safety,sf,'#58a6ff',0,1);
  spark('h'+sfx+'-speed',sg.target_speed_cm,sf,'#d29922',0,Math.max.apply(null,sg.target_speed_cm||[1]));}
function spark(id,arr,sf,col,lo,hi){const cv=$(id);if(!cv||!arr)return;const w=cv.width=cv.clientWidth||120,h=cv.height=cv.clientHeight||24,ctx=cv.getContext('2d');
  ctx.clearRect(0,0,w,h);ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.beginPath();const T=arr.length;
  for(let t=0;t<T;t++){const x=t/(T-1)*w,y=h-3-((arr[t]-lo)/(hi-lo))*(h-6);t?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();
  const cx=sf/(T-1)*w;ctx.strokeStyle='#e6edf3';ctx.globalAlpha=0.6;ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,h);ctx.stroke();ctx.globalAlpha=1;}
function renderMinimap(v,sf,mid){const cv=$(mid);if(!cv)return;const w=cv.width=cv.clientWidth||300,h=cv.height=cv.clientHeight||150,ctx=cv.getContext('2d'),b=v.bounds;if(!b)return;
  ctx.fillStyle='#1c2330';ctx.fillRect(0,0,w,h);const pad=14;
  const sx=x=>pad+(x-b.x0)/(b.x1-b.x0)*(w-2*pad),sy=y=>h-pad-(y-b.y0)/(b.y1-b.y0)*(h-2*pad);
  ctx.strokeStyle='#2b3340';ctx.strokeRect(pad,pad,w-2*pad,h-2*pad);
  for(const o of trackedV(v)){const p=oPos(v,sf,o.id);ctx.fillStyle=css(o.color);ctx.beginPath();ctx.arc(sx(p[0]),sy(p[1]),o.role==='target'?5:3.5,0,7);ctx.fill();}
  for(const side of['left','right']){const g=gPos(v,side,sf);ctx.fillStyle=GRIP_COL[side];ctx.beginPath();ctx.arc(sx(g[0]),sy(g[1]),4,0,7);ctx.fill();
    ctx.fillStyle='#0d1117';ctx.font='8px sans-serif';ctx.textAlign='center';ctx.fillText(side==='left'?'L':'R',sx(g[0]),sy(g[1])+3);ctx.textAlign='start';}
  ctx.fillStyle='#5b6470';ctx.font='9px monospace';ctx.fillText('top-down (world x→, y↑)',pad+2,h-4);}
function renderTimeline(sf){const cv=$('timeline');const w=cv.width=cv.clientWidth||1000,h=cv.height,ctx=cv.getContext('2d'),sg=V.S.signals,T=V.nf;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#1c2330';ctx.fillRect(0,0,w,h);
  if(sg.progress){ctx.strokeStyle='#3fb950';ctx.lineWidth=1.5;ctx.beginPath();for(let t=0;t<T;t++){const x=t/(T-1)*w,y=h-3-sg.progress[t]*(h-6);t?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();}
  for(const e of(sg.events||[])){const x=e.frame/(T-1)*w;ctx.strokeStyle=e.kind==='place'?'#3fb950':'#d29922';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
    ctx.fillStyle=e.kind==='place'?'#3fb950':'#d29922';ctx.font='9px sans-serif';ctx.fillText(e.label,x+2,11);}
  const cx=sf/(T-1)*w;ctx.strokeStyle='#58a6ff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,h);ctx.stroke();}
function updatePanels(sf){
  for(const o of V.S.objects.filter(x=>x.tracked)){const el=$(`m-${o.id}`);if(!el)continue;if(!objVis[o.id]){el.textContent='';continue;}
    const c=oCenter(V,sf,o);el.textContent=`L${(norm(sub(c,gPos(V,'left',sf)))*100).toFixed(0)} R${(norm(sub(c,gPos(V,'right',sf)))*100).toFixed(0)}cm`;}
  void 0;
  void 0;
  void 0;
  renderTimeline(sf);}

// ---- success recompute ----
function recomputeSuccess(){const sg=V.S.signals,T=V.nf;let pf=null;
  for(let t=Math.floor(T/2);t<T;t++)if(sg.dist_xy_cm[t]<success.eps&&sg.target_height_cm[t]<success.h){pf=t;break;}
  $('succ').textContent=pf!=null?`✓ placed @ f${pf}`:'✗ not placed';$('succ').style.color=pf!=null?'var(--good)':'var(--bad)';}

// ---- scene loading ----
async function loadInto(v,sceneRef){const s=await(await fetch(sceneRef)).json();v.S=s;v.nf=s.meta.n_frames;
  v.CAMS=s.cameras.filter(c=>c.rgb);if(!v.CAMS.find(c=>c.name===v.camName))v.camName=(v.CAMS.find(c=>c.name==='head_camera')||v.CAMS[0]).name;
  v.lastReady=null;v.imgC.clear();v.depthC.clear();
  const xs=[],ys=[];for(const o of s.objects)if(o.tracked)for(let t=0;t<s.meta.n_frames;t+=10){const p=oPos(v,t,o.id);xs.push(p[0]);ys.push(p[1]);}
  for(const side of['left','right'])for(let t=0;t<s.meta.n_frames;t+=10){const g=gPos(v,side,t);xs.push(g[0]);ys.push(g[1]);}
  const px=0.05;v.bounds={x0:Math.min(...xs)-px,x1:Math.max(...xs)+px,y0:Math.min(...ys)-px,y1:Math.max(...ys)+px};return s;}
async function loadScene(sceneRef){await loadInto(V,sceneRef);const m=V.S.meta;
  $('title').innerHTML=`Replayable State · <b>${m.task_name}</b>`;
  $('subtitle').textContent=`“${m.instruction}”  ·  ${m.n_frames} frames`;
  fillHeader(V,'');
  $('storage').textContent=`overlays from a ${V.S.storage.state_trace_kb} KB trace (pixels: ${V.S.storage.pixels_mb} MB).`;
  buildCams();buildObjs();recomputeSuccess();
  $('scrub').max=V.nf-1;setFrame(Math.min(frame,V.nf-1));preloadCam(V);needRender=true;
  if(cmpOn){cmpOn=false;$('cmpbox').classList.remove('on');$('cmpBtn').classList.remove('active');}}
function twinId(){const m=V.S.meta;if(m.twin_of)return m.twin_of;const t=IDX.scenes.find(s=>s.twin_of===m.scene_id);return t&&t.id;}
async function toggleCompare(){if(cmpOn){cmpOn=false;$('cmpbox').classList.remove('on');$('cmpBtn').classList.remove('active');needRender=true;return;}
  const id=twinId();if(!id){alert('This scene has no twin to compare.');return;}
  const ref=IDX.scenes.find(s=>s.id===id);CMP.camName=V.camName;await loadInto(CMP,ref.scene);fillHeader(CMP,'2');preloadCam(CMP);
  cmpOn=true;$('cmpbox').classList.add('on');$('cmpBtn').classList.add('active');needRender=true;}

// ---- panel builders ----
function buildCams(){const host=$('cams');host.innerHTML='';V.CAMS.forEach(c=>{const b=document.createElement('button');b.className='tbtn'+(c.name===V.camName?' active':'');
  b.textContent=c.name.replace('_camera','').replace('_',' ')+(c.depth?' ◫':'');
  b.onclick=()=>{V.camName=c.name;CMP.camName=c.name;[...host.children].forEach(x=>x.classList.remove('active'));b.classList.add('active');
    V.lastReady=null;CMP.lastReady=null;preloadCam(V);if(cmpOn)preloadCam(CMP);updateOccRow();needRender=true;};host.appendChild(b);});updateOccRow();}
function updateOccRow(){const c=camOf(V);$('row-occ').classList.toggle('dim',!(c&&c.depth));}
function buildObjs(){const host=$('objs');host.innerHTML='';for(const o of V.S.objects.filter(x=>x.tracked)){objVis[o.id]=true;
  const row=document.createElement('label');row.className='row';
  row.innerHTML=`<input type="checkbox" checked><span class="sw" style="background:${css(o.color)}"></span><span class="nm">${o.id} · ${o.label} <span style="color:var(--dim)">(${o.role})</span></span><span class="meta" id="m-${o.id}"></span>`;
  row.querySelector('input').onchange=e=>{objVis[o.id]=e.target.checked;needRender=true;};host.appendChild(row);}}
function setFrame(f){frame=Math.max(0,Math.min(V.nf-1,f|0));$('scrub').value=frame;$('flbl').textContent=`frame ${frame} / ${V.nf-1}`;needRender=true;}

// ---- click select ----
$('cv').addEventListener('click',ev=>{const c=camOf(V);if(!c)return;const r=ev.target.getBoundingClientRect();
  const scale=Math.min(r.width/c.width,r.height/c.height),ox=(r.width-c.width*scale)/2,oy=(r.height-c.height*scale)/2;
  const u=(ev.clientX-r.left-ox)/scale,vv=(ev.clientY-r.top-oy)/scale;const RT=camRT(c,Math.min(frame,V.nf-1)),K=c.K;
  let best=null,bd=24;for(const o of trackedV(V)){const q=project(oCenter(V,Math.min(frame,V.nf-1),o),RT,K);if(!q)continue;const d=Math.hypot(q[0]-u,q[1]-vv);if(d<bd){bd=d;best=o.id;}}
  selId=(best===selId)?null:best;needRender=true;});

// ---- boot ----
(async function(){
  IDX=await(await fetch('data/scenes.json')).json();
  const sel=$('sceneSel');IDX.scenes.forEach(s=>{const o=document.createElement('option');o.value=s.scene;o.textContent=`${s.id} (${s.outcome})`;sel.appendChild(o);});
  sel.onchange=()=>loadScene(sel.value);
  const dref=IDX.scenes.find(s=>s.id===IDX.default)||IDX.scenes[0];
  await loadScene(dref.scene);
  for(const k of Object.keys(layers)){const el=$('L-'+k);if(!el)continue;el.checked=layers[k];el.onchange=e=>{layers[k]=e.target.checked;needRender=true;};}
  function setAllLayers(val){for(const k of Object.keys(layers)){layers[k]=val;const el=$('L-'+k);if(el)el.checked=val;}needRender=true;}
  $('lay-all').onclick=()=>setAllLayers(true);$('lay-none').onclick=()=>setAllLayers(false);
  function setAllObjs(val){for(const o of V.S.objects.filter(x=>x.tracked))objVis[o.id]=val;document.querySelectorAll('#objs input').forEach(c=>c.checked=val);needRender=true;}
  $('obj-all').onclick=()=>setAllObjs(true);$('obj-none').onclick=()=>setAllObjs(false);
  document.querySelectorAll('#baseSel .tbtn').forEach(b=>b.onclick=()=>{baseMode=b.dataset.base;document.querySelectorAll('#baseSel .tbtn').forEach(x=>x.classList.toggle('active',x===b));preloadCam(V);needRender=true;});
  $('sl-eps').oninput=e=>{success.eps=+e.target.value;$('v-eps').textContent=success.eps.toFixed(1)+'cm';recomputeSuccess();};
  $('sl-h').oninput=e=>{success.h=+e.target.value;$('v-h').textContent=success.h.toFixed(1)+'cm';recomputeSuccess();};
  $('scrub').oninput=e=>setFrame(+e.target.value);
  $('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'❚❚':'▶';};
  $('cmpBtn').onclick=toggleCompare;
  $('snapBtn').onclick=()=>{const a=document.createElement('a');a.download=`${V.S.meta.scene_id}_f${frame}.png`;a.href=V.cv.toDataURL('image/png');a.click();};
  $('timeline').onclick=ev=>{const r=ev.target.getBoundingClientRect();setFrame((ev.clientX-r.left)/r.width*(V.nf-1));};

  function tick(now){try{
    const dt=(now-lastT)/1000;lastT=now;
    if(playing){acc+=dt*V.S.meta.fps;while(acc>=1){const nf=(frame+1)%V.nf;if(ready(imgFor(V,camOf(V),nf))){setFrame(nf);acc-=1;}else{acc=0;break;}}}
    if(needRender){const sf=renderView(V,true)||0;headSparks(V,'',sf);
      if(cmpOn&&CMP.S){const sf2=renderView(CMP,false)||0;headSparks(CMP,'2',sf2);}
      updatePanels(sf);
      let buffering=!ready(imgFor(V,camOf(V),Math.min(frame,V.nf-1)));   // only the visible frame gates the loop
      if(cmpOn&&CMP.S)buffering=buffering||!ready(imgFor(CMP,camOf(CMP),Math.min(frame,CMP.nf-1)));
      needRender=playing||buffering;}
  }catch(err){const el=$('camtag');if(el){el.style.color='#f85149';el.textContent='ERR: '+(err&&err.message||err);}if(!tick._logged){console.error(err);tick._logged=1;}}
    requestAnimationFrame(tick);}
  requestAnimationFrame(tick);
})().catch(err=>{document.body.innerHTML=`<pre style="color:#f85149;padding:20px">Failed to start:\n${err.stack||err}\n\nServe via serve.py (a local server), not file://</pre>`;});
