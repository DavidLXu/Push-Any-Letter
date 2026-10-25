/* Six ACT evaluation rollouts: recorded states, not simulated policy actions. */
(() => {
 'use strict';
 const root=document.getElementById('act-rollouts'),report=document.getElementById('imitation-learning');
 const play=document.getElementById('act-play'),restart=document.getElementById('act-restart'),slider=document.getElementById('act-progress'),speed=document.getElementById('act-speed');
 const pusherTrail=document.getElementById('act-pusher-trail'),objectTrail=document.getElementById('act-object-trail'),status=document.getElementById('act-load-status');
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 let data,loading=false,cards=[],step=0,playing=false,visible=false,last=0,raf=0,autoplayed=false;
 function glyph(ctx,rings,pose,fill,stroke){
  ctx.save();ctx.translate(pose[0],pose[1]);ctx.rotate(pose[2]);ctx.beginPath();
  for(const ring of rings){ring.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();}
  ctx.fillStyle=fill;ctx.fill('evenodd');ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();ctx.restore();
 }
 function trail(ctx,states,index,current,offset,color,dashed){
  ctx.save();ctx.strokeStyle=color;ctx.lineWidth=2.4;ctx.lineJoin='round';ctx.lineCap='round';if(dashed)ctx.setLineDash([7,6]);
  ctx.beginPath();ctx.moveTo(states[0][offset],states[0][offset+1]);
  for(let i=1;i<=index;i++)ctx.lineTo(states[i][offset],states[i][offset+1]);
  ctx.lineTo(current[offset],current[offset+1]);ctx.stroke();ctx.setLineDash([]);
  ctx.beginPath();ctx.arc(states[0][offset],states[0][offset+1],5,0,Math.PI*2);ctx.stroke();ctx.restore();
 }
 function render(){
  if(!data)return;
  slider.value=Math.floor(step);document.getElementById('act-time').value=`${(step/20).toFixed(1)} / 120.0 s`;
  for(const c of cards){
   const ep=c.ep,position=Math.min(step,ep.steps),i=Math.floor(position),j=Math.min(i+1,ep.steps),f=position-i;
   const a=ep.states[i],b=ep.states[j],s=a.map((v,k)=>v+(b[k]-v)*f);
   s[4]=a[4]+Math.atan2(Math.sin(b[4]-a[4]),Math.cos(b[4]-a[4]))*f;
   const ctx=c.ctx;ctx.drawImage(c.backdrop,0,0);
   if(pusherTrail.checked)trail(ctx,ep.states,i,s,0,'#567c91',false);
   if(objectTrail.checked)trail(ctx,ep.states,i,s,2,'#b7653d',true);
   glyph(ctx,data.glyphs[ep.letter],s.slice(2),'#ed7e43','#9e4b27');
   ctx.beginPath();ctx.arc(s[0],s[1],16,0,Math.PI*2);ctx.fillStyle='#1a282f';ctx.fill();
   ctx.beginPath();ctx.arc(s[0],s[1],12,0,Math.PI*2);ctx.lineWidth=2;ctx.strokeStyle='#fff';ctx.stroke();
   const finished=position>=ep.steps;
   c.figure.dataset.finished=String(finished);
   c.steps.textContent=`Step ${i.toLocaleString()} / ${ep.steps.toLocaleString()}`;
   c.coverage.textContent=`Coverage ${(ep.coverage[i]*100).toFixed(1)}%`;
   c.outcome.textContent=finished?(ep.success?'Success · final state held':'Timeout · final state held'):'Recorded rollout';
  }
 }
 function card(ep){
  const figure=document.createElement('figure');figure.className='rollout-card';figure.dataset.success=String(ep.success);figure.dataset.episode=ep.id;
  const header=document.createElement('header'),letter=document.createElement('span'),goal=document.createElement('span');
  letter.className='rollout-letter';letter.textContent=ep.letter;goal.className='rollout-goal';goal.textContent=`Goal ${ep.angle.toFixed(1)}° · test ${ep.slot+1}`;header.append(letter,goal);
  const canvas=document.createElement('canvas');canvas.width=canvas.height=640;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`ACT ${ep.letter}: ${ep.success?'successful':'failed'} test episode, goal ${ep.angle.toFixed(1)} degrees`);
  const caption=document.createElement('figcaption'),steps=document.createElement('span'),coverage=document.createElement('span');caption.append(steps,coverage);
  const outcome=document.createElement('p');outcome.className='rollout-outcome';figure.append(header,canvas,caption,outcome);
  document.getElementById(ep.success?'act-success-grid':'act-failure-grid').append(figure);
  const backdrop=document.createElement('canvas');backdrop.width=backdrop.height=640;const bg=backdrop.getContext('2d');
  bg.fillStyle='#f1f4ef';bg.fillRect(0,0,640,640);bg.fillStyle='#d8e0d8';
  for(let x=30;x<630;x+=24)for(let y=30;y<630;y+=24){bg.beginPath();bg.arc(x,y,1,0,Math.PI*2);bg.fill();}
  bg.strokeStyle='#b8c6bc';bg.lineWidth=2;bg.strokeRect(10,10,620,620);glyph(bg,data.glyphs[ep.letter],ep.goal,'#d7e8e2','#2b9a8f');
  return {ep,figure,ctx:canvas.getContext('2d'),backdrop,steps,coverage,outcome};
 }
 function stop(){playing=false;cancelAnimationFrame(raf);raf=0;last=0;play.textContent='Play';}
 function tick(now){
  if(!playing||!visible||!report.open||document.hidden){raf=0;last=0;return;}
  const delta=last?Math.min((now-last)/1000,.1):0;last=now;step=Math.min(2400,step+delta*20*Number(speed.value));render();
  if(step>=2400)stop();else raf=requestAnimationFrame(tick);
 }
 function resume(){if(playing&&visible&&report.open&&!document.hidden&&!raf){last=0;raf=requestAnimationFrame(tick);}}
 function start(){if(step>=2400)step=0;playing=true;play.textContent='Pause';resume();}
 function auto(){if(data&&visible&&report.open&&!autoplayed){autoplayed=true;if(!reduced.matches)start();}}
 async function load(){
  if(data||loading)return;loading=true;
  try{const r=await fetch('experiments/act-episodes.json');if(!r.ok)throw Error(r.status);const next=await r.json();
   if(next.episodes.length!==6||next.episodes.some(e=>e.states.length!==e.steps+1||e.coverage.length!==e.states.length))throw Error('Invalid recorded episode data');
   data=next;cards=data.episodes.map(card);play.disabled=restart.disabled=slider.disabled=false;
   status.textContent='Six verified test episodes · synchronized time · 4× by default. Playback pauses when the report is closed or off screen.';render();auto();
  }catch(error){status.textContent='Episodes could not load. Close and reopen the report to retry, or download the recorded JSON below.';}finally{loading=false;}
 }
 play.addEventListener('click',()=>playing?stop():start());
 restart.addEventListener('click',()=>{step=0;render();start();});
 slider.addEventListener('input',()=>{stop();step=Number(slider.value);render();});
 pusherTrail.addEventListener('change',render);objectTrail.addEventListener('change',render);
 report.addEventListener('toggle',()=>{if(report.open){load();resume();auto();}else{cancelAnimationFrame(raf);raf=0;last=0;}});
 new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(!visible){cancelAnimationFrame(raf);raf=0;last=0;}else{auto();resume();}},{threshold:0}).observe(root);
 document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelAnimationFrame(raf);raf=0;last=0;}else resume();});
 reduced.addEventListener('change',()=>{if(reduced.matches)stop();});
 if(report.open)load();
})();
