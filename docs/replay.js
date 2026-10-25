/* Display recorded dataset states; this player never runs a physics simulation. */
const letters=document.getElementById('letters'),angle=document.getElementById('angle');
const canvas=document.getElementById('replay'),ctx=canvas.getContext('2d');
const status=document.getElementById('replay-status'),pause=document.getElementById('pause-replay');
const progress=document.getElementById('replay-progress'),speed=document.getElementById('replay-speed');
let selectedLetter='T',episode=null,manifest=null,elapsed=0,playing=true,last=0,request=0,controller=null,debounce;
const cache=new Map();
const manifestPromise=fetch('media/replays/manifest.json').then(r=>{if(!r.ok)throw Error('manifest');return r.json()}).then(m=>manifest=m);
// An error is presented by loadEpisode, including when the first fetch fails.
manifestPromise.catch(()=>{});
function setPlaying(value){playing=value;pause.textContent=value?'Pause':'Play';pause.setAttribute('aria-label',value?'Pause trajectory':'Play trajectory')}
function drawGlyph(rings,pose,fill,stroke){
 ctx.save();ctx.translate(pose[0],pose[1]);ctx.rotate(pose[2]);ctx.beginPath();
 for(const ring of rings){ring.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath()}
 ctx.fillStyle=fill;ctx.fill('evenodd');ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();ctx.restore();
}
function background(){
 ctx.fillStyle='#f1f4ef';ctx.fillRect(0,0,640,640);ctx.fillStyle='#d8e0d8';
 for(let x=30;x<630;x+=24)for(let y=30;y<630;y+=24){ctx.beginPath();ctx.arc(x,y,1,0,Math.PI*2);ctx.fill()}
 ctx.strokeStyle='#b8c6bc';ctx.lineWidth=2;ctx.strokeRect(10,10,620,620);
}
function draw(){
 background();if(!episode)return;
 const pos=Math.min(elapsed/episode.dt,episode.states.length-1),i=Math.floor(pos),j=Math.min(i+1,episode.states.length-1),f=pos-i;
 const a=episode.states[i],b=episode.states[j],s=a.map((v,k)=>v+(b[k]-v)*f);
 const delta=Math.atan2(Math.sin(b[4]-a[4]),Math.cos(b[4]-a[4]));s[4]=a[4]+delta*f;
 const rings=manifest.glyphs[episode.letter];
 drawGlyph(rings,episode.goal,'#d7e8e2','#2b9a8f');drawGlyph(rings,s.slice(2),'#ed7e43','#9e4b27');
 ctx.beginPath();ctx.arc(s[0],s[1],16,0,Math.PI*2);ctx.fillStyle='#1a282f';ctx.fill();
 ctx.beginPath();ctx.arc(s[0],s[1],12,0,Math.PI*2);ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke();
 progress.max=episode.states.length-1;progress.value=pos;
 document.getElementById('replay-time').textContent=`${Math.min(elapsed,(episode.states.length-1)*episode.dt).toFixed(1)} / ${((episode.states.length-1)*episode.dt).toFixed(1)} s`;
}
async function loadEpisode(){
 const token=++request;controller?.abort();controller=new AbortController();episode=null;elapsed=0;draw();
 const degrees=Number(angle.value)*30,letter=selectedLetter;
 const key='U'+letter.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')+'_'+String(degrees).padStart(3,'0');
 canvas.setAttribute('aria-label',`Recorded pushing trajectory: ${letter}, goal ${degrees} degrees`);
 status.textContent=`Loading ${letter} · ${degrees}°…`;pause.disabled=true;progress.disabled=true;
 try{
  const m=await manifestPromise;if(token!==request)return;
  let e=cache.get(key);
  if(!e){const response=await fetch('media/replays/'+m.episodes[key].file,{signal:controller.signal});if(!response.ok)throw Error('episode');e=await response.json();cache.set(key,e);if(cache.size>24)cache.delete(cache.keys().next().value)}
  if(token!==request)return;
  episode=e;elapsed=0;setPlaying(true);pause.disabled=false;progress.disabled=false;
  status.textContent=`${letter} · ${degrees}° · Demonstration 01 · ${m.episodes[key].steps} steps`;
  draw();
 }catch(error){if(token===request&&error.name!=='AbortError')status.textContent='Could not load this trajectory. Select the letter again to retry.'}
}
function updateAngle(){
 const degrees=Number(angle.value)*30;document.getElementById('degrees').textContent=degrees+'°';angle.setAttribute('aria-valuetext',degrees+' degrees');
 ++request;controller?.abort();episode=null;draw();status.textContent=`Loading ${selectedLetter} · ${degrees}°…`;
 clearTimeout(debounce);debounce=setTimeout(loadEpisode,120);
}
for(const ch of 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'){
 const button=document.createElement('button');button.type='button';button.className='letter';button.textContent=ch;
 button.setAttribute('aria-label',(ch===ch.toUpperCase()?'Uppercase ':'Lowercase ')+ch);button.setAttribute('aria-pressed',String(ch===selectedLetter));
 button.addEventListener('click',()=>{selectedLetter=ch;for(const el of letters.children)el.setAttribute('aria-pressed',String(el===button));clearTimeout(debounce);loadEpisode()});letters.append(button);
}
angle.addEventListener('input',updateAngle);
pause.addEventListener('click',()=>setPlaying(!playing));
document.getElementById('restart-replay').addEventListener('click',()=>{elapsed=0;setPlaying(true);draw()});
progress.addEventListener('input',()=>{if(episode){elapsed=Number(progress.value)*episode.dt;draw()}});
function tick(now){
 const dt=last?Math.min((now-last)/1000,.1):0;last=now;
 if(episode&&playing&&!document.hidden){elapsed+=dt*Number(speed.value);if(elapsed>(episode.states.length-1)*episode.dt+Number(speed.value))elapsed=0;draw()}
 requestAnimationFrame(tick);
}
loadEpisode();requestAnimationFrame(tick);
