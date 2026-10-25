/* Display recorded dataset states; this player never runs a physics simulation. */
const letters=document.getElementById('letters'),angle=document.getElementById('angle');
const canvas=document.getElementById('replay'),ctx=canvas.getContext('2d');
const status=document.getElementById('replay-status'),progress=document.getElementById('replay-progress');
let selectedLetter='T',episode=null,manifest=null,elapsed=0,last=0,request=0,transitionStart=0;
const cache=new Map(),inflight=new Map();
const previousFrame=document.createElement('canvas');previousFrame.width=previousFrame.height=640;
const previousCtx=previousFrame.getContext('2d');
const reduceMotion=matchMedia('(prefers-reduced-motion: reduce)').matches;
const manifestPromise=fetch('media/replays/manifest.json?v=continuous').then(r=>{if(!r.ok)throw Error('manifest');return r.json()}).then(m=>manifest=m);
manifestPromise.catch(()=>{});
function bundleName(letter){return 'U'+letter.codePointAt(0).toString(16).toUpperCase().padStart(4,'0')+'.json'}
function getBundle(letter){
 const name=bundleName(letter);
 if(cache.has(name))return Promise.resolve(cache.get(name));
 if(inflight.has(name))return inflight.get(name);
 const pending=fetch('media/replays/'+name).then(r=>{if(!r.ok)throw Error('episode');return r.json()}).then(bundle=>{cache.set(name,bundle);return bundle}).finally(()=>inflight.delete(name));
 inflight.set(name,pending);return pending;
}
// Fetch all angles together, then warm the other letters in the background.
// A selection shares any in-flight fetch and never clears the visible canvas.
async function warmCache(){
 const queue=[...'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'].filter(c=>c!==selectedLetter);
 async function worker(){while(queue.length){const letter=queue.shift();try{await getBundle(letter)}catch{ /* A direct selection can retry. */ }}}
 await Promise.all([worker(),worker()]);
}
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
 if(transitionStart){const alpha=1-(performance.now()-transitionStart)/180;if(alpha>0){ctx.save();ctx.globalAlpha=alpha;ctx.drawImage(previousFrame,0,0);ctx.restore()}else transitionStart=0;}
 progress.max=episode.states.length-1;progress.value=pos;
 document.getElementById('replay-time').textContent=`${Math.min(elapsed,(episode.states.length-1)*episode.dt).toFixed(1)} / ${((episode.states.length-1)*episode.dt).toFixed(1)} s`;
}
async function loadEpisode(){
 const token=++request,degrees=Number(angle.value)*30,letter=selectedLetter;
 const key=bundleName(letter).slice(0,-5)+'_'+String(degrees).padStart(3,'0');
 if(episode?.letter===letter&&episode.angle===degrees)return;
 if(!episode)status.textContent='Loading recorded trajectory…';
 try{
  const [m,bundle]=await Promise.all([manifestPromise,getBundle(letter)]);
  if(token!==request)return;
  const next=bundle[key];if(!next)throw Error('episode');
  if(episode&&!reduceMotion){previousCtx.clearRect(0,0,640,640);previousCtx.drawImage(canvas,0,0);transitionStart=performance.now()}
  episode=next;elapsed=0;progress.disabled=false;
  canvas.setAttribute('aria-label',`Recorded pushing trajectory: ${letter}, goal ${degrees} degrees`);
  status.textContent=`${letter} · ${degrees}° · ${m.episodes[key].steps} steps`;
  draw();
 }catch(error){if(token===request)status.textContent=episode?'Could not switch trajectory. Select again to retry.':'Could not load trajectory. Refresh to retry.'}
}
function updateAngle(){
 const degrees=Number(angle.value)*30;document.getElementById('degrees').textContent=degrees+'°';angle.setAttribute('aria-valuetext',degrees+' degrees');
 loadEpisode();
}
for(const ch of 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'){
 const button=document.createElement('button');button.type='button';button.className='letter';button.textContent=ch;
 button.setAttribute('aria-label',(ch===ch.toUpperCase()?'Uppercase ':'Lowercase ')+ch);button.setAttribute('aria-pressed',String(ch===selectedLetter));
 button.addEventListener('click',()=>{selectedLetter=ch;for(const el of letters.children)el.setAttribute('aria-pressed',String(el===button));loadEpisode()});letters.append(button);
}
angle.addEventListener('input',updateAngle);
progress.addEventListener('input',()=>{if(episode){elapsed=Number(progress.value)*episode.dt;draw()}});
function tick(now){
 const dt=last?Math.min((now-last)/1000,.1):0;last=now;
 if(episode&&!document.hidden){elapsed+=dt*2;if(elapsed>(episode.states.length-1)*episode.dt+2)elapsed=0;draw()}
 requestAnimationFrame(tick);
}
background();loadEpisode().then(()=>setTimeout(warmCache,1200));requestAnimationFrame(tick);
