(() => {
  'use strict';
  const report = document.getElementById('imitation-learning');
  function controller(panel) {
  const version = panel.dataset.version;
  const get = id => document.getElementById(`${version}-${id}`);
  const slider = get('report-progress');
  const play = get('report-play');
  const final = get('report-final');
  const message = get('chart-message');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const NS = 'http://www.w3.org/2000/svg';
  let data, charts = [], progress = 100000, raf = 0, loading;
  const svgElement = (tag, attrs, text) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  function chart(id, title, mode, loss = false, split = '') {
    const mobile = matchMedia("(max-width:760px)").matches;
    const width = loss ? (mobile ? 360 : 480) : 280, height = loss ? 270 : 240;
    const box = {left:42, right:16, top:25, bottom:40};
    const w = width-box.left-box.right, h = height-box.top-box.bottom;
    const svg = svgElement('svg', {viewBox:`0 0 ${width} ${height}`, role:'img', 'aria-label':title});
    svg.append(svgElement('title', {}, title));
    const x = v => box.left+v/100000*w;
    const y = v => box.top+(loss ? (3-Math.log10(Math.max(v,1e-4)))/7 : (100-v)/100)*h;
    for (const value of loss ? [1000,10,.1,.001,.0001] : [0,25,50,75,100]) {
      svg.append(svgElement('line', {x1:box.left,x2:width-box.right,y1:y(value),y2:y(value),class:'grid'}));
      svg.append(svgElement('text', {x:box.left-9,y:y(value)+3,'text-anchor':'end'}, loss ? ({1000:'10³',10:'10¹','0.1':'10⁻¹','0.001':'10⁻³','0.0001':'10⁻⁴'}[value]) : value+'%'));
    }
    for (const value of loss ? [0,20000,40000,60000,80000,100000] : [0,50000,100000]) {
      svg.append(svgElement('text', {x:x(value),y:height-22,'text-anchor':'middle'}, value ? `${value/1000}k` : '0'));
    }
    svg.append(svgElement('text', {x:box.left+w/2,y:height-3,'text-anchor':'middle'}, 'Optimizer updates'));
    const series = data[mode].series.map((s, i) => {
      const line = svgElement('path', {class:'series-line',stroke:s.color});
      // Different dash patterns keep coincident zero-success lines identifiable.
      if (!loss) line.setAttribute('stroke-dasharray', ['2 5','8 4','12 3','none'][i]);
      const marks = svgElement('g', {});svg.append(line,marks);
      return {s,line,marks,points:loss ? s.loss.map(v=>({step:v.step,value:v.smooth})) : s.curve.map(v=>({step:v.step,value:v[split+'_success_rate']*100}))};
    });
    get(id).replaceChildren(svg);
    return {svg,series,x,y,loss};
  }
  function buildCharts() {
    charts = ['state','image'].flatMap(mode => [
      chart(`${mode}-loss-chart`, `${mode} training losses; logarithmic scale`, mode, true),
      chart(`${mode}-seen-chart`, `${mode} success on 52 seen training starts`, mode, false, 'train'),
      chart(`${mode}-unseen-chart`, `${mode} success on 52 unseen-angle cases`, mode, false, 'test')
    ]);
  }
  function draw(value) {
    progress = value; slider.value = value;
    get('report-step').value = Math.round(value/100)*100 >=100000 ? '100,000' : (Math.round(value/100)*100).toLocaleString('en-US');
    for (const c of charts) for (const s of c.series) {
      const points = s.points.filter(p=>p.step<=value);
      s.line.setAttribute('d', points.map((p,i)=>`${i?'L':'M'}${c.x(p.step).toFixed(2)},${c.y(p.value).toFixed(2)}`).join(' '));
      s.marks.replaceChildren();
      const dots = c.loss ? points.slice(-1) : points;
      for (const p of dots) {
        const dot=svgElement('circle',{cx:c.x(p.step),cy:c.y(p.value),r:c.loss?3:3.5,fill:s.s.color,stroke:'#e8eee6','stroke-width':1});
        dot.append(svgElement('title',{},`${s.s.label} · ${p.step.toLocaleString()} updates · ${c.loss?p.value.toPrecision(4):p.value.toFixed(2)+'%'}`));
        s.marks.append(dot);
      }
    }
  }
  function stop() { cancelAnimationFrame(raf);raf=0;play.textContent='Replay curves'; }
  function animate() {
    stop();const start=performance.now();draw(0);play.textContent='Pause';
    const frame = now => { draw(Math.min(100000,(now-start)/10000*100000));if(progress<100000)raf=requestAnimationFrame(frame);else stop(); };
    raf=requestAnimationFrame(frame);
  }
  async function load() {
    if(loading)return loading;
    loading=(async()=>{
      try {
        const modes = ['state','image'];
        const loaded = await Promise.all(modes.map(async mode => {
          const response = await fetch(`experiments/${mode}-${version}.json`);
          if (!response.ok) throw Error(response.status);
          return response.json();
        }));
        data = Object.fromEntries(modes.map((mode,i)=>[mode,loaded[i]]));
        buildCharts();
        slider.disabled=play.disabled=final.disabled=false;draw(100000);
        message.textContent='';
        message.hidden=true;

      } catch(error) {message.hidden=false;message.textContent='Interactive curves could not load. Use the recorded metrics and PDF downloads below.';loading=null;}
    })();return loading;
  }
  report.addEventListener('toggle',()=>{if(report.open && !panel.hidden)load();else stop();});
  slider.addEventListener('input',()=>{stop();draw(Number(slider.value));});
  play.addEventListener('click',()=>raf?stop():animate());
  final.addEventListener('click',()=>{stop();draw(100000);});
  reduced.addEventListener('change',()=>{if(reduced.matches && data){stop();draw(100000);}});
  matchMedia('(max-width:760px)').addEventListener('change',()=>{
    if(!data)return;
    buildCharts();
    draw(progress);
  });
  return {load,stop};
  }
  const panels = [...report.querySelectorAll('.il-panel')];
  const tabs = [...report.querySelectorAll('[role="tab"]')];
  const controllers = panels.map(controller);
  function select(index, focus=false) {
    controllers.forEach(c=>c.stop());
    panels.forEach((panel,i)=>{panel.hidden=i!==index;});
    tabs.forEach((tab,i)=>{tab.setAttribute('aria-selected',String(i===index));tab.tabIndex=i===index?0:-1;});
    if(focus)tabs[index].focus();
    controllers[index].load();
    report.dispatchEvent(new CustomEvent('il-version-change',{detail:tabs[index].dataset.version}));
  }
  tabs.forEach((tab,i)=>{
    tab.addEventListener('click',()=>select(i));
    tab.addEventListener('keydown',event=>{
      let index;
      if(event.key==='ArrowRight')index=(i+1)%tabs.length;
      if(event.key==='ArrowLeft')index=(i+tabs.length-1)%tabs.length;
      if(event.key==='Home')index=0;
      if(event.key==='End')index=tabs.length-1;
      if(index!==undefined){event.preventDefault();select(index,true);}
    });
  });
  if(report.open)controllers[0].load();
})();
