/* Native modal dialogs preserve report controllers, keyboard focus and deep links. */
(()=>{'use strict';
const dialogs=[...document.querySelectorAll('.report-dialog')],launchers=[...document.querySelectorAll('.report-launcher')];let active=null,returnFocus=null;
function close(updateHash=true){if(!active)return;const d=active;active=null;d.querySelector('details').open=false;d.close();document.documentElement.classList.remove('report-modal-open');if(updateHash)history.replaceState(null,'',location.pathname+location.search+'#experiments');if(returnFocus?.isConnected)returnFocus.focus({preventScroll:true});returnFocus=null;}
function open(id,trigger){const d=document.getElementById(id+'-dialog');if(!dialogs.includes(d))return;if(active===d)return;if(active)close(false);returnFocus=trigger||launchers.find(b=>b.dataset.report===id);active=d;document.documentElement.classList.add('report-modal-open');d.showModal();d.scrollTop=0;d.querySelector('details').open=true;d.scrollTop=0;requestAnimationFrame(()=>{if(active===d)d.scrollTop=0;});history.replaceState(null,'',location.pathname+location.search+'#'+id);}
launchers.forEach(b=>b.onclick=()=>open(b.dataset.report,b));
for(const d of dialogs){d.querySelector('.report-dialog-close').onclick=()=>close();d.addEventListener('cancel',e=>{e.preventDefault();close();});let outsideDown=false;const outside=e=>{const r=d.getBoundingClientRect();return e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom;};d.addEventListener('pointerdown',e=>outsideDown=e.target===d&&outside(e));d.addEventListener('click',e=>{if(outsideDown&&e.target===d&&outside(e))close();outsideDown=false;});}
function route(){const id=location.hash.slice(1);if(dialogs.some(d=>d.id===id+'-dialog'))open(id);else close(false);}
document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(a&&dialogs.some(d=>d.id===a.hash.slice(1)+'-dialog')){e.preventDefault();open(a.hash.slice(1),a);}});
addEventListener('hashchange',route);addEventListener('load',()=>{if(active)requestAnimationFrame(()=>{if(active)active.scrollTop=0;});});route();
})();
