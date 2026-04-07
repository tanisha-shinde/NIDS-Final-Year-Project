/* ============================================================
   NIDS Dashboard — Real-Time JavaScript
   Connects to Flask-SocketIO, updates UI, manages charts
   ============================================================ */

'use strict';

// ── Socket.IO connection ──────────────────────────────────────
const socket = io({ transports: ['websocket', 'polling'] });

// ── Chart.js global defaults ─────────────────────────────────
Chart.defaults.color          = '#7a99b8';
Chart.defaults.borderColor    = 'rgba(0,255,170,.07)';
Chart.defaults.font.family    = "'JetBrains Mono', monospace";
Chart.defaults.font.size      = 10;

// ── Color map for attack types ────────────────────────────────
const ATK_COLORS = {
  NORMAL:      'rgba(57,255,20,.82)',
  DDoS:        'rgba(255,59,92,.88)',
  PortScan:    'rgba(255,140,0,.88)',
  BruteForce:  'rgba(155,89,255,.88)',
  Bot:         'rgba(0,150,255,.88)',
  Infiltration:'rgba(255,59,92,.65)',
  'ZERO-DAY':  'rgba(255,59,92,1)',
  ANOMALY:     'rgba(255,215,0,.88)',
};

const ATK_DOT_COLORS = {
  NORMAL:      '#39ff14',
  DDoS:        '#ff3b5c',
  PortScan:    '#ff8c00',
  BruteForce:  '#9b59ff',
  Bot:         '#0096ff',
  Infiltration:'#ff3b5c',
  'ZERO-DAY':  '#ff3b5c',
  ANOMALY:     '#ffd700',
};

const ATK_ICONS = {
  NORMAL:      '✓',
  DDoS:        '💥',
  PortScan:    '🔍',
  BruteForce:  '🔨',
  Bot:         '🤖',
  'ZERO-DAY':  '☠️',
  Infiltration:'🕵️',
  ANOMALY:     '⚠️',
};

// ── Traffic line chart ────────────────────────────────────────
const trafficChart = new Chart(
  document.getElementById('chart-traffic').getContext('2d'),
  {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Packets/s',
          data: [],
          borderColor: 'rgba(0,255,170,.9)',
          backgroundColor: 'rgba(0,255,170,.06)',
          borderWidth: 2, fill: true, tension: 0.4,
          pointRadius: 0, pointHoverRadius: 4,
        },
        {
          label: 'Attacks',
          data: [],
          borderColor: 'rgba(255,59,92,.9)',
          backgroundColor: 'rgba(255,59,92,.06)',
          borderWidth: 2, fill: true, tension: 0.4,
          pointRadius: 0, pointHoverRadius: 4,
          yAxisID: 'y2',
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 200 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position:'top', align:'end', labels:{ boxWidth:9, padding:14, font:{size:10} } },
        tooltip: {
          backgroundColor: 'rgba(12,20,32,.96)',
          borderColor: 'rgba(0,255,170,.18)', borderWidth: 1,
          titleFont:{size:9}, bodyFont:{size:9},
        }
      },
      scales: {
        x:  { grid:{color:'rgba(0,255,170,.04)'}, ticks:{maxRotation:0,maxTicksLimit:8} },
        y:  { grid:{color:'rgba(0,255,170,.04)'}, beginAtZero:true,
              title:{display:true,text:'Pkts/s',font:{size:9}} },
        y2: { position:'right', grid:{drawOnChartArea:false}, beginAtZero:true,
              ticks:{color:'rgba(255,59,92,.7)'},
              title:{display:true,text:'Attacks',font:{size:9},color:'rgba(255,59,92,.7)'} }
      }
    }
  }
);

// ── Attack donut chart ────────────────────────────────────────
const donutChart = new Chart(
  document.getElementById('chart-donut').getContext('2d'),
  {
    type: 'doughnut',
    data: {
      labels: ['No Data'],
      datasets: [{ data:[1], backgroundColor:['rgba(255,255,255,.05)'],
                   borderColor:['rgba(0,255,170,.1)'], borderWidth:1, hoverOffset:7 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '72%',
      animation: { duration: 400 },
      plugins: {
        legend: { position:'bottom', labels:{ boxWidth:8, padding:7, font:{size:9} } },
        tooltip: {
          backgroundColor:'rgba(12,20,32,.96)',
          borderColor:'rgba(0,255,170,.18)', borderWidth:1,
          callbacks: {
            label: ctx => {
              const total = ctx.dataset.data.reduce((a,b)=>a+b,0) || 1;
              return ` ${ctx.label}: ${ctx.raw} (${((ctx.raw/total)*100).toFixed(1)}%)`;
            }
          }
        }
      }
    }
  }
);

// ═══════════════════════ SOCKET.IO ════════════════════════════

socket.on('connect', ()=>{
  console.log('[WS] Connected');
  setSystemStatus(true);
});
socket.on('disconnect', ()=>{
  console.log('[WS] Disconnected');
  setSystemStatus(false);
});
socket.on('initial_data', data=>{
  if (data.stats)   updateStats(data.stats);
  if (data.alerts)  populateTable(data.alerts);
  if (data.history) updateTrafficChart(data.history);
  fetchHealth();
});
socket.on('stats_update', stats=>{
  updateStats(stats);
  updateTrafficChart(stats.traffic_history || []);
});
socket.on('new_alert', alert=>{
  prependRow(alert);
  if (alert.is_attack) showToast(alert);
});

// ═══════════════════════ UI UPDATES ═══════════════════════════

function setSystemStatus(online){
  const dot  = document.getElementById('dot-system');
  const txt  = document.getElementById('txt-system');
  if (online){
    dot.className = 'chip-dot chip-dot-green pulse-dot';
    txt.textContent = 'Online';
  } else {
    dot.className = 'chip-dot chip-dot-red';
    txt.textContent = 'Offline';
  }
}

// Animated number counter
function countTo(id, target){
  const el = document.getElementById(id);
  if (!el) return;
  const current = parseInt(el.textContent.replace(/,/g,'')) || 0;
  if (current === target) return;
  const diff = target - current;
  const steps = Math.min(Math.abs(diff), 18);
  const inc   = diff / steps;
  let val = current, step = 0;
  const t = setInterval(()=>{
    val  += inc; step++;
    el.textContent = Math.round(val).toLocaleString();
    if (step >= steps){ el.textContent = target.toLocaleString(); clearInterval(t); }
  }, 28);
}

function updateStats(s){
  countTo('sc-packets', s.total_packets    || 0);
  countTo('sc-flows',   s.total_flows      || 0);
  countTo('sc-attacks', s.total_attacks    || 0);

  // Inferences badge
  const el = document.getElementById('badge-inf');
  if (el) el.textContent = (s.inference_count || 0).toLocaleString() + ' inferences';

  // Attacks badge
  const atk = document.getElementById('badge-atk');
  if (atk) atk.textContent = s.total_attacks > 0
    ? s.total_attacks + ' attack(s)' : 'No attacks';

  // ESP32 dot
  const esp = document.getElementById('dot-esp32');
  if (esp) esp.className = 'chip-dot ms-1 ' + (s.esp32_online ? 'chip-dot-green pulse-dot' : 'chip-dot-yellow');

  // Model badges
  const ml = s.models_loaded || {};
  setModelBadge('badge-lstm', ml.lstm);
  setModelBadge('badge-cnn',  ml.cnn);
  setModelBadge('badge-ae',   ml.autoencoder);

  // ESP32 row badge
  const erow = document.getElementById('badge-esp32-row');
  if (erow){
    erow.className  = 'model-badge ' + (s.esp32_online ? 'mb-online' : 'mb-offline');
    erow.textContent = s.esp32_online ? 'Online' : 'Offline';
  }

  // Loaded model count
  const loaded = Object.values(ml).filter(Boolean).length;
  const scm = document.getElementById('sc-models');
  if (scm) scm.textContent = loaded;

  // Donut + breakdown
  updateDonut(s.attack_counts || {});
  updateBreakdown(s.attack_counts || {});

  // Alert count badge
  const bc = document.getElementById('badge-count');
  if (bc) bc.textContent = s.total_attacks || 0;
}

function setModelBadge(id, loaded){
  const el = document.getElementById(id);
  if (!el) return;
  if (loaded === true)  { el.className='model-badge mb-loaded';  el.textContent='Loaded' }
  else if (loaded===false){ el.className='model-badge mb-demo'; el.textContent='Demo'   }
  else                  { el.className='model-badge mb-offline'; el.textContent='...'   }
}

// ── Charts ────────────────────────────────────────────────────

function updateTrafficChart(history){
  if (!history || !history.length) return;
  trafficChart.data.labels            = history.map(h=>h.time||'');
  trafficChart.data.datasets[0].data  = history.map(h=>h.pps||0);
  trafficChart.data.datasets[1].data  = history.map(h=>h.attacks||0);
  trafficChart.update('none');

  // Update PPS badge from last point
  const last = history[history.length-1];
  const pps  = document.getElementById('badge-pps');
  if (pps && last) pps.textContent = (last.pps||0).toFixed(1) + ' pps';
}

function updateDonut(counts){
  const entries = Object.entries(counts).filter(([,v])=>v>0);
  if (!entries.length){
    donutChart.data.labels = ['No Attacks'];
    donutChart.data.datasets[0].data            = [1];
    donutChart.data.datasets[0].backgroundColor = ['rgba(255,255,255,.05)'];
    donutChart.data.datasets[0].borderColor     = ['rgba(0,255,170,.1)'];
  } else {
    donutChart.data.labels = entries.map(([k])=>k);
    donutChart.data.datasets[0].data            = entries.map(([,v])=>v);
    donutChart.data.datasets[0].backgroundColor = entries.map(([k])=>ATK_COLORS[k]||'rgba(255,215,0,.8)');
    donutChart.data.datasets[0].borderColor     = entries.map(([k])=>(ATK_COLORS[k]||'rgba(255,215,0,.8)').replace(/[\d.]+\)$/,'1)'));
  }
  const total = entries.reduce((s,[,v])=>s+v,0);
  const dn    = document.getElementById('donut-num');
  if (dn) dn.textContent = total.toLocaleString();
  donutChart.update('none');
}

function updateBreakdown(counts){
  const box = document.getElementById('atk-breakdown');
  if (!box) return;
  const entries = Object.entries(counts);
  const total   = entries.reduce((s,[,v])=>s+v,0) || 1;
  if (!entries.some(([,v])=>v>0)){
    box.innerHTML = '<div class="empty-info"><i class="fas fa-shield-check me-2"></i>All clear — monitoring…</div>';
    return;
  }
  box.innerHTML = entries.map(([label,count])=>{
    const pct   = Math.round((count/total)*100);
    const color = ATK_DOT_COLORS[label] || '#7a99b8';
    return `<div class="atk-item">
      <div class="atk-dot" style="background:${color};box-shadow:0 0 4px ${color}"></div>
      <div class="atk-name">${label}</div>
      <div class="atk-bar-wrap"><div class="atk-bar" style="width:${pct}%;background:${color}"></div></div>
      <div class="atk-num">${count}</div>
    </div>`;
  }).join('');
}

// ═══════════════════════ ALERT TABLE ══════════════════════════

const MAX_ROWS = 100;
let rowCounter = 0;

function getLblClass(label){
  const safe = (label||'').replace(/[^a-zA-Z0-9-]/g,'-');
  return 'lbl-chip lbl-' + (safe || 'other');
}
function getSevClass(sev){
  return 'sev sev-' + (sev||'info');
}
function fmtTime(iso){
  try { return new Date(iso).toLocaleTimeString() } catch{ return iso||'' }
}
function fmtFlow(key){
  if (!key) return '—';
  const p = String(key).replace(/[()' ]/g,'').split(',');
  if (p.length>=4) return `${p[0]}:${p[2]} → ${p[1]}:${p[3]}`;
  return String(key).slice(0,30);
}

function populateTable(alerts){
  const tbody = document.getElementById('alert-tbody');
  tbody.innerHTML = '';
  rowCounter = 0;
  if (!alerts || !alerts.length){
    tbody.innerHTML = `<tr class="empty-tr"><td colspan="9">
      <i class="fas fa-shield-check me-2" style="color:var(--green)"></i>No alerts yet — monitoring…</td></tr>`;
    return;
  }
  alerts.slice(0,MAX_ROWS).forEach(a=>tbody.appendChild(buildRow(a,false)));
  rowCounter = alerts.length;
}

function prependRow(alert){
  const tbody = document.getElementById('alert-tbody');
  const empty = tbody.querySelector('.empty-tr');
  if (empty) empty.remove();
  tbody.insertBefore(buildRow(alert,true), tbody.firstChild);
  rowCounter++;
  while(tbody.children.length > MAX_ROWS) tbody.removeChild(tbody.lastChild);
  document.getElementById('badge-count').textContent = rowCounter;
}

function buildRow(a, animate){
  const tr  = document.createElement('tr');
  if (animate) tr.className = 'row-new';
  const conf  = ((a.confidence||0)*100).toFixed(1);
  const anomaly = (a.anomaly_score||0).toFixed(4);
  const isHigh  = parseFloat(anomaly) > 0.05;
  tr.innerHTML = `
    <td style="color:var(--t3)">${a.id||'—'}</td>
    <td style="white-space:nowrap">${fmtTime(a.timestamp)}</td>
    <td><span class="${getLblClass(a.label)}">${ATK_ICONS[a.label]||'⚡'} ${a.label||'?'}</span></td>
    <td><span class="${getSevClass(a.severity)}">${a.severity||'info'}</span></td>
    <td>
      <div class="conf-wrap">
        <div class="conf-track"><div class="conf-fill" style="width:${conf}%"></div></div>
        <span class="conf-txt">${conf}%</span>
      </div>
    </td>
    <td style="color:${isHigh?'var(--red)':'var(--t3)'};font-size:.68rem">${anomaly}</td>
    <td>${a.pkt_count||0}</td>
    <td>${(a.duration||0).toFixed(2)}s</td>
    <td style="color:var(--t3);font-size:.65rem" title="${a.flow_key||''}">${fmtFlow(a.flow_key)}</td>`;
  return tr;
}

// ═══════════════════════ TOAST ════════════════════════════════

function showToast(a){
  const box   = document.getElementById('toast-box');
  const sev   = a.severity||'medium';
  const emojis= {critical:'🚨',high:'🔴',medium:'⚠️',info:'ℹ️'};
  const t     = document.createElement('div');
  t.className = `toast-item toast-${sev}`;
  t.innerHTML = `
    <div class="toast-title">${emojis[sev]||'⚠️'} ${a.label} Detected</div>
    <div class="toast-body">Conf: ${((a.confidence||0)*100).toFixed(1)}%  |  Pkts: ${a.pkt_count||0}  |  ${new Date().toLocaleTimeString()}</div>`;
  box.appendChild(t);
  setTimeout(()=>{
    t.style.animation='toast-out .3s ease forwards';
    setTimeout(()=>t.remove(),320);
  }, 5000);
}

// ═══════════════════════ HEALTH ═══════════════════════════════

function fetchHealth(){
  fetch('/api/system').then(r=>r.json()).then(updateHealth).catch(()=>{});
}

function updateHealth(d){
  if (d.error) return;
  const setBar = (id,barId,val,cls)=>{
    const el=document.getElementById(id); if(el)el.textContent=val;
    const b=document.getElementById(barId); if(b){b.style.width=Math.min(val,100)+'%'; if(cls)b.className='progress-bar '+cls;}
  };
  const cpu=d.cpu_percent||0, ram=d.ram_percent||0, disk=d.disk_percent||0;
  setBar('h-cpu','bar-cpu',cpu.toFixed(1)+'%', cpu>80?'bg-red':cpu>60?'bg-orange':'bg-cyan');
  setBar('h-ram','bar-ram',ram.toFixed(1)+'%', ram>85?'bg-red':'bg-purple');
  setBar('h-disk','bar-disk',disk.toFixed(1)+'%','bg-orange');
  if (d.cpu_temp!=null){
    const tmp=d.cpu_temp;
    setBar('h-temp','bar-temp',tmp.toFixed(1)+'°C', tmp>70?'bg-red':tmp>55?'bg-orange':'bg-cyan');
  } else {
    const el=document.getElementById('h-temp'); if(el)el.textContent='N/A';
  }
  const ts=document.getElementById('health-ts');
  if(ts)ts.textContent=new Date().toLocaleTimeString();
}

setInterval(fetchHealth, 10000);
setTimeout(fetchHealth, 800);

// ═══════════════════════ CLEAR ════════════════════════════════

function clearAlerts(){
  fetch('/api/alerts/clear',{method:'POST'}).then(()=>{
    document.getElementById('alert-tbody').innerHTML=`<tr class="empty-tr">
      <td colspan="9"><i class="fas fa-shield-check me-2" style="color:var(--green)"></i>Cleared — monitoring…</td></tr>`;
    document.getElementById('badge-count').textContent='0';
    rowCounter=0;
  });
}

// ═══════════════════════ CLOCK ════════════════════════════════

function updateClock(){
  const el=document.getElementById('nav-clock');
  if(el) el.textContent=new Date().toLocaleTimeString('en-GB',{hour12:false});
}
setInterval(updateClock, 1000);
updateClock();

// ═══════════════════════ PARTICLES ════════════════════════════

(function spawnParticles(){
  const container=document.getElementById('bg-particles');
  if (!container) return;
  for (let i=0;i<25;i++){
    const p=document.createElement('div');
    const size=Math.random()*2.5+1;
    const dur =Math.random()*22+14;
    p.className='particle';
    p.style.cssText=`
      width:${size}px;height:${size}px;
      left:${Math.random()*100}%;
      top:${Math.random()*100}%;
      animation-duration:${dur}s;
      animation-delay:${-Math.random()*dur}s;
      opacity:0;
    `;
    container.appendChild(p);
  }
})();
