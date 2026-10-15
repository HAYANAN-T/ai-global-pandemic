"""Build results/dashboard.html: a self-contained interactive dashboard (no external libraries).
Choose a scenario, play/pause/reset/replay the outbreak for every strategy on the same seed,
and read the 20-seed summary with 95% intervals versus the baseline."""
import json
from world import load_config
from experiment_runner import run
from strategies import REFERENCE

cfg = load_config(); A = json.load(open("results/analysis.json"))
data = {"cities": [{"name": c["name"], "x": c["x"], "y": c["y"], "pop": c["population"]} for c in cfg["cities"]],
        "links": cfg["travel_links"], "scenarios": {}, "summary": A["table"], "vs": A["vs_baseline"]}
for sc in cfg["scenarios"]:
    data["scenarios"][sc] = {}
    for n, fn in REFERENCE.items():
        m, h, log = run(fn, n, sc, 101, cfg, keep_history=True)
        data["scenarios"][sc][n] = dict(inf=[x["active_infectious"] for x in h],
            occ=[round(100 * float(x["occupancy"].sum() / x["beds"].sum()), 1) for x in h],
            trust=[round(x["trust"], 3) for x in h], econ=[round(x["activity"], 3) for x in h],
            city=[[round(v, 5) for v in x["city_infectious"]] for x in h], active=[x["active"] for x in h],
            log=[dict(day=l["day"], acc=l["accepted"], rej=[r[0] for r in l["rejected"]]) for l in log])
html = """<!doctype html><html><head><meta charset="utf-8"><title>Meridia outbreak dashboard</title>
<style>body{font-family:system-ui,sans-serif;margin:16px;color:#222}canvas{border:1px solid #ccc;background:#fafafa}
.row{display:flex;gap:16px;flex-wrap:wrap}.box{flex:1;min-width:420px}table{border-collapse:collapse;font-size:13px}
td,th{border:1px solid #ddd;padding:4px 8px;text-align:right}th{background:#f3f3f3}#log{font-size:12px;height:160px;overflow:auto;border:1px solid #ddd;padding:6px}</style></head><body>
<h2>Meridia outbreak dashboard</h2><div>Scenario <select id="sc"></select> Strategy for map <select id="st"></select>
<button id="play">Play</button><button id="pause">Pause</button><button id="reset">Reset</button>
Day <input id="day" type="range" min="0" max="179" value="0"> <span id="dl">0</span> (seed 101)</div>
<div class="row"><div class="box"><canvas id="map" width="520" height="440"></canvas></div>
<div class="box"><canvas id="c1" width="560" height="200"></canvas><canvas id="c2" width="560" height="200"></canvas></div></div>
<div class="row"><div class="box"><b>Policy log (cause and effect)</b><div id="log"></div></div>
<div class="box"><b>20-seed summary (means; deaths difference vs baseline with 95% interval)</b><div id="tbl"></div></div></div>
<script>const D=__DATA__;const COL={"Baseline (neutral rule)":"#777","Do nothing":"#d62728","Early proportionate":"#2ca02c","Strict lockdown":"#9467bd","Economy first":"#ff7f0e"};
const sc=document.getElementById('sc'),st=document.getElementById('st'),day=document.getElementById('day');let timer=null;
Object.keys(D.scenarios).forEach(s=>sc.add(new Option(s,s)));Object.keys(COL).forEach(s=>st.add(new Option(s,s)));
function line(id,key,label,ymax){const c=document.getElementById(id),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 const S=D.scenarios[sc.value];let mx=ymax||Math.max(...Object.values(S).map(v=>Math.max(...v[key])))*1.05;
 g.fillText(label,8,12);for(const [n,v] of Object.entries(S)){g.strokeStyle=COL[n];g.beginPath();v[key].forEach((y,i)=>{const X=30+i*(c.width-40)/179,Y=c.height-20-y/mx*(c.height-35);i?g.lineTo(X,Y):g.moveTo(X,Y)});g.stroke();}
 const X=30+(+day.value)*(c.width-40)/179;g.strokeStyle='#000';g.beginPath();g.moveTo(X,15);g.lineTo(X,c.height-20);g.stroke();}
function map(){const c=document.getElementById('map'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);const v=D.scenarios[sc.value][st.value],d=+day.value;
 g.strokeStyle='#888';D.links.forEach(([a,b,w])=>{g.lineWidth=1+120*w;g.beginPath();g.moveTo(D.cities[a].x*2.6+20,420-D.cities[a].y*2.6);g.lineTo(D.cities[b].x*2.6+20,420-D.cities[b].y*2.6);g.stroke()});g.lineWidth=1;
 D.cities.forEach((ct,i)=>{const r=Math.sqrt(ct.pop)/22,p=Math.min(1,v.city[d][i]*40);g.fillStyle=`rgb(255,${Math.round(255*(1-p))},${Math.round(255*(1-p))})`;
 g.beginPath();g.arc(ct.x*2.6+20,420-ct.y*2.6,r,0,6.3);g.fill();g.stroke();g.fillStyle='#000';g.fillText(ct.name+' '+(100*v.city[d][i]).toFixed(2)+'% infectious',ct.x*2.6+24+r,415-ct.y*2.6)});
 g.fillText('Active: '+(v.active[d].join(', ')||'none'),10,435);}
function logs(){const v=D.scenarios[sc.value][st.value];document.getElementById('log').innerHTML=v.log.map(l=>`day ${l.day}: <b>${l.acc.join(', ')||'nothing'}</b>${l.rej.length?' <span style="color:#c00">rejected: '+l.rej.join(', ')+'</span>':''}`).join('<br>');}
function table(){const S=D.summary[sc.value],V=D.vs[sc.value]||{};let h='<table><tr><th>strategy</th><th>deaths</th><th>vs baseline [95% CI]</th><th>economy</th><th>trust end</th><th>rejected</th></tr>';
 for(const [n,m] of Object.entries(S)){const c=V[n]&&V[n].deaths;h+=`<tr><td style="text-align:left">${n}</td><td>${m.deaths.toFixed(0)}</td><td>${c?c.mean_diff.toFixed(0)+' ['+c.ci[0].toFixed(0)+', '+c.ci[1].toFixed(0)+']':'-'}</td><td>${(100*m.economic_activity_retained).toFixed(1)}%</td><td>${m.trust_end.toFixed(2)}</td><td>${m.rejected_actions.toFixed(1)}</td></tr>`}
 document.getElementById('tbl').innerHTML=h+'</table>';}
function draw(){document.getElementById('dl').textContent=day.value;line('c1','inf','people infectious (all strategies)');line('c2','occ','hospital occupancy % (all strategies)');map();}
function all(){draw();logs();table();}
sc.onchange=all;st.onchange=all;day.oninput=draw;
document.getElementById('play').onclick=()=>{if(timer)return;timer=setInterval(()=>{day.value=(+day.value+1)%180;draw()},60)};
document.getElementById('pause').onclick=()=>{clearInterval(timer);timer=null};document.getElementById('reset').onclick=()=>{clearInterval(timer);timer=null;day.value=0;draw()};all();</script></body></html>"""
open("results/dashboard.html", "w").write(html.replace("__DATA__", json.dumps(data)))
print("dashboard written", round(len(html) / 1000), "kB template")
