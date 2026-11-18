"""Choose the cases for the clinician panel (manuscript 3.5 stage 9) and build a blinded, scrollable viewer with a rating form.

    python src/select_panel_cases.py --npy-dir D:\\tcia\\processed --out D:\\panel                  # main panel: inside CT Colonography
    python src/select_panel_cases.py --npy-dir D:\\tcia\\processed --out D:\\panel_cross --design cross   # secondary: StageII vs CT Colonography

Question the raters answer for every case: "Is there a lesion that needs action (cancer, or an advanced adenoma: 10 mm or larger,
or tubulovillous / villous)?"  yes / no, confidence 1-5, and a guess of the scan type (screening CT Colonography or staging CT).

Designs
  within (default, MAIN)  Only CT Colonography patients, so the scan source cannot give the answer away.
        positive = advanced adenoma (validation + test patients) and the 3 held-out cancers;
        negative = non-advanced adenoma and non-neoplastic (hyperplastic / normal) patients (validation + test).
        Same groups as within_ctc_cv.py, so the clinicians and the models answer the same question.  Small and exploratory.
  cross (SECONDARY)       StageII malignant vs CT Colonography benign test patients (as the main model test).  Clinicians can tell
        these sources apart by the picture (oral/IV contrast, collapsed vs air-filled colon), so this panel measures source recognition as
        well; the scan-type guess column documents that.  The 3 CT Colonography cancers are added to the positives.
Selection never looks at model output.  Order is shuffled with a fixed seed (20261023); case numbers carry no information.
--repeats N shows N cases a second time under a new number (rater self-consistency).

What the raters see: for each case ONE scan (the series with most kept colon slabs), ALL kept colon slabs, scrollable (mouse wheel, arrow keys
or slider), same window as the models (-400..400 HU), cropped to the body.  No names, labels, scan identifiers or slab numbers in the pages.

Outputs in --out
  site/index.html, site/case_NNN.html   the viewer.  Serve it:   cd <out>\\site   then   python -m http.server 8000   and open http://localhost:8000
                                       (a local server is needed so the browser can keep the ratings).  Each rater types their id on the
                                       index page, rates the cases, and clicks "Download my ratings (CSV)".
  panel_key.csv                        PRIVATE: case_id -> patient_id, series_uid, group, true answer (1 = needs action), split, source,
                                       is_repeat, repeat_of_case, design.  Keep it away from the raters.
  panel_cases.csv                      patient_id, series_uid, slab_idx (slab with the largest colon outline) -> gradcam.py --cases
Needs: numpy pandas pillow (installed with matplotlib) pyarrow (colon_dataset.py and within_ctc_cv.py in the same folder)
"""
import argparse, base64, io, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colon_dataset as cd
import within_ctc_cv as w

ROOT = Path(__file__).resolve().parent.parent
QUESTION = "Is there a lesion that needs action: cancer, or an advanced adenoma (10 mm or larger, or tubulovillous / villous)?"

CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#16181c;color:#e8e8e8}
header{padding:10px 16px;background:#20242b;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
a{color:#7fb4ff}main{display:flex;gap:16px;padding:12px;flex-wrap:wrap}.view{flex:1 1 560px;min-width:320px}
#img{width:100%;max-width:760px;image-rendering:auto;background:#000;display:block}input[type=range]{width:100%;max-width:760px}
.form{flex:0 0 320px;background:#20242b;padding:14px;border-radius:8px}.form h3{margin:.2em 0 .6em;font-size:15px}
label{display:block;margin:.35em 0}select,textarea,input[type=text]{width:100%;box-sizing:border-box;background:#111;color:#eee;border:1px solid #444;padding:6px}
button{background:#2e86e4;color:#fff;border:0;padding:9px 14px;border-radius:6px;cursor:pointer;margin-top:10px}.ok{color:#6fd08c}.todo{color:#e4b45e}
table{border-collapse:collapse}td,th{padding:6px 12px;border-bottom:1px solid #333;text-align:left}"""

CASE_JS = """
const IMGS = __IMGS__, CASE = "__CASE__", N = IMGS.length;
let i = Math.floor(N/2); const img = document.getElementById('img'), sl = document.getElementById('sl'), pos = document.getElementById('pos');
sl.max = N-1; function show(k){ i = Math.max(0, Math.min(N-1, k)); img.src = IMGS[i]; sl.value = i; pos.textContent = (i+1)+' / '+N; }
sl.oninput = e => show(+e.target.value); img.onwheel = e => { e.preventDefault(); show(i + (e.deltaY>0?1:-1)); };
document.onkeydown = e => { if(e.key==='ArrowDown'||e.key==='ArrowRight') show(i+1); if(e.key==='ArrowUp'||e.key==='ArrowLeft') show(i-1); };
show(i);
const rater = localStorage.getItem('rater') || ''; document.getElementById('who').textContent = rater ? 'Rater: '+rater : 'No rater id set - go to the index page first';
const key = 'rating_'+rater+'_'+CASE, f = document.getElementById('f'); const saved = JSON.parse(localStorage.getItem(key) || 'null');
if (saved){ for (const k in saved){ const el = f.elements[k]; if(!el) continue; if(el.length && el[0] && el[0].type==='radio'){ for(const r of el) r.checked = (r.value===saved[k]); } else el.value = saved[k]; }
  document.getElementById('st').textContent = 'saved'; document.getElementById('st').className='ok'; }
f.onsubmit = e => { e.preventDefault(); if(!rater){ alert('Set your rater id on the index page first.'); return; }
  const d = {action: f.elements['action'].value, confidence: f.elements['confidence'].value, scan_type: f.elements['scan_type'].value, notes: f.elements['notes'].value};
  if(!d.action || !d.confidence || !d.scan_type){ alert('Please answer all three questions.'); return; }
  localStorage.setItem(key, JSON.stringify(d)); document.getElementById('st').textContent = 'saved'; document.getElementById('st').className='ok'; };
"""

CASE_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Case __CASE__</title><style>__CSS__</style></head><body>
<header><b>Case __CASE__</b><a href="index.html">&larr; all cases</a><span id="who"></span><span id="pos"></span><span id="st" class="todo">not saved</span></header>
<main><div class="view"><img id="img" alt="CT slice"><input id="sl" type="range" min="0" value="0"><small>Scroll with the mouse wheel, arrow keys or the slider. Window -400 to 400 HU, all kept colon slices of one scan.</small></div>
<form class="form" id="f"><h3>__Q__</h3>
<label><input type="radio" name="action" value="1"> Yes</label><label><input type="radio" name="action" value="0"> No</label>
<h3>Confidence</h3><select name="confidence"><option value="">choose</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>
<h3>What kind of scan do you think this is?</h3><select name="scan_type"><option value="">choose</option><option value="screening_ctc">Screening CT colonography</option><option value="staging_ct">Staging / diagnostic CT</option><option value="unsure">Unsure</option></select>
<h3>Notes (optional)</h3><textarea name="notes" rows="3"></textarea><button type="submit">Save rating</button></form></main><script>__JS__</script></body></html>"""

INDEX_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Clinician panel</title><style>__CSS__</style></head><body>
<header><b>Clinician panel</b><span>Rater id: <input type="text" id="rid" style="width:140px"> <button id="setr" style="margin:0">Set</button></span></header>
<main><div><p>__INTRO__</p><table id="t"><tr><th>Case</th><th>Status</th></tr></table><p><button id="dl">Download my ratings (CSV)</button></p></div></main>
<script>const CASES = __CASES__;
const rid = document.getElementById('rid'); rid.value = localStorage.getItem('rater') || '';
document.getElementById('setr').onclick = () => { localStorage.setItem('rater', rid.value.trim()); location.reload(); };
const t = document.getElementById('t'), rater = localStorage.getItem('rater') || '';
CASES.forEach(c => { const v = localStorage.getItem('rating_'+rater+'_'+c); const r = t.insertRow(); r.insertCell().innerHTML = '<a href="case_'+c+'.html">Case '+c+'</a>';
  const s = r.insertCell(); s.textContent = v ? 'rated' : 'to do'; s.className = v ? 'ok' : 'todo'; });
document.getElementById('dl').onclick = () => { if(!rater){ alert('Set your rater id first.'); return; }
  const q = s => '"'+String(s||'').replace(/"/g,'""')+'"'; let out = 'case_id,rater_id,needs_action,confidence,scan_type_guess,notes\\n';
  CASES.forEach(c => { const v = JSON.parse(localStorage.getItem('rating_'+rater+'_'+c) || 'null'); if(v) out += [c, q(rater), v.action, v.confidence, v.scan_type, q(v.notes)].join(',')+'\\n'; });
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([out], {type:'text/csv'})); a.download = 'ratings_'+rater+'.csv'; a.click(); };
</script></body></html>"""


def render_case(vol, z, quality=88):
    """All slabs z of a volume -> list of base64 JPEG data URIs, windowed and cropped to the body."""
    from PIL import Image
    hu = np.asarray(vol[z], np.float32); body = (hu > -300).any(0); rows, cols = np.where(body)
    if len(rows):
        r0, r1, c0, c1 = max(rows.min() - 6, 0), min(rows.max() + 7, hu.shape[1]), max(cols.min() - 6, 0), min(cols.max() + 7, hu.shape[2])
    else: r0, r1, c0, c1 = 0, hu.shape[1], 0, hu.shape[2]
    out = []
    for s in hu:
        u8 = ((np.clip(s[r0:r1, c0:c1], cd.WINDOW[0], cd.WINDOW[1]) - cd.WINDOW[0]) / (cd.WINDOW[1] - cd.WINDOW[0]) * 255).astype(np.uint8)
        im = Image.fromarray(u8).resize((u8.shape[1] * 2, u8.shape[0] * 2), Image.BICUBIC); b = io.BytesIO(); im.save(b, "JPEG", quality=quality)
        out.append("data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--design", choices=["within", "cross"], default="within")
    ap.add_argument("--slices", default=str(ROOT / "data/processed/colon_slices.csv"))
    ap.add_argument("--split", default=str(ROOT / "data/processed/cohort_split.csv"))
    ap.add_argument("--polyps", default=str(ROOT / "data/processed/polyps.parquet"))
    ap.add_argument("--panel-splits", nargs="+", default=["val", "test"], help="splits the panel patients come from")
    ap.add_argument("--n-pos", type=int, default=0, help="cap on positive patients (0 = all available)")
    ap.add_argument("--n-neg", type=int, default=0, help="cap on negative patients (0 = all available)")
    ap.add_argument("--include-challenge", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--seed", type=int, default=cd.SEED)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); out = Path(a.out); site = out / "site"; site.mkdir(parents=True, exist_ok=True)
    cs = pd.read_csv(a.split)[["patient_id", "split", "label"]]; raw = pd.read_csv(a.slices)
    sl = raw.drop(columns=[c for c in ["split", "label"] if c in raw.columns]).merge(cs, on="patient_id")
    have = set(sl.patient_id)
    if a.design == "within":
        pt = w.patient_table(a.polyps, a.split); pt = pt[pt.patient_id.isin(have) & pt.split.isin(a.panel_splits)]
        pos = pt[pt.group == "advanced_adenoma"][["patient_id", "group"]]; neg = pt[pt.group.isin(["nonadvanced_adenoma", "nonneoplastic"])][["patient_id", "group"]]
    else:
        P = sl.drop_duplicates("patient_id"); P = P[P.split.isin(a.panel_splits)]
        pos = P[P.label == "malignant"][["patient_id"]].assign(group="stageII_cancer"); neg = P[P.label == "benign"][["patient_id"]].assign(group="ctc_benign")
    pos = pos.sort_values("patient_id"); neg = neg.sort_values("patient_id")
    if a.n_pos: pos = pos.iloc[rng.permutation(len(pos))[:a.n_pos]]
    if a.n_neg: neg = neg.iloc[rng.permutation(len(neg))[:a.n_neg]]
    ch = pd.DataFrame(columns=["patient_id", "group"])
    if a.include_challenge:
        c = sl[(sl.split == "challenge") & (sl.source == "CTC")].drop_duplicates("patient_id"); ch = c[["patient_id"]].assign(group="ctc_cancer")
    chosen = pd.concat([pos.assign(answer=1), ch.assign(answer=1), neg.assign(answer=0)], ignore_index=True)
    cases = [dict(patient_id=r.patient_id, group=r.group, answer=int(r.answer), is_repeat=False) for r in chosen.itertuples()]
    if a.repeats and cases:
        for j in rng.permutation(len(cases))[:a.repeats]: cases.append(dict(cases[j], is_repeat=True))
    cases = [cases[i] for i in rng.permutation(len(cases))]
    key, mids, first = [], [], {}
    for n, c in enumerate(cases, 1):
        g = sl[sl.patient_id == c["patient_id"]]; cnt = g.groupby("series_uid").size().sort_values(ascending=False, kind="stable")
        uid = sorted(cnt[cnt == cnt.max()].index)[0]; gg = g[g.series_uid == uid].sort_values("slab_idx"); z = gg.slab_idx.astype(int).tolist()
        imgs = render_case(cd._open(str(Path(a.npy_dir) / f"{uid}.npy")), z); cid = f"{n:03d}"
        html = CASE_HTML.replace("__CSS__", CSS).replace("__CASE__", cid).replace("__Q__", QUESTION).replace("__JS__", CASE_JS.replace("__IMGS__", json.dumps(imgs)).replace("__CASE__", cid))
        (site / f"case_{cid}.html").write_text(html, encoding="utf-8")
        r = sl[sl.patient_id == c["patient_id"]].iloc[0]
        key.append(dict(case_id=cid, patient_id=c["patient_id"], series_uid=uid, group=c["group"], needs_action=c["answer"], split=r.split, source=r.source,
                        n_slabs=len(z), is_repeat=c["is_repeat"], repeat_of_case="", design=a.design))
        if not c["is_repeat"]: first[c["patient_id"]] = cid
        mids.append(dict(patient_id=c["patient_id"], series_uid=uid, slab_idx=int(gg.sort_values("colon_area_px", ascending=False).slab_idx.iloc[0])))
        print(f"case {cid} ({len(z)} slabs)", flush=True)
    K = pd.DataFrame(key); K["repeat_of_case"] = [first.get(p, "") if rep else "" for p, rep in zip(K.patient_id, K.is_repeat)]
    K.to_csv(out / "panel_key.csv", index=False); pd.DataFrame(mids).drop_duplicates("patient_id").to_csv(out / "panel_cases.csv", index=False)
    intro = ("Each case is one CT scan. Scroll through all slices of the colon and answer the question on the case page. Please rate every case; "
             "there are no time limits and you may skip back. " + QUESTION)
    (site / "index.html").write_text(INDEX_HTML.replace("__CSS__", CSS).replace("__INTRO__", intro).replace("__CASES__", json.dumps(list(K.case_id))), encoding="utf-8")
    print(f"\n{len(K)} cases: {int(K.needs_action.sum())} needing action, {int((K.needs_action == 0).sum())} not, {int(K.is_repeat.sum())} repeats | design {a.design}")
    print("groups:", K.drop_duplicates(['patient_id']).group.value_counts().to_dict())
    print(f"viewer: cd {site}  then  python -m http.server 8000  and open http://localhost:8000   |   keep {out / 'panel_key.csv'} away from the raters")


if __name__ == "__main__":
    main()
