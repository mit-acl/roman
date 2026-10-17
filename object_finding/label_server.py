"""Local web page for labeling every object crop from make_crops.py with free-text labels.

Serves the crops next to a label CSV and writes each label back to that CSV as soon as
it is entered (a .bak copy of the original CSV is kept). Only listens on localhost.

Example:
    python label_server.py labels_indoor_small_objs/small_objs_labels.csv
    # then open http://localhost:8765 in a browser
"""
import argparse
import csv
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

LABEL_FIELDS = ['label', 'object_id']
OLD_FIELDS = ['adjective', 'noun', 'color', 'category']  # earlier label columns, merged into label

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Object Labeler</title>
<style>
:root { --bg:#f6f6f4; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#d9d9d4; --accent:#2563eb; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#ececea; --muted:#9a9a94; --card:#20201f; --line:#3a3a37; --accent:#60a5fa; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.4 system-ui, sans-serif; }
header { display:flex; gap:12px; align-items:center; flex-wrap:wrap; padding:10px 16px; border-bottom:1px solid var(--line); }
header .grow { flex:1; }
.bar { height:6px; background:var(--line); border-radius:3px; width:220px; overflow:hidden; }
.bar > div { height:100%; background:var(--accent); width:0; }
main { max-width:1100px; margin:0 auto; padding:16px; }
.crop { width:100%; height:30vh; object-fit:contain; background:#000; border-radius:6px; display:block; }
.ctx { width:100%; height:40vh; object-fit:contain; background:#000; border-radius:6px; display:block; margin-top:8px; }
.meta { color:var(--muted); margin:8px 0 14px; height:1.4em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:10px 0; }
.row > label { width:70px; color:var(--muted); }
button { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line); border-radius:6px; padding:6px 10px; cursor:pointer; }
input[type=text] { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line); border-radius:6px; padding:6px 10px; width:260px; }
.chips button { padding:3px 8px; font-size:13px; }
.help { color:var(--muted); font-size:13px; margin-top:18px; }
.saved { color:var(--muted); font-size:13px; min-width:80px; }
select { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line); border-radius:6px; padding:4px 6px; }
</style></head><body>
<header>
  <strong>Object Labeler</strong>
  <span id="pos"></span>
  <div class="bar"><div id="prog"></div></div>
  <span id="count"></span>
  <span class="grow"></span>
  <label><input type="checkbox" id="onlyTodo"> only unlabeled</label>
  <button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
  <input type="text" id="jump" placeholder="go to id" style="width:100px">
</header>
<main>
  <img id="img" class="crop" alt="object crops">
  <img id="ctx" class="ctx" alt="full frame with the object boxed in yellow">
  <div class="meta" id="meta"></div>
  <div class="row"><label>Label</label>
    <input type="text" id="lab" list="labs" placeholder="anything, e.g. orange tennis ball or background" autocomplete="off" style="width:320px">
    <datalist id="labs"></datalist>
    <span class="saved" id="msg"></span>
  </div>
  <div class="row"><label>Labels</label><span class="chips" id="recent"></span></div>
  <div class="row"><label>Same as</label>
    <input type="text" id="oid" placeholder="optional: object_id shared by duplicates" style="width:320px">
    <span class="saved" id="saved"></span>
  </div>
  <div class="help">
    Type any label and press <b>Enter</b> to save it and move to the next unlabeled object; <b>Esc</b> leaves the box.
    Keys (when not typing): <b>b</b> background (saves and moves on) &middot; <b>l</b> label box &middot; <b>&larr;/&rarr;</b> prev/next.
    Clicking a label button saves it and moves on; buttons keep their place, new labels are added at the end. Every change is saved to the CSV immediately.
  </div>
</main>
<script>
let rows = [], idx = 0;
const $ = id => document.getElementById(id);

function labelOf(r) { return r.label || ''; }
function labeled(r) { return labelOf(r) !== ''; }
function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;'); }
function order() { return $('onlyTodo').checked ? rows.map((r,i)=>i).filter(i=>!labeled(rows[i]) || i===idx) : rows.map((r,i)=>i); }

// labels in the order they were first used, so buttons never jump around
let labelOrder = [];
try { labelOrder = JSON.parse(localStorage.getItem('labeler-labels') || '[]'); } catch (e) {}
function usedLabels() {
  const used = new Set(rows.map(labelOf).filter(Boolean));
  labelOrder = labelOrder.filter(l => used.has(l));
  used.forEach(l => { if (!labelOrder.includes(l)) labelOrder.push(l); });
  try { localStorage.setItem('labeler-labels', JSON.stringify(labelOrder)); } catch (e) {}
  return labelOrder;
}

function render() {
  const r = rows[idx];
  $('img').src = r.crop;
  $('ctx').style.visibility = '';
  $('ctx').src = r.crop.replace(/^crops\//, 'context/');
  $('meta').textContent = `${r.run} · id ${r.segment_id} · seen in ${r.n_views} frames`;
  $('pos').textContent = `${idx+1} / ${rows.length}`;
  const done = rows.filter(labeled).length;
  $('count').textContent = `${done} labeled`;
  $('prog').style.width = (100*done/rows.length) + '%';
  $('lab').value = labelOf(r);
  $('oid').value = r.object_id || '';
  const labs = usedLabels();
  $('labs').innerHTML = labs.map(l => `<option value="${esc(l)}">`).join('');
  $('recent').innerHTML = labs.map(l => `<button data-lab="${esc(l)}">${esc(l)}</button>`).join('');
  $('msg').textContent = '';
  $('saved').textContent = '';
  try { localStorage.setItem('labeler-idx', String(idx)); } catch (e) {}
}

async function save(fields) {
  const r = rows[idx];
  Object.assign(r, fields);
  $('saved').textContent = 'saving…';
  const res = await fetch('/api/save', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({run:r.run, segment_id:r.segment_id, ...fields})});
  render();
  $('saved').textContent = res.ok ? 'saved' : 'SAVE FAILED';
  return res.ok;
}

async function submitLabel(advance) {
  await save({label: $('lab').value.trim().replace(/\s+/g, ' ')});
  if (advance && labeled(rows[idx])) { $('lab').blur(); nextTodo(); }
}

function move(step) {
  const ord = order();
  let p = ord.indexOf(idx);
  if (p < 0) p = 0;
  p = Math.min(Math.max(p + step, 0), ord.length - 1);
  idx = ord[p];
  render();
}

function nextTodo() {
  for (let k = 1; k <= rows.length; k++) {
    const i = (idx + k) % rows.length;
    if (!labeled(rows[i])) { idx = i; render(); return; }
  }
  move(1);
}

$('recent').onclick = e => {
  const l = e.target.dataset.lab;
  if (l) { $('lab').value = l; submitLabel(true); }
};
$('lab').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); submitLabel(true); }
  if (e.key === 'Escape') $('lab').blur();
});
$('oid').addEventListener('change', () => save({object_id: $('oid').value.trim()}));
$('ctx').onerror = () => { $('ctx').style.visibility = 'hidden'; };  // keep its space so nothing moves
$('prev').onclick = () => move(-1);
$('next').onclick = () => move(1);
$('onlyTodo').onchange = render;
$('jump').addEventListener('keydown', e => {
  if (e.key !== 'Enter') return;
  const i = rows.findIndex(r => r.segment_id === $('jump').value.trim());
  if (i >= 0) { idx = i; render(); $('jump').blur(); }
});
document.addEventListener('keydown', e => {
  if (['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'b') { save({label: 'background'}).then(nextTodo); e.preventDefault(); }
  else if (e.key === 'l') { $('lab').focus(); e.preventDefault(); }
  else if (e.key === 'ArrowRight') move(1);
  else if (e.key === 'ArrowLeft') move(-1);
});

fetch('/api/rows').then(r => r.json()).then(data => {
  rows = data;
  let saved = 0;
  try { saved = parseInt(localStorage.getItem('labeler-idx') || '0', 10) || 0; } catch (e) {}
  idx = Math.min(saved, rows.length - 1);
  render();
});
</script></body></html>
"""


class Labels:
    def __init__(self, csv_path):
        self.path = csv_path
        self.lock = threading.Lock()
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            self.fields = list(reader.fieldnames)
            self.rows = list(reader)
        for name in LABEL_FIELDS:
            if name not in self.fields:
                self.fields.append(name)
                for r in self.rows:
                    r[name] = ''
        # merge labels from older adjective/noun (or color/category) columns into label
        old = [c for c in OLD_FIELDS if c in self.fields]
        if old and not os.path.exists(csv_path + '.premerge.bak'):
            shutil.copy2(csv_path, csv_path + '.premerge.bak')
        for r in self.rows:
            if not r['label']:
                r['label'] = ' '.join(r[c] for c in old if r[c])
            for c in old:
                del r[c]
        self.fields = [c for c in self.fields if c not in old]
        backup = csv_path + '.bak'
        if not os.path.exists(backup):
            shutil.copy2(csv_path, backup)

    def update(self, run, segment_id, fields):
        with self.lock:
            for r in self.rows:
                if r['run'] == run and r['segment_id'] == segment_id:
                    for k, v in fields.items():
                        if k in LABEL_FIELDS:
                            r[k] = str(v)
                    break
            else:
                return False
            tmp = self.path + '.tmp'
            with open(tmp, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written CSV
            return True


def make_handler(labels, root):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == '/':
                page = PAGE
                return self._send(200, page.encode(), 'text/html; charset=utf-8')
            if path == '/api/rows':
                return self._send(200, json.dumps(labels.rows).encode(), 'application/json')
            if path.startswith('/crops/') or path.startswith('/context/'):
                file_path = os.path.realpath(os.path.join(root, path.lstrip('/')))
                if file_path.startswith(os.path.realpath(root) + os.sep) and os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        return self._send(200, f.read(), 'image/jpeg')
            self._send(404, b'not found', 'text/plain')

        def do_POST(self):
            if urlparse(self.path).path != '/api/save':
                return self._send(404, b'not found', 'text/plain')
            data = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            fields = {k: data[k] for k in LABEL_FIELDS if k in data}
            ok = labels.update(str(data.get('run')), str(data.get('segment_id')), fields)
            self._send(200 if ok else 404, b'{}', 'application/json')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', help='Label CSV written by make_crops.py (crops/ must sit next to it)')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    csv_path = os.path.abspath(os.path.expanduser(args.csv))
    labels = Labels(csv_path)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(labels, os.path.dirname(csv_path)))
    print(f'Labeling {len(labels.rows)} objects from {csv_path}')
    print(f'Open http://localhost:{args.port} in a browser (Ctrl+C to stop; labels are saved as you go)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
