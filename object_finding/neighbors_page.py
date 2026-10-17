"""Write an HTML page for browsing retrieval of the labeled ROMAN objects, with two tabs.

Text prompts (with --text-map): a set of prompts such as "find the orange ball", each
followed by the --top ROMAN objects that score highest for it. The object phrase is taken
from the prompt ("orange ball") and scored with Talk2DINO against a map built with
dinov2-with-registers-base, alone or with the color-name term from segment_color.py
(score - w * (1 - fraction of the object's pixels with the named color)); a switch on the
page flips between the two. Results that are the target object (any segment of it) have a
green border, and each prompt shows where the target first ranks.

Similar objects: every labeled (non-background) ROMAN object in --map, followed by the --top
ROMAN objects with the most similar DINOv2 descriptors, most similar first. The list
includes the object itself, which always comes first. Entries that are the same physical
object (the object itself, or another segment with the same object_id) have a green
border. Similarity is cosine between mean-centered descriptors, computed over the labeled
segments (background included).

The page is written next to the label CSV and loads crops (from make_crops.py) by relative
path, so it opens from disk in a browser.

Example:
    python neighbors_page.py --map /path/to/small_objs.roman_map.pkl \
        --labels labels_indoor_small_objs/small_objs_labels.csv \
        --text-map rerun/small_objs_dinov2_reg_color.pkl \
        --text-colors rerun/small_objs_dinov2_reg_color.colors.npz \
        --text-labels rerun/small_objs_dinov2_reg_labels.csv
"""
import argparse
import csv
import html
import os
import re
from collections import defaultdict

import numpy as np

from roman.map.map import load_roman_map

BALLS = ['orange ball', 'yellow ball', 'blue soccer ball', 'tennis balls', 'rainbow ball with magenta stripe',
         'rainbow ball with large yellow stripe']
# (prompt, labels of the target object(s)); a result counts if its object carries one of them
PROMPTS = [
    ('find the orange ball', ['orange ball']),
    ('where is the yellow ball?', ['yellow ball']),
    ('find the blue soccer ball', ['blue soccer ball']),
    ('find the soccer ball', ['blue soccer ball']),
    ('find the tennis balls', ['tennis balls']),
    ('find the rainbow ball with the magenta stripe', ['rainbow ball with magenta stripe']),
    ('find the rainbow ball with the yellow stripe', ['rainbow ball with large yellow stripe']),
    ('find a ball', BALLS),
    ('find the black battery', ['black battery']),
    ('find the battery', ['black battery']),
    ('find the blue lunch box', ['blue lunch box']),
    ('find the water bottle', ['maroon and black water bottle']),
    ('find the maroon water bottle', ['maroon and black water bottle']),
    ('find the bottle cap', ['black water bottle cap']),
    ('find the red and white target bag', ['red and white target bag']),
    ('find the shopping bag', ['red and white target bag']),
    ('find the refrigerator', ['refridgerator']),
    ('find the fridge', ['refridgerator']),
    ('find the white mail bin', ['white mail bin', 'white mail bin with brown box on top of it']),
    ('find the cardboard box', ['white mail bin with brown box on top of it']),
    ('find the red airplane', ['red airplane']),
    ('find the toy airplane', ['red airplane', 'red airplane wing', 'yellow airplane wing']),
    ('find the red airplane wing', ['red airplane wing']),
    ('find the yellow airplane wing', ['yellow airplane wing']),
]

STYLE = """
:root { --bg:#f4f4f1; --fg:#1c1c1a; --muted:#6d6d67; --card:#ffffff; --line:#e2e2dc; --thumb:#0e0e0d;
        --accent:#2f5bd3; --accent-soft:#e8eefc; --same:#15803d; --same-soft:#e6f4ea; --bar:#c9c9c2;
        --ok:#a16207; --ok-soft:#fdf3d7; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg:#131312; --fg:#ededea; --muted:#9b9b94; --card:#1d1d1b; --line:#33332f; --thumb:#000;
        --accent:#8aa8ff; --accent-soft:#1f2a47; --same:#4ade80; --same-soft:#163322; --bar:#4a4a45;
        --ok:#facc15; --ok-soft:#3a3214; } }
:root[data-theme="dark"] { --bg:#131312; --fg:#ededea; --muted:#9b9b94; --card:#1d1d1b; --line:#33332f;
        --thumb:#000; --accent:#8aa8ff; --accent-soft:#1f2a47; --same:#4ade80; --same-soft:#163322; --bar:#4a4a45;
        --ok:#facc15; --ok-soft:#3a3214; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1560px; margin:0 auto; padding:24px 16px 48px; }
header h1 { font-size:22px; letter-spacing:-0.01em; margin:0 0 6px; }
.intro p { margin:0 0 6px; color:var(--muted); max-width:960px; }
.tabs { display:flex; gap:4px; margin:16px 0 0; border-bottom:1px solid var(--line); }
.tabs button { font:inherit; font-weight:600; color:var(--muted); background:none; border:0; padding:10px 14px;
               border-bottom:3px solid transparent; margin-bottom:-1px; cursor:pointer; }
.tabs button[aria-selected="true"] { color:var(--fg); border-bottom-color:var(--accent); }
.panel { padding-top:12px; }
.toolbar { position:sticky; top:0; z-index:2; display:flex; flex-wrap:wrap; gap:12px 20px; align-items:center;
           background:var(--bg); padding:12px 0; margin:8px 0 16px; border-bottom:1px solid var(--line); }
.toolbar input { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line);
                 border-radius:8px; padding:7px 10px; width:240px; max-width:100%; }
.switch { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:var(--card); }
.switch button { font:inherit; font-size:13px; color:var(--muted); background:none; border:0; padding:7px 12px; cursor:pointer; }
.switch button[aria-pressed="true"] { background:var(--accent); color:#fff; }
.legend { display:flex; flex-wrap:wrap; gap:16px; color:var(--muted); font-size:13px; }
.legend span::before { content:""; display:inline-block; width:14px; height:14px; border-radius:4px;
                       margin-right:6px; vertical-align:-2px; border:3px solid var(--same); }
.legend .q::before { border-color:var(--accent); background:var(--accent-soft); }
.count { color:var(--muted); font-size:13px; margin-left:auto; }
.stats { display:flex; flex-wrap:wrap; gap:10px; margin:4px 0 4px; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 14px; min-width:150px; }
.stat b { display:block; font-size:22px; font-variant-numeric:tabular-nums; }
.stat span { color:var(--muted); font-size:12px; }
.row { display:grid; grid-template-columns:minmax(240px, 300px) 1fr; gap:16px; background:var(--card);
       border:1px solid var(--line); border-radius:12px; padding:14px; margin-bottom:14px; }
.query { background:var(--accent-soft); border:3px solid var(--accent); border-radius:10px; padding:10px; min-width:0; }
.eyebrow { font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--accent); margin-bottom:8px; }
.matches-head { font-size:11px; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); margin:2px 0 8px; }
.grid { list-style:none; margin:0; padding:0; display:grid; grid-template-columns:repeat(var(--n), minmax(0, 1fr)); gap:10px; }
.match { position:relative; border:3px solid transparent; border-radius:10px; padding:6px; min-width:0; }
.match.same { border-color:var(--same); background:var(--same-soft); }
.thumb { display:block; aspect-ratio:2 / 1; background:var(--thumb); border-radius:6px; overflow:hidden; }
.thumb img { width:100%; height:100%; object-fit:contain; display:block; }
.thumb.none { display:flex; align-items:center; justify-content:center; text-align:center; padding:6px;
              color:#9b9b94; font-size:11px; }
.query .thumb { aspect-ratio:1.8 / 1; }
.rank { position:absolute; top:12px; right:12px; min-width:22px; height:22px; padding:0 6px; border-radius:11px;
        background:rgba(0,0,0,0.72); color:#fff; font-size:12px; font-weight:700; line-height:22px; text-align:center; }
.name { font-weight:650; margin-top:7px; overflow-wrap:anywhere; }
.query .name { font-size:16px; }
.name.bg { font-weight:500; color:var(--muted); font-style:italic; }
.meta { color:var(--muted); font-size:12px; margin-top:2px; }
.meta b { color:var(--fg); font-weight:600; }
.sim { display:flex; align-items:center; gap:8px; margin-top:5px; font-size:12px; color:var(--muted);
       font-variant-numeric:tabular-nums; }
.sim .bar { flex:1; height:5px; border-radius:3px; background:var(--line); overflow:hidden; }
.sim .bar span { display:block; height:100%; background:var(--bar); }
.same .sim .bar span { background:var(--same); }
.chip { display:inline-block; margin-top:6px; font-size:11px; font-weight:700; color:var(--same); }
.others { margin-top:8px; padding-top:8px; border-top:1px solid var(--line); font-size:12px; color:var(--muted); }
.others b { color:var(--fg); }
.prompt-text { font-size:18px; font-weight:650; line-height:1.3; margin-bottom:6px; overflow-wrap:anywhere; }
.target { display:grid; grid-template-columns:96px 1fr; gap:8px; align-items:center; margin-top:10px; }
.target .thumb { aspect-ratio:2 / 1; }
.verdict { margin-top:10px; padding-top:8px; border-top:1px solid var(--line); font-size:13px; }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-weight:700; font-size:12px;
        background:var(--line); color:var(--fg); }
.pill.good { background:var(--same-soft); color:var(--same); }
.pill.ok { background:var(--ok-soft); color:var(--ok); }
.empty { color:var(--muted); padding:24px 0; display:none; }
.legend-wrap { display:contents; }
.legend-wrap[hidden], .panel[hidden] { display:none !important; }
body.mode-t2d .m-color, body:not(.mode-t2d) .m-t2d { display:none !important; }
@media (max-width:1100px) { .row { grid-template-columns:1fr; } .query { max-width:420px; } }
@media (max-width:700px) { .grid { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
"""

SCRIPT = """
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
const input = document.getElementById('filter');
const count = document.getElementById('count');
const empty = document.getElementById('empty');
const tabs = [...document.querySelectorAll('.tabs button')];
let active = null;
function applyFilter() {
  const rows = [...active.querySelectorAll('.row')];
  const q = input.value.trim().toLowerCase();
  let shown = 0;
  for (const r of rows) {
    const hit = !q || r.dataset.search.includes(q);
    r.style.display = hit ? '' : 'none';
    shown += hit;
  }
  const what = active.dataset.noun;
  count.textContent = q ? `${shown} of ${rows.length} ${what}` : `${rows.length} ${what}`;
  empty.style.display = shown ? 'none' : 'block';
}
function showTab(id) {
  for (const b of tabs) {
    const on = b.dataset.tab === id;
    b.setAttribute('aria-selected', on);
    document.getElementById(b.dataset.tab).hidden = !on;
    if (on) active = document.getElementById(b.dataset.tab);
  }
  document.querySelectorAll('[data-for]').forEach(el => { el.hidden = el.dataset.for !== id; });
  store.set('roman-retrieval-tab', id);
  applyFilter();
}
function setMode(mode) {
  document.body.classList.toggle('mode-t2d', mode === 't2d');
  document.querySelectorAll('.switch button').forEach(b => b.setAttribute('aria-pressed', b.dataset.mode === mode));
  store.set('roman-retrieval-mode', mode);
}
tabs.forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
document.querySelectorAll('.switch button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
input.addEventListener('input', applyFilter);
setMode(store.get('roman-retrieval-mode') === 't2d' ? 't2d' : 'color');
const saved = store.get('roman-retrieval-tab');
showTab(tabs.some(b => b.dataset.tab === saved) ? saved : tabs[0].dataset.tab);
"""


def ordinal(n):
    return f'{n}{"th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")}'


PAGE_DIR = '.'  # folder the page is written to; crop paths are relative to it


def thumb(img):
    if not os.path.exists(os.path.join(PAGE_DIR, img)):
        return '<div class="thumb none">no crop: too small in every view</div>'
    return (f'<a class="thumb" href="{html.escape(img)}" target="_blank" title="Open full size">'
            f'<img src="{html.escape(img)}" alt=""></a>')


def name(label):
    muted = label in ('background', 'unlabeled')
    return f'<div class="name{" bg" if muted else ""}">{html.escape(label)}</div>'


def bar(value, full=1.0):
    return (f'<div class="sim"><span class="bar"><span style="width:{100 * min(max(value / full, 0), 1):.0f}%">'
            f'</span></span>{value:.2f}</div>')


def rank_pill(rank):
    kind = 'good' if rank == 1 else ('ok' if rank <= 5 else '')
    return f'<span class="pill {kind}">{ordinal(rank)}</span>'


def search_phrase(prompt):
    """The object phrase in an instruction: 'find the orange ball' -> 'orange ball'."""
    p = prompt.strip().rstrip('?.!').lower()
    p = re.sub(r"^(please\s+)?(can you\s+)?(find|locate|look for|search for|where is|where are|where's|"
               r"show me|go to)\s+", '', p)
    return re.sub(r"^(me\s+)?(the|a|an|some|any)\s+", '', p)


def similar_objects_section(args):
    """Rows of (labeled ROMAN object, its most similar ROMAN objects) for the --map."""
    descriptors = {str(seg.id): seg.semantic_descriptor.ravel() for seg in
                   load_roman_map(os.path.expanduser(args.map)).segments if seg.semantic_descriptor is not None}
    with open(args.labels, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['segment_id'] in descriptors]
    ids = [r['segment_id'] for r in rows]
    label = {r['segment_id']: r['label'] for r in rows}
    obj = {r['segment_id']: r['object_id'] or 'seg' + r['segment_id'] for r in rows}
    crop = lambda i: f'{args.crops}/{rows[0]["run"]}_{i}.jpg'

    Z = np.stack([descriptors[i] for i in ids]).astype(np.float64)
    Z -= Z.mean(axis=0)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    S = Z @ Z.T

    queries = sorted((qi for qi, q in enumerate(ids) if label[q] != 'background'),
                     key=lambda qi: (label[ids[qi]].lower(), int(ids[qi])))
    n_physical = len({obj[ids[qi]] for qi in queries})
    blocks = []
    for qi in queries:
        q = ids[qi]
        order = list(np.argsort(-S[qi]))
        order.remove(qi)
        order.insert(0, qi)  # the object itself first, even if a duplicate descriptor ties it
        rank = {ids[j]: r + 1 for r, j in enumerate(order)}
        same = [i for i in ids if obj[i] == obj[q] and i != q]

        meta = f'segment {q}'
        if same:
            meta += f' · {html.escape(obj[q])}, {len(same) + 1} segments'
        others = ''
        if same:
            parts = ', '.join(f'<b>{i}</b> ({html.escape(label[i])}) ranked {ordinal(rank[i])}'
                              for i in sorted(same, key=rank.get))
            others = (f'<div class="others">Other segment{"s" * (len(same) > 1)} of this object: '
                      f'{parts} of {len(ids)}</div>')
        query = (f'<div class="query"><div class="eyebrow">ROMAN object</div>{thumb(crop(q))}{name(label[q])}'
                 f'<div class="meta">{meta}</div>{others}</div>')

        cards = []
        for r, j in enumerate(order[:args.top]):
            i = ids[j]
            is_same = obj[i] == obj[q]
            chip = ''
            if i == q:
                chip = '<span class="chip">this object</span>'
            elif is_same:
                chip = '<span class="chip">same object</span>'
            cards.append(f'<li class="match{" same" if is_same else ""}"><span class="rank">{r + 1}</span>'
                         f'{thumb(crop(i))}{name(label[i])}{bar(S[qi, j])}<div class="meta">segment {i}</div>'
                         f'{chip}</li>')

        search = html.escape(' '.join([label[q], q, obj[q] if same else '']).lower())
        blocks.append(f'<section class="row" data-search="{search}">{query}<div><div class="matches-head">'
                      f'Top {args.top} most similar ROMAN objects</div><ol class="grid" style="--n:{args.top}">'
                      f'{"".join(cards)}</ol></div></section>')

    intro = (f'<div class="intro"><p>Each row starts with one labeled <b>ROMAN object</b> (a segment of '
             f'<code>{html.escape(os.path.basename(args.map))}</code>, DINOv2 base), followed by the {args.top} '
             'ROMAN objects with the most similar DINOv2 descriptors, most similar first. The object itself is '
             'included and always ranks 1st.</p>'
             f'<p>{len(queries)} labeled objects ({n_physical} physical objects), compared against all {len(ids)} '
             'labeled segments including background. Similarity: cosine of mean-centered descriptors.</p></div>')
    return intro, blocks


def text_prompt_section(args):
    """Rows of (text prompt, the highest-scoring ROMAN objects) for the --text-map."""
    import torch

    from segment_color import color_fraction, query_colors
    from talk2dino import Talk2DINOText

    segs = [s for s in load_roman_map(os.path.expanduser(args.text_map)).segments
            if s.semantic_descriptor is not None]
    ids = [str(s.id) for s in segs]
    X = np.stack([s.semantic_descriptor.ravel() for s in segs]).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    col = np.load(args.text_colors)
    color_of = dict(zip(map(str, col['segment_id']), col['color_names']))
    colors = np.stack([color_of[i] for i in ids])
    with open(args.text_labels, newline='') as f:
        rows = {r['segment_id']: r for r in csv.DictReader(f)}
    label = {i: rows[i]['label'] if i in rows else 'unlabeled' for i in ids}
    obj = {i: (rows[i]['object_id'] if i in rows and rows[i]['object_id'] else 'seg' + i) for i in ids}
    labels_of_obj = defaultdict(set)
    for i in ids:
        labels_of_obj[obj[i]].add(label[i])
    crop = lambda i: f'{args.text_crops}/{args.text_run}_{i}.jpg'

    phrases = [search_phrase(p) for p, _ in PROMPTS]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    S = Talk2DINOText(device).encode(phrases) @ X.T
    C = np.stack([color_fraction(colors, p) for p in phrases])
    methods = {'color': S - args.color_weight * (1 - C), 't2d': S}
    full = max(m.max() for m in methods.values())

    first = {m: [] for m in methods}
    blocks = []
    for k, ((prompt, targets), phrase) in enumerate(zip(PROMPTS, phrases)):
        missing = [t for t in targets if t not in label.values()]
        assert not missing, f'{prompt!r}: no segment labeled {missing}'
        hit = np.array([label[i] in targets or bool(labels_of_obj[obj[i]] & set(targets)) for i in ids])
        words = query_colors(phrase)
        tops, ranks = {}, {}
        for m, score in methods.items():
            order = np.argsort(-score[k])
            ranks[m] = int(np.nonzero(hit[order])[0][0]) + 1
            first[m].append(ranks[m])
            cards = []
            for r, j in enumerate(order[:args.top]):
                i = ids[j]
                detail = f'segment {i}'
                if m == 'color' and words:
                    detail += f' · Talk2DINO {S[k, j]:.2f} · {"+".join(words)} {100 * C[k, j]:.0f}%'
                cards.append(f'<li class="match{" same" if hit[j] else ""}"><span class="rank">{r + 1}</span>'
                             f'{thumb(crop(i))}{name(label[i])}{bar(score[k, j], full)}'
                             f'<div class="meta">{detail}</div>'
                             f'{"<span class=chip>target object</span>" if hit[j] else ""}</li>')
            tops[m] = f'<ol class="grid m-{m}" style="--n:{args.top}">{"".join(cards)}</ol>'

        direct = sorted((i for i in ids if label[i] in targets), key=lambda i: (targets.index(label[i]), int(i)))
        n_obj, n_seg = len({obj[i] for i, h in zip(ids, hit) if h}), int(hit.sum())
        target_text = html.escape(', '.join(targets))
        color_text = f' · color: <b>{html.escape(" + ".join(words))}</b>' if words else ' · no color word'
        query = (f'<div class="query"><div class="eyebrow">Text prompt</div>'
                 f'<div class="prompt-text">&ldquo;{html.escape(prompt)}&rdquo;</div>'
                 f'<div class="meta">searched for: <b>{html.escape(phrase)}</b>{color_text}</div>'
                 f'<div class="target">{thumb(crop(direct[0]))}<div class="meta">target: <b>{target_text}</b>'
                 f'<br>{n_obj} object{"s" * (n_obj > 1)}, {n_seg} segment{"s" * (n_seg > 1)}</div></div>'
                 f'<div class="verdict"><span class="m-color">target first ranked {rank_pill(ranks["color"])} of '
                 f'{len(ids)} <span class="meta">(Talk2DINO only: {ordinal(ranks["t2d"])})</span></span>'
                 f'<span class="m-t2d">target first ranked {rank_pill(ranks["t2d"])} of {len(ids)} '
                 f'<span class="meta">(with color: {ordinal(ranks["color"])})</span></span></div></div>')
        search = html.escape(' '.join([prompt, phrase] + targets).lower())
        blocks.append(f'<section class="row" data-search="{search}">{query}<div><div class="matches-head">'
                      f'Top {args.top} ROMAN objects for this prompt</div>{tops["color"]}{tops["t2d"]}</div></section>')

    def stats(m, other, other_name):
        r, o = np.array(first[m]), np.array(first[other])
        vs = lambda x: f'<br>{other_name}: {x}'
        return (f'<div class="stat m-{m}"><b>{np.sum(r == 1)} / {len(r)}</b><span>target ranked 1st'
                f'{vs(np.sum(o == 1))}</span></div>'
                f'<div class="stat m-{m}"><b>{np.sum(r <= args.top)} / {len(r)}</b><span>target in top {args.top}'
                f'{vs(np.sum(o <= args.top))}</span></div>'
                f'<div class="stat m-{m}"><b>{np.median(r):g}</b><span>median rank of target'
                f'{vs(f"{np.median(o):g}")}</span></div>')

    intro = (f'<div class="intro"><p>Each row starts with a <b>text prompt</b>, followed by the {args.top} ROMAN '
             f'objects that score highest for it. Green marks the target object (any of its segments). The object '
             'phrase is taken from the prompt (&ldquo;find the orange ball&rdquo; &rarr; &ldquo;orange ball&rdquo;) '
             'and scored with Talk2DINO: its CLIP text embedding is projected into the DINOv2-with-registers space '
             'and compared by cosine with each object&rsquo;s descriptor. <b>With color</b>, the score drops by '
             f'up to {args.color_weight:g} when the object&rsquo;s pixels lack the color the prompt names '
             f'(score &minus; {args.color_weight:g} &times; (1 &minus; color fraction)).</p>'
             f'<p>Map: <code>{html.escape(os.path.basename(args.text_map))}</code>, {len(ids)} objects. Labels '
             'were carried over from the labeled map by 3D overlap; &ldquo;unlabeled&rdquo; objects had no '
             'match.</p></div><div class="stats">' + stats('color', 't2d', 'Talk2DINO only')
             + stats('t2d', 'color', 'with color') + '</div>')
    return intro, blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', required=True, help='Labeled ROMAN map .pkl (similar-objects tab)')
    parser.add_argument('--labels', required=True, help='Label CSV from label_server.py for --map')
    parser.add_argument('--crops', default='crops', help='Crop folder for --map, relative to the page')
    parser.add_argument('--text-map', default=None,
                        help='ROMAN map built with dinov2-with-registers-base and --color-names (text tab)')
    parser.add_argument('--text-colors', default=None, help='<text-map>.colors.npz')
    parser.add_argument('--text-labels', default=None, help='Label CSV for --text-map (from compare_maps.py)')
    parser.add_argument('--text-crops', default='registers_map/crops',
                        help='Crop folder for --text-map, relative to the page')
    parser.add_argument('--text-run', default='indoor', help='Run name in the --text-map crop file names')
    parser.add_argument('--color-weight', type=float, default=0.2, help='w in score - w * (1 - color fraction)')
    parser.add_argument('--top', type=int, default=5, help='Entries per row (similar objects: including itself)')
    parser.add_argument('--out', default=None, help='Output HTML (default: neighbors.html next to the CSV)')
    args = parser.parse_args()
    if args.text_map and not (args.text_colors and args.text_labels):
        parser.error('--text-map needs --text-colors and --text-labels')
    global PAGE_DIR
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.labels)), 'neighbors.html')
    PAGE_DIR = os.path.dirname(os.path.abspath(out))

    sections = []  # (id, tab title, noun for the row count, intro, rows, legend)
    if args.text_map:
        intro, blocks = text_prompt_section(args)
        legend = ('<div class="switch" role="group" aria-label="Scoring"><button data-mode="color">'
                  'Talk2DINO + color</button><button data-mode="t2d">Talk2DINO only</button></div>'
                  '<div class="legend"><span class="q">text prompt</span><span>target object</span></div>')
        sections.append(('tab-text', 'Text prompts', 'prompts', intro, blocks, legend))
    intro, blocks = similar_objects_section(args)
    legend = '<div class="legend"><span class="q">ROMAN object</span><span>same physical object</span></div>'
    sections.append(('tab-sim', 'Similar objects', 'objects', intro, blocks, legend))

    tab_buttons = ''.join(f'<button data-tab="{sid}" role="tab">{title} ({len(blocks)})</button>'
                          for sid, title, _, _, blocks, _ in sections)
    legends = ''.join(f'<div class="legend-wrap" data-for="{sid}">{legend}</div>'
                      for sid, _, _, _, _, legend in sections)
    panels = ''.join(f'<div class="panel" id="{sid}" data-noun="{noun}" role="tabpanel">{intro}'
                     f'<div class="rows">{"".join(blocks)}</div></div>'
                     for sid, _, noun, intro, blocks, _ in sections)
    with open(out, 'w') as f:
        f.write(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>ROMAN Object Retrieval</title><style>{STYLE}</style></head><body><main>'
            '<header><h1>ROMAN object retrieval</h1></header>'
            f'<nav class="tabs" role="tablist">{tab_buttons}</nav>'
            '<div class="toolbar"><input id="filter" type="search" placeholder="Filter rows" aria-label="Filter">'
            f'{legends}<div class="count" id="count"></div></div>'
            f'{panels}<p class="empty" id="empty">No rows match the filter.</p>'
            '<p class="intro"><span class="meta">Click a crop to open it full size.</span></p>'
            f'</main><script>{SCRIPT}</script></body></html>')
    print(f'Wrote {out} ({", ".join(f"{len(b)} {n}" for _, _, n, _, b, _ in sections)})')


if __name__ == '__main__':
    main()
