# Object finding

Tools for finding objects in a ROMAN map from a text prompt ("find the orange ball"), and for
checking how well ROMAN's DINOv2 segment descriptors support that, on the
`10-09-2026_indoor_roman_objects` recording (small_objs map).

- **Text:** [Talk2DINO](https://arxiv.org/abs/2411.19331) projects CLIP text embeddings into the
  patch space of DINOv2 ViT-B/14 with registers, so a prompt can be compared directly to ROMAN's
  descriptors (means of patch tokens inside each mask). This needs a map built with
  `facebook/dinov2-with-registers-base`; the change to `roman/map/fastsam_wrapper.py` and
  `roman/tensorrt/dinov2.py` on this branch drops the register tokens along with CLS.
- **Color:** DINOv2 is trained with hue jitter of up to +-36 degrees, so it barely tells red,
  orange and yellow apart. `segment_color.py` keeps an 11-bin color-name histogram per segment
  (van de Weijer et al., TIP 2009, table in `data/w2c.txt.gz`), and a prompt's color words
  are scored as `talk2dino_cosine - w * (1 - fraction of the segment's pixels with that color)`,
  with w = 0.2.

On 24 prompts for the labeled objects (`neighbors_page.py`), the target object ranks:

| | Talk2DINO + color | Talk2DINO only |
|---|---|---|
| 1st | 12 / 24 | 11 / 24 |
| top 5 | 20 / 24 | 20 / 24 |
| median rank | 1.5 | 2 |

## Files

| Script | What it does |
|---|---|
| `run_roman_on_times.py` | Rebuilds a map from the bag on exactly the frames a reference map used; `--color-names` also saves `<out>.colors.npz` |
| `segment_color.py` | Color-name histograms per segment, via hooks on ROMAN (the map itself is unchanged), and color matching for prompts |
| `talk2dino.py` | Talk2DINO text encoder |
| `make_crops.py` | Image crops of every segment and a label CSV template |
| `label_server.py` | Local web page for labeling the crops |
| `compare_maps.py` | Matches two maps of the same run by 3D overlap; `--out-labels` carries labels over to the rebuilt map |
| `neighbors_page.py` | HTML page: text prompts with their top 5 objects, and each labeled object with its most similar objects |

Data:
- `roman_params/indoor_small_objs/`: ROMAN params for the recording.
- `labels_indoor_small_objs/small_objs_labels.csv`: labels for the original small_objs map.
- `rerun/small_objs_dinov2_reg_labels.csv`: the same labels carried over to the map rebuilt with
  dinov2-with-registers.

Maps, crops and pages are generated and not tracked.

## Usage

Run from this directory, with ROMAN installed and `ROMAN_WEIGHTS` set as in the main README.
`REF=/mnt/acl-aerovault/projects/mad_summer_2026_mason/10-09-2026_indoor_roman_objects/small_objs.roman_map.pkl`

```
# map with DINOv2 with registers and color histograms (~3 min on GPU)
python run_roman_on_times.py --params roman_params/indoor_small_objs --color-names \
    --dino-model facebook/dinov2-with-registers-base \
    --reference-map $REF --out rerun/small_objs_dinov2_reg_color.pkl

# crops for the labeled map, and for the rebuilt map (shown on the text-prompt tab)
python make_crops.py --bag $(dirname $REF)/bag --topic /gauss/gauss_zed/rgb/image_rect_color \
    --run small_objs --map $REF --out labels_indoor_small_objs
python make_crops.py --params roman_params/indoor_small_objs --run indoor \
    --map rerun/small_objs_dinov2_reg_color.pkl --out labels_indoor_small_objs/registers_map

# labels: label_server.py for new labels, then carry them over to the rebuilt map
python label_server.py labels_indoor_small_objs/small_objs_labels.csv
python compare_maps.py --ref $REF --new rerun/small_objs_dinov2_reg_color.pkl \
    --labels labels_indoor_small_objs/small_objs_labels.csv --out-labels rerun/small_objs_dinov2_reg_labels.csv

# page, written to labels_indoor_small_objs/neighbors.html
python neighbors_page.py --map $REF --labels labels_indoor_small_objs/small_objs_labels.csv \
    --text-map rerun/small_objs_dinov2_reg_color.pkl \
    --text-colors rerun/small_objs_dinov2_reg_color.colors.npz \
    --text-labels rerun/small_objs_dinov2_reg_labels.csv
```
