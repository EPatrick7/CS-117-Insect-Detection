# INSECT12C

These 1,800 soybean insect photos and their annotations come from [Everton Tetila and colleagues’ INSECT12C dataset](https://github.com/EvertonTetila/INSECT12C-Dataset/tree/ffa37d73b7a0bece6385236b336ab4c36497bc27). The authors describe the dataset in [*YOLO performance analysis for real-time detection of soybean pests* (2024)](https://doi.org/10.1016/j.atech.2024.100405).

Each photo sits beside its YOLO text label and original XML annotation. The upstream archive supplied only XML for `img1146`, so its YOLO label was converted from that annotation.

## Physical split

| Folder | Photos | Labeled boxes | Capture groups with active photos |
| --- | ---: | ---: | ---: |
| `train/` | 1,497 | 2,310 | 59 |
| `val/` | 150 | 219 | 11 |
| `test/` | 140 | 208 | 14 |
| `reserve/` | 13 | 21 | — |

The [notebook](../../02_stock_insect_baseline.ipynb) and [Ultralytics config](../../insect12c.yaml) read the three active folders directly. The config is for single-class detection (`single_cls=True`). Each text annotation retains its original species ID from [classes.txt](classes.txt). To use Joshua's training script with these folders, pass `--data insect12c.yaml --single-cls` from the repository root.

## How photos were grouped

The numbered archives contain repeated shots of insects. For example, `img23` and `img1058` show the same beetle on a leaf one second apart. The earlier archive split put one in test and one in training. Both now live in `val/`.

Within each capture date and camera model, consecutive shots at most 60 seconds apart form a burst. A search using whole-photo features and annotated insect crops identified additional links across bursts. Those [visual links](visual_group_links.csv) join bursts before assigning folders. Grouping is transitive. A burst can contain different insects, and the dataset has no persistent IDs for individual insects.

The ten original smoke-test photos (`img1`–`img10`) and three photos without an original capture timestamp (`img273`, `img770`, and `img781`) are in `reserve/`. The active relatives of smoke-test photos were assigned to training. The notebook uses the existing preview copies only in its smoke-test cell.

Whole groups were assigned using labeled-box counts to balance species coverage, with a fixed seed of 117. Training and test each include all 12 original categories. Validation includes 11. Only one group eligible for a holdout contained the *Nezara viridula* nymph category, and it was assigned to test.

Capture dates can appear in multiple active folders. This split evaluates other capture bursts in the same collection. Use a checkpoint trained from stock on these folders for that evaluation. The preceding archive split is preserved in Git at `c6d9d06`.
