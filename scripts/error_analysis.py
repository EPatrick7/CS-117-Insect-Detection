"""Compare predictions (from predict.py) to the labeled boxes.

TP  prediction matches a labeled insect (IoU >= 0.5)
FP  prediction with no matching insect
FN  labeled insect nobody matched

FN "no guess nearby" / "box was off" (IoU > 0 but < 0.5)
FP "on background" (IoU = 0) / "duplicate/sloppy" (IoU > 0)

outputs: summary.md, errors.csv (fill in 'category' by hand), crops/FN, crops/FP, overlays/
overlays: green = labels, red = predictions

usage:
    python error_analysis.py --images VAL_DIR --preds preds/baseline_val --sweep
    python error_analysis.py --images TEST_DIR --preds preds/baseline_test --conf 0.01 --out analysis/baseline
    python error_analysis.py --tally analysis/baseline/errors.csv
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png"}
# box area as % of image
SIZE_BINS = [(0, 0.25, "tiny (<0.25%)"), (0.25, 0.5, "small (0.25-0.5%)"),
             (0.5, 1.0, "medium (0.5-1%)"), (1.0, 101, "large (>1%)")]


def read_boxes(path: Path, with_conf: bool):
    cols = 6 if with_conf else 5
    if not path.exists() or not path.read_text().strip():
        return np.zeros((0, cols))
    rows = [list(map(float, line.split()))[:cols] for line in path.read_text().splitlines() if line.strip()]
    return np.array(rows, dtype=float)


def iou_matrix(a, b):
    # boxes are (x, y, w, h), centered
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a1, a2 = a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2
    b1, b2 = b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2
    lo = np.maximum(a1[:, None], b1[None])
    hi = np.minimum(a2[:, None], b2[None])
    inter = np.clip(hi - lo, 0, None).prod(-1)
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-12)


def match(gt, pred, iou_thr, match_class):
    # greedy, highest confidence first
    order = np.argsort(-pred[:, 5]) if len(pred) else np.array([], int)
    ious = iou_matrix(pred[:, 1:5], gt[:, 1:5])  # (n_pred, n_gt)
    gt_taken = np.zeros(len(gt), bool)
    pred_ok = np.zeros(len(pred), bool)
    for p in order:
        cand = ious[p].copy()
        cand[gt_taken] = -1
        if match_class and len(gt):
            cand[gt[:, 0] != pred[p, 0]] = -1
        if len(gt) and cand.max() >= iou_thr:
            g = int(cand.argmax())
            gt_taken[g] = True
            pred_ok[p] = True
    best_iou_pred = ious.max(1) if len(gt) else np.zeros(len(pred))
    best_iou_gt = ious.max(0) if len(pred) else np.zeros(len(gt))
    return pred_ok, gt_taken, best_iou_pred, best_iou_gt


def size_bin(w, h):
    pct = w * h * 100
    for lo, hi, name in SIZE_BINS:
        if lo <= pct < hi:
            return name
    return SIZE_BINS[-1][2]


def load_all(images_dir, labels_dir, preds_dir):
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit(f"no images in {images_dir}")
    data = []
    for img in imgs:
        gt = read_boxes(labels_dir / (img.stem + ".txt"), with_conf=False)
        pr = read_boxes(preds_dir / (img.stem + ".txt"), with_conf=True)
        data.append((img, gt, pr))
    return data


def score(data, conf, iou_thr, match_class):
    tp = fp = fn = 0
    for _, gt, pr in data:
        pr = pr[pr[:, 5] >= conf] if len(pr) else pr
        ok, taken, _, _ = match(gt, pr, iou_thr, match_class)
        tp += ok.sum(); fp += (~ok).sum(); fn += (~taken).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1


def crop(im, box, pad=1.5, min_px=96):
    W, H = im.size
    x, y, w, h = box
    half_w, half_h = max(w * W * pad, min_px) / 2, max(h * H * pad, min_px) / 2
    cx, cy = x * W, y * H
    return im.crop((int(max(cx - half_w, 0)), int(max(cy - half_h, 0)),
                    int(min(cx + half_w, W)), int(min(cy + half_h, H))))


def draw(im, gt, pr):
    im = im.copy(); d = ImageDraw.Draw(im); W, H = im.size
    for boxes, color in ((gt, (0, 200, 0)), (pr, (230, 0, 0))):
        for b in boxes:
            x, y, w, h = b[1:5]
            d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H],
                        outline=color, width=3)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="image folder")
    ap.add_argument("--labels", type=Path, help="defaults to --images")
    ap.add_argument("--preds", type=Path, required=True, help="predict.py output")
    ap.add_argument("--classes", type=Path, help="classes.txt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--match-class", action="store_true", help="species must match too")
    ap.add_argument("--sweep", action="store_true", help="try several conf cutoffs")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    ap.add_argument("--no-images", action="store_true", help="skip crops/overlays")
    args = ap.parse_args()

    labels_dir = args.labels or args.images
    data = load_all(args.images, labels_dir, args.preds)
    cls_file = args.classes or next((p for p in [args.images / "classes.txt",
                                                 args.images.parent / "classes.txt"] if p.exists()), None)
    names = cls_file.read_text().split() if cls_file else []
    cname = lambda k: names[int(k)] if int(k) < len(names) else str(int(k))
    pn_file = args.preds / "names.txt"
    pred_names = pn_file.read_text().split("\n") if pn_file.exists() else names
    pname = lambda k: pred_names[int(k)].strip() if int(k) < len(pred_names) else str(int(k))

    if args.sweep:
        print(f"{'conf':>7} {'TP':>5} {'FP':>6} {'FN':>5} {'prec':>6} {'recall':>6} {'F1':>6}")
        best = None
        for c in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.5]:
            tp, fp, fn, p, r, f1 = score(data, c, args.iou, args.match_class)
            print(f"{c:>7} {tp:>5} {fp:>6} {fn:>5} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
            if best is None or f1 > best[1]:
                best = (c, f1)
        print(f"\nbest F1 at conf={best[0]}")
        return

    out = args.out
    (out / "crops/FN").mkdir(parents=True, exist_ok=True)
    (out / "crops/FP").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    # clear pictures from an earlier run so old crops don't mix in
    for sub in ("crops/FN", "crops/FP", "overlays"):
        for old in (out / sub).glob("*.jpg"):
            old.unlink()

    rows = []
    gt_total, gt_found = Counter(), Counter()
    size_total, size_found = Counter(), Counter()
    fn_kind, fp_kind, fp_size = Counter(), Counter(), Counter()
    n_err = 0
    for img_path, gt, pr in data:
        pr = pr[pr[:, 5] >= args.conf] if len(pr) else pr
        ok, taken, iou_p, iou_g = match(gt, pr, args.iou, args.match_class)
        for g, b in enumerate(gt):
            sp, sz = cname(b[0]), size_bin(b[3], b[4])
            gt_total[sp] += 1; size_total[sz] += 1
            if taken[g]:
                gt_found[sp] += 1; size_found[sz] += 1
        errors = []
        for g in np.where(~taken)[0]:
            kind = "box was off" if iou_g[g] > 0 else "no guess nearby"
            fn_kind[kind] += 1
            errors.append(("FN", gt[g], kind, "", iou_g[g]))
        for p in np.where(~ok)[0]:
            kind = "duplicate/sloppy" if iou_p[p] > 0 else "on background"
            fp_kind[kind] += 1; fp_size[size_bin(pr[p, 3], pr[p, 4])] += 1
            errors.append(("FP", pr[p], kind, f"{pr[p, 5]:.3f}", iou_p[p]))
        if errors and not args.no_images:
            im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            draw(im, gt, pr).save(out / "overlays" / f"{img_path.stem}.jpg", quality=85)
        for etype, b, kind, conf, iou in errors:
            n_err += 1
            crop_name = f"{etype}_{n_err:04d}_{img_path.stem}.jpg"
            if not args.no_images:
                crop(im, b[1:5]).save(out / "crops" / etype / crop_name, quality=90)
            rows.append({"id": n_err, "image": img_path.name, "type": etype, "auto_kind": kind,
                         "label_or_guess_class": cname(b[0]) if etype == "FN" else pname(b[0]), "confidence": conf,
                         "best_iou": f"{iou:.2f}", "size": size_bin(b[3], b[4]),
                         "x": f"{b[1]:.4f}", "y": f"{b[2]:.4f}", "w": f"{b[3]:.4f}", "h": f"{b[4]:.4f}",
                         "crop": f"crops/{etype}/{crop_name}", "category": ""})

    with open(out / "errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader(); w.writerows(rows)

    tp, fp, fn, p, r, f1 = score(data, args.conf, args.iou, args.match_class)
    L = [f"# Error analysis: `{args.preds.name}`", "",
         f"{len(data)} images, confidence cutoff {args.conf}, IoU >= {args.iou}, "
         f"{'species must match' if args.match_class else 'any species counts'}", "",
         "| | count |", "|---|---|",
         f"| labeled insects | {tp + fn} |", f"| found (TP) | {tp} |",
         f"| missed (FN) | {fn} |", f"| false alarms (FP) | {fp} |",
         f"| **precision** | **{p:.3f}** |", f"| **recall** | **{r:.3f}** |", f"| F1 | {f1:.3f} |", "",
         "## Missed insects, by type", "", "| type | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in fn_kind.most_common()]
    L += ["", "## False alarms, by type", "", "| type | count |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in fp_kind.most_common()]
    L += ["", "## Recall by insect size (box area as % of photo)", "",
          "| size | insects | found | recall |", "|---|---|---|---|"]
    for _, _, nm in SIZE_BINS:
        if size_total[nm]:
            L.append(f"| {nm} | {size_total[nm]} | {size_found[nm]} | {size_found[nm] / size_total[nm]:.2f} |")
    L += ["", "## Recall by species", "", "| species | insects | found | recall |", "|---|---|---|---|"]
    for sp, n in sorted(gt_total.items(), key=lambda kv: gt_found[kv[0]] / kv[1]):
        L.append(f"| {sp} | {n} | {gt_found[sp]} | {gt_found[sp] / n:.2f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:15]))
    print(f"\nwrote {out}/summary.md, {out}/errors.csv ({len(rows)} mistakes)"
          + ("" if args.no_images else f", crops and overlays"))


def tally(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    by = defaultdict(Counter)
    for r in rows:
        by[r["type"]][r["category"].strip() or "(not sorted yet)"] += 1
    for t in ("FN", "FP"):
        total = sum(by[t].values())
        print(f"\n{'Missed insects (FN)' if t == 'FN' else 'False alarms (FP)'}: {total}")
        print("| category | count | share |\n|---|---|---|")
        for k, v in by[t].most_common():
            print(f"| {k} | {v} | {v / total:.0%} |")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--tally":
        tally(Path(sys.argv[2]))
    else:
        main()
