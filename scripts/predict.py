"""Run a model on a folder of images and save predictions for error_analysis.py.

output: one txt per image, "class x y w h conf" (normalized)

usage:
    python predict.py --model yolov8s-worldv2.pt --prompt insect --source DIR --out preds/baseline_test
    python predict.py --model runs/detect/finetune/weights/best.pt --source DIR --out preds/finetuned_test
"""
import argparse
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", type=Path, required=True, help="image folder")
    ap.add_argument("--out", type=Path, required=True, help="output folder")
    ap.add_argument("--prompt", nargs="*", help='yolo-world classes, e.g. insect')
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.001)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    if args.prompt:
        model.set_classes(args.prompt)

    images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMG_EXT)
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for img in images:
        r = model.predict(str(img), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        lines = []
        for (x, y, w, h), c, k in zip(r.boxes.xywhn.tolist(), r.boxes.conf.tolist(),
                                      r.boxes.cls.tolist()):
            lines.append(f"{int(k)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {c:.5f}")
        (args.out / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)
    # so error_analysis knows the class names
    (args.out / "names.txt").write_text("\n".join(model.names[i] for i in sorted(model.names)) + "\n")
    print(f"{len(images)} images, {total} boxes saved to {args.out}")


if __name__ == "__main__":
    main()
