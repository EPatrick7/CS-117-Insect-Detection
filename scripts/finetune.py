"""Fine-tune YOLO on INSECT12C, then score on the test split.

usage:
    python finetune.py --data-root /content/INSECT12C --quick   # test run
    python finetune.py --data-root /content/INSECT12C
"""
import argparse
from pathlib import Path

import yaml

# zip 1 = test, last zip = val, rest = train
DEFAULT_TEST = ["part-01-12-img-1-150"]
DEFAULT_VAL = ["part-12-12-img-1651-1800"]


IMG_EXT = {".jpg", ".jpeg", ".png"}


def write_list(root: Path, parts, out: Path) -> str:
    # yolo-world needs one path per split, not a list of folders
    imgs = [str(p.resolve()) for part in parts for p in sorted((root / part).iterdir())
            if p.suffix.lower() in IMG_EXT]
    out.write_text("\n".join(imgs) + "\n")
    return str(out.resolve())


def build_data_yaml(root: Path, test_parts, val_parts, out: Path) -> Path:
    parts = sorted(p.name for p in root.iterdir()
                   if p.is_dir() and p.name.startswith("part-"))
    if not parts:
        raise SystemExit(f"no part-XX folders in {root}")
    for name in test_parts + val_parts:
        if name not in parts:
            raise SystemExit(f"{name} not found in {root}")
    train_parts = [p for p in parts if p not in test_parts + val_parts]
    names = (root / "classes.txt").read_text().split()
    data = {
        "path": str(root.resolve()),
        "train": write_list(root, train_parts, Path("train.txt")),
        "val": write_list(root, val_parts, Path("val.txt")),
        "test": write_list(root, test_parts, Path("test.txt")),
        "names": {i: n for i, n in enumerate(names)},
    }
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    n = lambda f: len(Path(f).read_text().split())
    print(f"wrote {out}: train {n('train.txt')} images ({len(train_parts)} parts), "
          f"val {n('val.txt')}, test {n('test.txt')}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, help="folder with unzipped part-XX folders")
    ap.add_argument("--data", type=Path, help="existing data.yaml")
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=800)  # images are 600x800
    ap.add_argument("--batch", type=int, default=8, help="lower if out of memory")
    ap.add_argument("--single-cls", action="store_true",
                    help="all species as one class")
    ap.add_argument("--quick", action="store_true",
                    help="10%% of data, 3 epochs")
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--resume", type=Path, help="last.pt to resume from")
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(str(args.resume)).train(resume=True)
        return

    if args.data:
        data_yaml = args.data
    elif args.data_root:
        data_yaml = build_data_yaml(args.data_root, DEFAULT_TEST, DEFAULT_VAL,
                                    Path("insect12c.yaml"))
    else:
        raise SystemExit("need --data-root or --data")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=3 if args.quick else args.epochs,
        fraction=0.1 if args.quick else 1.0,
        imgsz=args.imgsz,
        batch=args.batch,
        single_cls=args.single_cls,
        patience=10,
        save_period=5,  # in case colab disconnects
        seed=0,
        name=args.name + ("_quick" if args.quick else ""),
        exist_ok=True,
    )

    # test set
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nbest checkpoint: {best}")
    metrics = YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=args.imgsz,
                                  batch=args.batch, single_cls=args.single_cls,
                                  name=args.name + "_test", exist_ok=True)
    box = metrics.box
    print("\ntest set")
    print(f"  precision   {box.mp:.3f}")
    print(f"  recall      {box.mr:.3f}")
    print(f"  mAP@0.5     {box.map50:.3f}")
    print(f"  mAP@0.5:.95 {box.map:.3f}")


if __name__ == "__main__":
    main()
