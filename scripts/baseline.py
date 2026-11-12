"""Baseline of untrained YOLO-World with the prompt "insect", scored on the test split

usage:
    python scripts/finetune.py --data-root data/INSECT12C --quick # writes insect12c.yaml
    python scripts/baseline.py
"""
from ultralytics import YOLOWorld

# pretrained weights below
model = YOLOWorld("models/yolov8s-worldv2.pt")
model.set_classes(["insect"])

metrics = model.val(data="insect12c.yaml", split="test", conf=0.001, single_cls=True)

box = metrics.box
print(f"precision   {box.mp:.3f}")
print(f"recall      {box.mr:.3f}")
print(f"mAP@0.5     {box.map50:.3f}")
print(f"mAP@0.5:.95 {box.map:.3f}")
