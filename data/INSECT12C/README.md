# INSECT12C

These 1,800 soybean insect photos and their annotations come from [Everton Tetila and colleagues’ INSECT12C dataset](https://github.com/EvertonTetila/INSECT12C-Dataset/tree/ffa37d73b7a0bece6385236b336ab4c36497bc27). The authors describe the dataset in [*YOLO performance analysis for real-time detection of soybean pests* (2024)](https://doi.org/10.1016/j.atech.2024.100405).

Each photo has a YOLO text label and an XML annotation. The upstream archive supplied only XML for `img1146`, so its YOLO label was converted from that annotation.

The notebook reads the image lists in this folder. `train_images.txt` contains parts 02–11 (1,500 photos), `val_images.txt` contains part 12 (150), and `test_images.txt` contains part 01 after excluding the ten original smoke-test photos (140). Paths in each list are relative to this folder.
