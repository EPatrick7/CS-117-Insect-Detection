# Finding insects on soybean leaves with YOLO-World

**UC Irvine CS 117 · Fall 2026 · Team 12**

Eric Patrick, Joshua Nunez, William Park, and Xavier Chung

How much does a short round of fine-tuning help a general-purpose detector find pests on soybean leaves? We give YOLOv8s-WorldV2 the text prompt `insect` and ask it to draw a box around each pest. We compare the pretrained model with a copy trained for three epochs on annotated field photos. Both models see the same evaluation photos and use the same prompt. The 12 original pest categories all count as one target. [Experiment notebook](02_stock_insect_baseline.ipynb) · [CS 117 project description](https://jhoffman.github.io/cs117-fa26/project/)

## Run the comparison

1. From the repository root, use a Python 3.12 environment with Git installed. Install the dependencies and register the notebook kernel.

   ```sh
   python -m pip install -r requirements.txt
   python -m ipykernel install --user --name cs117 --display-name "CS117"
   ```

2. Open the [experiment notebook](02_stock_insect_baseline.ipynb) in Jupyter or VS Code, select **CS117**, and run the cells in order. **Score** evaluates the stock weights. **Fine-tune** starts from the same weights, trains with 640-pixel inputs and a batch size of 4, and saves the validation-selected checkpoint under `runs/insect_burst_experiments/`. The notebook reads the local photos and annotations directly. [Dataset configuration](insect12c.yaml)

3. The comparison table scores the entire selected split. The gallery shows three sampled validation photos at a separate cutoff of **0.25**. Dashed blue boxes are human marks. Solid orange boxes are model predictions. Change `gallery_seed` to see another sample. To measure precision and recall at the gallery cutoff, set `settings["conf"] = 0.25` in **Settings**, then rerun **Settings** and the comparison cell.

The [original smoke test](01_yolo_world_v2.ipynb) is a shorter introduction. The comparison notebook builds on the team’s [stock baseline](scripts/baseline.py), [fine-tuning script](scripts/finetune.py), and [box-matching code](scripts/error_analysis.py).

## Sources and rights

**INSECT12C photos and annotations.** The dataset creators are Everton Castelão Tetila, Fábio Amaral Godoy da Silveira, Anderson Bessa da Costa, Willian Paraguassu Amorim, Gilberto Astolfi, Hemerson Pistori, and Jayme Garcia Arnal Barbedo. Their requested citation is “YOLO performance analysis for real-time detection of soybean pests,” *Smart Agricultural Technology* 7 (2024), article 100405. [Paper and DOI](https://doi.org/10.1016/j.atech.2024.100405) · [Open copy at Embrapa](https://www.alice.cnptia.embrapa.br/alice/handle/doc/1161634)

The authors make INSECT12C available for academic research. Their release does not specify a redistribution license for the photographs or annotations. For permission to redistribute those files, contact the dataset authors. [Dataset revision used](https://github.com/EvertonTetila/INSECT12C-Dataset/tree/ffa37d73b7a0bece6385236b336ab4c36497bc27) · [Authors’ download page](https://evertontetila.ws.ufgd.edu.br/Datasets/)

**YOLO-World and Ultralytics.** Tianheng Cheng and colleagues introduced YOLO-World at CVPR 2024. We used Ultralytics’ implementation and pretrained YOLOv8s-WorldV2 weights. Ultralytics identifies the stock checkpoint as AGPL-3.0. [YOLO-World paper](https://arxiv.org/abs/2401.17270) · [Model documentation](https://docs.ultralytics.com/models/yolo-world) · [Official weights](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt) · [Ultralytics licensing](https://www.ultralytics.com/license)

The root `predict.py`, `download.py`, and `cog.yaml` come from Ultralytics’ Replicate starter. The smoke-test folder includes `bus.jpg` and `zidane.jpg` from the Ultralytics sample image collection. The notebook uses Ultralytics’ fork of OpenAI CLIP to encode the prompt. [Original starter](https://github.com/ultralytics/replicate/tree/0a50b5d7a86c907107461b554728e676086c2e62/yolov8s-worldv2) · [Ultralytics sample images](https://github.com/ultralytics/assets/tree/main/im) · [CLIP source and license](https://github.com/ultralytics/CLIP/tree/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40)
