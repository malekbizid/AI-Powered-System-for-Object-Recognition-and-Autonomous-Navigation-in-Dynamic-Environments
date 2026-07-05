# YOLO-NAV: AI-Powered System for Object Recognition and Autonomous Navigation in Dynamic Environments

## Overview

YOLO-NAV is an AI-powered object detection framework designed for autonomous navigation in dynamic driving environments. The project focuses on real-time recognition of road-related objects such as vehicles, pedestrians, and other relevant obstacles using YOLOv11-based object detection.

The system was developed to support research on autonomous vehicles by combining:

- a custom Roboflow dataset for initial training and baseline evaluation;
- the NuImages dataset for more diverse autonomous driving scenarios;
- a YOLO-compatible dataset conversion pipeline;
- YOLOv11x training and evaluation;
- real-time object detection experiments and result visualization.

The objective of this repository is to provide a reproducible implementation aligned with the methodology and results reported in the associated manuscript.

---

## Repository Structure

The repository is organized as follows:

```text
├── Roboflow_dataset/
│   └── Custom dataset used for baseline training and preliminary experiments.
│
├── convert_nuimages_dataset/
│   └── Scripts used to convert NuImages annotations into YOLO-compatible format.
│
├── yolov11x_nuimages/
│   └── YOLOv11x training, validation, and evaluation files for the NuImages dataset.
│
└── README.md
```

### Folder description

| Folder | Description |
|---|---|
| `Roboflow_dataset/` | Contains the custom dataset prepared using Roboflow. This dataset is used as the initial training baseline. |
| `convert_nuimages_dataset/` | Contains the conversion scripts required to transform NuImages annotations into YOLO format. |
| `yolov11x_nuimages/` | Contains the YOLOv11x configuration files, training outputs, evaluation results, and related experiment files. |

---

## Methodology Summary

The experimental workflow follows four main stages.

### 1. Dataset preparation

Two datasets are used:

1. **Custom Roboflow dataset**  
   This dataset contains street-view images relevant to autonomous driving scenarios. It is used to establish the first training baseline and to allow the model to learn initial visual features related to road environments.

2. **NuImages dataset**  
   NuImages is used to improve model generalization and evaluate the system on more diverse autonomous driving scenes.

The NuImages annotations are converted into YOLO format before training.

---

### 2. Preprocessing

The following preprocessing operations are applied:

- image auto-orientation;
- resizing images to a fixed resolution of `640 × 640`;
- conversion of annotations to YOLO format;
- dataset organization into training, validation, and test subsets.

The dataset split used in the manuscript is:

| Subset | Percentage |
|---|---:|
| Training set | 65% |
| Validation set | 20% |
| Test set | 15% |

---

### 3. Data augmentation

To improve robustness under real-world driving conditions, the following augmentation techniques are used:

- horizontal flipping;
- brightness variation;
- saturation adjustment;
- random noise injection.

These augmentations are intended to simulate visual variations caused by lighting changes, camera noise, shadows, glare, and different road-scene conditions.

---

### 4. Model training and evaluation

The main detection model used in this project is **YOLOv11x**.

The model is trained and evaluated using standard object detection metrics:

- Precision;
- Recall;
- mAP50;
- F1-score;
- confusion matrix;
- training and validation box loss.

---

## Requirements

The implementation requires Python and common deep learning/computer vision libraries.

Recommended environment:

```text
Python >= 3.9
PyTorch
Ultralytics
OpenCV
NumPy
Pandas
Matplotlib
PyYAML
```

You can install the main dependencies using:

```bash
pip install ultralytics opencv-python numpy pandas matplotlib pyyaml
```

If GPU acceleration is used, make sure that the installed PyTorch version is compatible with your CUDA version.

---

## Dataset Preparation

### Custom Roboflow dataset

Place the exported Roboflow dataset inside:

```text
Roboflow_dataset/
```

The expected YOLO dataset structure is:

```text
Roboflow_dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

---

### NuImages dataset conversion

The NuImages dataset must be converted to YOLO format before training.

Use the scripts available in:

```text
convert_nuimages_dataset/
```

A typical command may be:

```bash
python convert_nuimages_dataset/conversion_script.py
```

After conversion, the expected output structure is:

```text
nuimages_yolo/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

Make sure that the paths inside `data.yaml` correctly point to the training, validation, and test directories.

---

## Training

To train YOLOv11x on the prepared dataset, use the following command:

```bash
yolo detect train model=yolo11x.pt data=data.yaml imgsz=640 epochs=100 batch=16
```

The main parameters are:

| Parameter | Description |
|---|---|
| `model=yolo11x.pt` | Pretrained YOLOv11x model used as the initial detector. |
| `data=data.yaml` | Dataset configuration file. |
| `imgsz=640` | Input image size. |
| `epochs=100` | Number of training epochs. |
| `batch=16` | Batch size. Adjust according to GPU memory. |

If your hardware has limited memory, reduce the batch size:

```bash
yolo detect train model=yolo11x.pt data=data.yaml imgsz=640 epochs=100 batch=8
```

---

## Validation and Evaluation

After training, the best model weights are usually saved in:

```text
runs/detect/train/weights/best.pt
```

To evaluate the model on the validation or test set:

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml imgsz=640
```

The evaluation outputs include:

- Precision;
- Recall;
- mAP50;
- mAP50-95;
- confusion matrix;
- prediction examples;
- loss curves.

---

## Inference

To run inference on new images:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/images imgsz=640
```

To run inference on a video:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4 imgsz=640
```

To run inference using a webcam:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=0 imgsz=640
```

---

## Reproducing the Reported Experiments

To reproduce the experiments reported in the manuscript, follow these steps:

1. Prepare the custom Roboflow dataset in YOLO format.
2. Download and organize the NuImages dataset.
3. Convert NuImages annotations to YOLO format using the conversion scripts.
4. Verify the dataset paths inside `data.yaml`.
5. Train YOLOv11x using the training command provided above.
6. Evaluate the trained model using the validation command.
7. Compare the generated metrics with the values reported in the manuscript.

The main experimental configuration is summarized below:

| Item | Configuration |
|---|---|
| Model | YOLOv11x |
| Input image size | 640 × 640 |
| Dataset 1 | Custom Roboflow dataset |
| Dataset 2 | NuImages dataset |
| Dataset split | 65% train, 20% validation, 15% test |
| Task | Object detection for autonomous navigation |
| Metrics | Precision, Recall, mAP50, F1-score |
| Output | Bounding boxes, class labels, confidence scores |

---

## Expected Results

The exact results may vary depending on:

- hardware configuration;
- random seed;
- training duration;
- batch size;
- dataset version;
- preprocessing and augmentation settings.

A results table should be reported using the following format:

| Dataset | Model | Precision | Recall | mAP50 | F1-score |
|---|---|---:|---:|---:|---:|
| Custom Roboflow dataset | YOLOv11x | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| NuImages dataset | YOLOv11x | XX.XX% | XX.XX% | XX.XX% | XX.XX% |

Replace the `XX.XX%` values with the final values obtained from the trained model outputs.

---

## Ablation Study Reproducibility

For ablation experiments, the following variants can be evaluated:

| Variant | Preprocessing | Augmentation | Two-stage training | Metrics |
|---|---|---|---|---|
| Baseline YOLOv11x | No | No | No | Precision, Recall, mAP50, F1-score |
| YOLOv11x + preprocessing | Yes | No | No | Precision, Recall, mAP50, F1-score |
| YOLOv11x + preprocessing + augmentation | Yes | Yes | No | Precision, Recall, mAP50, F1-score |
| Final YOLO-NAV configuration | Yes | Yes | Yes | Precision, Recall, mAP50, F1-score |

All variants should be trained using the same input size, optimizer settings, batch size, number of epochs, and evaluation protocol to ensure a fair comparison.

---

## Real-Time Performance Evaluation

For real-time evaluation, report the following information:

| Metric | Description |
|---|---|
| Latency | Average inference time per frame, usually in milliseconds. |
| FPS | Number of processed frames per second. |
| Input size | Image resolution used during inference. |
| Batch size | Batch size used during inference, usually 1 for real-time systems. |
| Hardware | CPU/GPU/embedded platform used for testing. |
| Memory usage | GPU or system memory consumed during inference. |

Example command for prediction:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/video.mp4 imgsz=640
```

If edge deployment is considered, additional optimization steps may be required:

- ONNX export;
- TensorRT acceleration;
- model pruning;
- quantization;
- use of lighter YOLOv11 variants.

Example export command:

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640
```

---

## Output Files

Typical training and evaluation outputs are saved under:

```text
runs/detect/
```

Common output files include:

```text
runs/detect/train/results.png
runs/detect/train/confusion_matrix.png
runs/detect/train/weights/best.pt
runs/detect/train/weights/last.pt
runs/detect/val/confusion_matrix.png
runs/detect/predict/
```

These files can be used to verify the training process, evaluate the model, and reproduce the figures reported in the manuscript.

---

## Notes on Reproducibility

To improve reproducibility, users are encouraged to report:

- Python version;
- PyTorch version;
- Ultralytics version;
- CUDA version;
- GPU model;
- CPU model;
- RAM size;
- random seed;
- dataset version;
- exact training command.

Example:

```bash
yolo checks
```

This command displays useful information about the current YOLO and hardware environment.

---

## Limitations

This repository provides the implementation used for the YOLO-NAV object detection pipeline. However, exact reproduction of the reported results may depend on the availability of the same dataset versions, hardware configuration, and training settings.

Deployment on low-power embedded platforms may require additional optimization, including model compression, quantization, TensorRT acceleration, or the use of smaller YOLOv11 variants.

---

## Citation

If you use this repository or build upon this work, please cite the associated manuscript:

```bibtex
@article{yolonav2026,
  title={YOLO-NAV: An AI-Powered System for Object Recognition and Autonomous Navigation in Dynamic Environments},
  author={},
  journal={Neural Computing and Applications},
  year={2026}
}
```

Please update the BibTeX entry once the final publication details are available.

---

## Contact

For questions or reproducibility issues, please open an issue in the GitHub repository or contact the authors of the associated manuscript.

---

## License

Please add the appropriate license for this repository before public reuse.

Recommended options include:

- MIT License;
- Apache License 2.0;
- Creative Commons license for documentation only.

