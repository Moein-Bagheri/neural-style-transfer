# Neural Style Transfer

An image stylization project using **VGG19 features and PyTorch**. It combines the structure of a content image with the visual style of a reference image through iterative optimization, then exports the result and an analysis figure.

**Python · PyTorch · torchvision · Pillow · NumPy · Matplotlib**

## Saved examples

The original project includes an architectural content image and Bauhaus, Flat, Japanese and Luxury style references. These two results are retained from the completed runs:

| Content | Bauhaus result | Luxury result |
| --- | --- | --- |
| ![Content image](input/BRUTALISM.jpg) | ![Bauhaus result](output/professional_BRUTALISM_BAUHAUS_1753043197.jpg) | ![Luxury result](output/professional_BRUTALISM_LUXURY_1753048282.jpg) |

Open the analysis figures for the source/style/result comparison and optimization history:

- [Bauhaus analysis](output/professional_BRUTALISM_BAUHAUS_1753043197_analysis.png)
- [Luxury analysis](output/professional_BRUTALISM_LUXURY_1753048282_analysis.png)

## Method

- A pretrained VGG19 network provides intermediate image features.
- Content and style objectives are combined with image regularization.
- Optimization proceeds through multiple stages.
- The interface offers 256 × 256 and 512 × 512 image sizes.
- Final images and comparison plots are written to `output/`.

## Run

Open Windows PowerShell in this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe NeuralStyleTransfer.py
```

At the prompts, enter:

```text
Content image path: input/BRUTALISM.jpg
Style image path: styles/BAUHAUS.jpg
Choose (1-2): 2
```

The program may download the pretrained VGG19 weights on first use. It chooses an available compute device through the original implementation. `requirements.txt` lists the imported libraries, without claiming a version-locked environment.

## Repository layout

| Path | Contents |
| --- | --- |
| [`NeuralStyleTransfer.py`](NeuralStyleTransfer.py) | Feature extraction, losses, optimization and interactive entry point |
| [`input/`](input/) | Original content image |
| [`styles/`](styles/) | Four original style reference images |
| [`output/`](output/) | Two saved results and their analysis figures |

Developed as university coursework in machine learning. The example images and generated results are retained from the project archive.
