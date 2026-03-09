# CLIP Search

A CLIP powered application for searching through images using natural language.

## Setup

1. Clone the Repository:
```bash
git clone https://github.com/Abdullah-hmed/clip-search.git
```

2. Move into the repository:
```bash
cd clip-search
```

3. Initialize a virtual environment:
```bash
python -m venv .venv
```

4. Source the virtual environment:
```bash
.venv\Scripts\activate
```

5. Install all the dependencies
```bash
pip install -r requirements.txt
```

> [!TIP]
> **If CUDA acceleration isn't working**
>
> If the terminal prints **"Loading OpenCLIP ViT-L/14 on cpu…"** on startup, PyTorch may not be using CUDA.
>
> You can reinstall PyTorch with CUDA support using:
>
> ```bash
> pip uninstall torch torchvision
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```
6. Once the dependencies are installed, either double click the `run.bat` file, or run the command 
```bash
python clip_search_gui.py
```
