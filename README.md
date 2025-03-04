# FaCET-ESREL2025

![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-red.svg)
![NLP Badge](https://img.shields.io/badge/Dataset-NLP-blue.svg)

## 📝 Introduction

![alt text](FaCET.webp)

The addressed task in this paper is the **FaCET** task in which we aim at **Fa**iled **C**omponent **E**xtraction from **T**ext.

The dataset used in this work is **CuReFaCET** (Customer Review for Failed Component Extraction from Text). It is derived from teh CuReFaDe dataset and is created specifically for the FaCET task.

## ⚖️ License

This dataset is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## 🔍 Dataset Overview

The **[CuReFaCET dataset](https://github.com/jmpion/FaCET-ESREL2025/raw/refs/heads/master/data/CRD_components.xlsx)** is accessible in the `data` folder of this repository.

- **Size:** 745 kB
- **Format:** xlsx
- **Sheet names:**
    - `Reviews for CRD-FD`
    - `Component labels`

### `Reviews for CRD-FD`

- **Number of Records:** 1,215
- **Features:**
    - `Review_id`: a unique identifier for each customer review.
    - `Tablet_id`: a unique idenfitier for each tablet model.
    - `Comment`: the customer review textual comment.
    - `Stars`: a satisfaction score given in the form of a number of stars between 1 (very bad) and 5 (very good).
    - `Failure_class`: a failure label, either `IF` for **I**ntolerable **F**ailure, `TF` for **T**olerable **F**ailure, or empty for no failure.

### `Component labels`

- **Number of Records:** 356
- **Features:**
    - `Review_id`: a unique identifier for each customer review.
    - `Hinge`: the health state of the 'Hinge'.
    - `Left hinge`: the health state of the 'Left hinge'.
    - ...
    - ...
    - ...
    - `Volume control`: the health state of the 'Volume control'.
    - `Failure comment / Summary`: a summary of the mentioned failure(s).
    - `Uncertain data flag`: an indicator of whether the review is ambiguous or not.
    - `Time-to-failure`: the time-to-failure if and as indicated by the customer.

## ⬇️ Installation

### 💻 Virtual environment

#### ⚙️ Local virtual environment.

```bash
virtualenv .venv
```

```bash
source .venv/scripts/activate
```

## ▶️ Running code

All code should be run from the root. For instance `python src/main.py --model google/gemma-2-9b-it --prompt Good_baseline`.

## Metrics

For this project, I advocate against using Hamming Loss and Exact Match ratio, as the dataset is highly imbalanced towards instances with no failure. Such metrics will favor a dummy baseline predicting everything as not failed.

On the other hand, aggregated macro F1-score at the component level looks like a promising metric, which will directly penalize any constant dummy baseline.

F1-score at the component level can be defined as:

$$Metric(y, \hat{y}) = \dfrac{1}{p}\sum\limits_{j = 1}^p F_{macro}(y^j, \hat{y}^j)$$

where $j$ is an index iterating over all components, and $y^j$ is the vector of labels associated to component $j$, over all examples.

## 🤖 Models

| Model name | #Parameters | Context length |
|---|---|---| 
| [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | 9B | 8192 |
| [google/gemma-2-27b-it](https://huggingface.co/google/gemma-2-27b-it) | 27B | 8192 |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | 8B | 8000 |
|[meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)|8B|8000|
|[CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)|35B|128k|

## 📮 Contact Information

For any question, please contact: **jean.meunier-pion@centralesupelec.fr**