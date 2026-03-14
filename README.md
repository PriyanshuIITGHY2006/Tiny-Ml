# Fruit Classification — TinyML Pipeline

RGB-based fruit classification running end-to-end from raw sensor data on an Arduino to a deployed model that makes predictions in real time. Two notebooks cover the full spectrum: classical machine learning and neural network training. Both pipelines produce a deployable C++ or C header file that runs on microcontrollers with as little as 10–20 KB of RAM.

---

## What This Project Does

An Arduino with a colour sensor reads the red, green, and blue light reflected off a piece of fruit. That three-number reading goes into a trained model. The model outputs the fruit name. The whole inference runs on the microcontroller itself — no cloud, no network, no PC required after deployment.

The two notebooks approach this from different angles. The classical ML notebook (`fruit_classification.ipynb`) produces a hand-crafted C++ class (`FruitChain.h`) containing a MinMaxScaler and a Random Forest encoded as pure if-else logic — no external library needed, runs on any 8-bit board. The neural network notebook (`fruit_nn_training.ipynb`) goes further: it trains, tunes, and quantises a multi-layer network down to int8 precision and exports it as a flat byte array (`fruit_model_int8.h`) for TFLite Micro.

---

## Project Structure

```
project/
│
├── fruits/                        # one CSV per fruit class
│   ├── apple.csv
│   ├── banana.csv
│   └── orange.csv
│
├── fruit_classification.ipynb     # classical ML pipeline
├── fruit_nn_training.ipynb        # neural network pipeline
├── requirements.txt
│
├── checkpoints/                   # saved Keras models (auto-created)
├── logs/                          # training CSVs and TensorBoard logs (auto-created)
│
├── FruitChain.h                   # output of notebook 1 — Random Forest in C++
├── fruit_model_int8.h             # output of notebook 2 — TFLite int8 byte array
└── fruit_model_int8.tflite        # output of notebook 2 — raw TFLite binary
```

If you collected a single combined CSV instead of one file per class, place `fruits.csv` in the project root and see the data loading note in each notebook.

---

## Hardware

One of the following:

- Arduino Nano 33 BLE Sense — has an APDS9960 colour and proximity sensor built in
- Any other microcontroller + an external TCS3200 colour sensor wired to digital output pins

For `FruitChain.h` deployment (notebook 1), any board that compiles C++ works. For TFLite Micro deployment (notebook 2), you need a 32-bit board — the Nano 33 BLE Sense qualifies.

Fruits or any objects with clearly distinct colours. Avoid two objects that look similar under the sensor (banana and lemon, for example).

---

## Setup

### Python environment

Python 3.10 or higher is required.

```bash
python -m venv tinyml
```

Activate:

```bash
# macOS / Linux
source tinyml/bin/activate

# Windows Command Prompt
tinyml\Scripts\activate.bat

# Windows PowerShell
tinyml\Scripts\activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are on an Apple Silicon Mac, replace the `tensorflow` line in requirements.txt with `tensorflow-macos` and `tensorflow-metal` before installing.

If you do not need XGBoost (it is optional — the notebook skips it gracefully), you can drop that line too.

### Arduino IDE

Install version 2.x from arduino.cc. Open the Library Manager (`Sketch → Include Library → Manage Libraries`) and install:

- `tinyml4all` — companion library used in data capture sketches
- `Arduino_APDS9960` — if using the Nano 33 BLE Sense
- `TensorFlowLite` — required only if deploying the neural network model (notebook 2)

---

## Data Collection

Skip this section if you already have CSVs.

Flash the capture sketch from the book (`Listing 2-1` or `Listing 2-2`) onto the board. Open the Serial Monitor at 115200 baud. Point the board at the fruit from 15–30 cm. Collect 30–50 samples per fruit while moving the sensor slightly to capture variation.

**Option A — Manual copy paste**

Copy the Serial Monitor output into a text file. Remove the prompt lines, add a header row `r,g,b,fruit`, save as `fruits.csv` in the project root.

**Option B — Automated Python capture**

With the board flashed and connected:

```bash
python capture_colors.py
```

The script prompts for the fruit name and sample count, reads from the serial port, and appends rows to `fruits.csv` automatically.

**Option C — SD card**

Flash the SD card sketch, collect data unplugged, then copy files from the card to the `fruits/` folder and add headers. Each file should be named after the class it contains: `banana.csv`, etc.

Whatever method you use, the expected format inside each file is:

```
r,g,b
38,23,18
40,22,18
```

Or, if using a single combined file:

```
r,g,b,fruit
38,23,18,orange
17,12,9,banana
```

---

## Notebook 1 — Classical ML (`fruit_classification.ipynb`)

### What it does

Runs a complete scikit-learn pipeline: load → clean → EDA → feature engineering → train five classifiers → compare → export a deployable C++ header.

### How to run

```bash
jupyter notebook fruit_classification.ipynb
```

Run cells top to bottom. Nothing requires modification unless you are using the single-file layout — in that case, swap the data loading cell as noted in the comment at the top of Section 2.

### What each section produces

**EDA** — histogram grids, box plots, a pair plot matrix, a correlation heatmap, and a t-SNE scatter. The t-SNE plot is the most important one to look at before training: if the three fruit clouds do not separate visually, you have a problem with data quality, not model choice.

**Cleaning** — duplicates dropped, nulls dropped, 3σ outlier removal per channel.

**Feature engineering** — MinMaxScaler fitted on training data only and applied to test. Three feature selection methods are run (univariate F-score, sequential forward selection, RFE) so you can see which channels matter most. For this dataset all three channels are kept because the dataset is small.

**Five classifiers** — Decision Tree, Random Forest, XGBoost, Logistic Regression, SVM. All share the same train/test split. Cross-validated F1 score and held-out test accuracy are reported for each.

**FruitChain pipeline** — Random Forest is chosen as the final estimator. The scaler and classifier are wrapped in a `sklearn.Pipeline`, retrained on the full dataset, and passed to the C++ code generator.

### The generated FruitChain.h

The code generator at the end of the notebook writes out a self-contained C++ header. It contains:

- The MinMaxScaler parameters (`SCALE_MIN[]` and `SCALE_RANGE[]` arrays)
- Every decision tree in the Random Forest, each compiled to a standalone static function of nested `if-else` blocks
- A `predict_tree(t, x)` dispatcher that calls the right tree by index
- A voting loop that tallies predictions across all trees and picks the winner
- The predicted label as a null-terminated string and as an integer index, exposed through `chain.output.label` and `chain.output.idx`

The class lives in the `tinyml4all` namespace and has a single public interface:

```cpp
bool operator()(float r, float g, float b)
```

Call it with raw sensor values. It normalises them internally using the stored scale parameters, runs all trees, and updates `output`. Returns `true` on success.

There are no heap allocations, no dynamic memory, no external dependencies. The if-else tree logic compiles to pure comparisons and branches — the fastest possible inference on constrained hardware.

### Deploying FruitChain.h to Arduino

Copy `FruitChain.h` into the same folder as your `.ino` sketch. The deployment sketch is in the last section of the notebook as a reference. The relevant part is:

```cpp
#include "./FruitChain.h"

tinyml4all::FruitChain chain;

void loop() {
    sensor.readColor();
    if (!chain(sensor.r, sensor.g, sensor.b)) return;
    Serial.println(chain.output.classification.label);
    delay(1000);
}
```

Upload, open Serial Monitor at 115200 baud, point at fruit.

---

## Notebook 2 — Neural Network (`fruit_nn_training.ipynb`)

### What it does

Trains multiple neural network architectures from scratch, performs a manual hyperparameter grid search, runs 5-fold cross-validation, fine-tunes the best model in two phases, analyses calibration, then quantises to int8 and exports a TFLite byte array ready for TFLite Micro on Arduino.

### How to run

```bash
jupyter notebook fruit_nn_training.ipynb
```

Training 32 models in the grid search takes a few minutes on CPU for this dataset. If you want to skip the grid search and just use the default architecture, comment out the loop in Section 8 and set `best_units`, `best_dr`, `best_l2`, `best_bn` manually before Section 11.

TensorBoard logs are written to `logs/`. To view them in a separate terminal:

```bash
tensorboard --logdir logs/
```

Then open `http://localhost:6006` in a browser.

### Architecture progression

The notebook trains architectures in order of increasing sophistication:

**Baseline MLP** — two hidden layers (16 → 8 neurons). Establishes the floor. If a two-layer network already hits high accuracy on this dataset, every subsequent complication needs to justify itself.

**Regularised deep MLP** — three hidden layers (64 → 32 → 16) with BatchNormalisation, Dropout, and L2 weight decay applied together. BatchNorm stabilises gradient flow between layers. Dropout prevents co-adaptation of neurons. L2 penalises large weights. The three work at different scales and are complementary — removing any one of them measurably hurts performance on datasets where overfitting is a risk.

**Residual MLP** — adds skip connections between layers. Each residual block computes `output = F(input) + input`. This lets gradients flow directly back through the shortcut path during backpropagation, which matters for deeper networks. On a 3-feature dataset this is architectural overkill but it demonstrates the implementation pattern.

### Grid search details

32 combinations are evaluated across:
- Layer widths: (32, 16), (64, 32), (64, 32, 16), (128, 64, 32)
- Dropout: 0.2, 0.3
- L2: 1e-4, 1e-3
- BatchNorm: on/off

Each model trains until early stopping triggers (patience=20), then test F1 is recorded. The best configuration is carried forward to all subsequent steps.

### Fine-tuning

Two phases:

**Phase 1** — all layers except the final Dense layer are frozen (`trainable=False`). Only the output layer is updated. Learning rate: 1e-4. This reconditions the decision boundary without disturbing the learned feature representations.

**Phase 2** — all layers unfrozen, LR reduced to 1e-5 (fifty times smaller than the original training LR). Subtle end-to-end refinement. The low LR is not optional — higher values will overwrite good weights rather than improve them.

### Quantisation

Three levels are produced and evaluated side-by-side:

**Float32 TFLite** — identical accuracy to the Keras model. Just converts the format, no precision loss. Baseline for size/accuracy comparison.

**Dynamic-range int8** — weights compressed from float32 to int8 (4× smaller). Activations stay float32. No calibration data needed. Good first step if the board has an FPU.

**Full integer int8** — weights and activations both int8. Requires 200 representative training samples to calibrate activation ranges. Input and output tensors are also int8. This is what runs on the Nano 33 BLE Sense with TFLite Micro. Boards without an FPU (Cortex-M0, M0+) can only run this variant efficiently.

### The generated fruit_model_int8.h

The final export cell converts the `.tflite` binary to a C header containing:

```c
const unsigned int   fruit_model_int8_len  = 2048;
const unsigned char  fruit_model_int8_data[] = { 0x1c, 0x00, ... };
```

When included in an Arduino sketch, this array sits in flash memory. TFLite Micro reads from it directly using a `MicroInterpreter`. No SD card, no SPIFFS, no filesystem.

### Deploying fruit_model_int8.h to Arduino

The full sketch is in Section 19 of the notebook. Key points:

The scale parameters must match what the Python scaler learned. Print them at the end of the notebook:

```python
print("SCALE_MIN  =", scaler.data_min_.tolist())
print("SCALE_RANGE=", (scaler.data_max_ - scaler.data_min_).tolist())
```

Paste those values into the `SCALE_MIN[]` and `SCALE_RANGE[]` arrays in the sketch.

The `CLASS_NAMES[]` array must be in the same order as `le.classes_` from the notebook — alphabetical by default. Print it with:

```python
print(le.classes_)
```

Inference loop:

```cpp
// normalise
float norm = (raw_value - SCALE_MIN[i]) / SCALE_RANGE[i];

// quantise to int8
input->data.int8[i] = (int8_t)(norm / input->params.scale + input->params.zero_point);

// run
interpreter->Invoke();

// dequantise output and pick winner
float p = (output->data.int8[c] - output->params.zero_point) * output->params.scale;
```

`TENSOR_ARENA_SIZE` in the sketch is the working memory TFLite Micro needs for intermediate computations. Start with 8 KB and reduce by 1 KB increments until `AllocateTensors()` starts returning errors — the lowest value that succeeds is your minimum.

---

## Choosing Between the Two Models

| | FruitChain.h | fruit_model_int8.h |
|---|---|---|
| Board requirement | Any C++ board (8 or 32-bit) | 32-bit, TFLite Micro support |
| External library | None | TensorFlowLite Arduino |
| Flash usage | ~5–15 KB depending on forest size | ~8–50 KB depending on network size |
| RAM usage | ~0 KB (pure if-else) | ~8–16 KB (tensor arena) |
| Inference speed | Microseconds | Milliseconds |
| Accuracy ceiling | Good, rarely beats neural networks on complex data | Higher ceiling, especially with more classes |

For this specific 3-feature 3-class problem both models perform comparably. On a richer dataset — more classes, more sensors, images — the neural network gap widens.

---

## Common Issues

**`FileNotFoundError: fruits/`** — the notebook looks for the `fruits/` folder relative to wherever Jupyter was launched from. Run `jupyter notebook` from the project root folder, not from inside a subfolder.

**Serial port error during capture** — close the Arduino IDE Serial Monitor first. Python and the IDE cannot both hold the port. On Linux you may need to add yourself to the `dialout` group: `sudo usermod -a -G dialout $USER`, then log out and back in.

**`ModuleNotFoundError: tinyml4all`** — you forgot to activate the virtual environment before starting Jupyter. Quit Jupyter, run `source tinyml/bin/activate`, then restart.

**Grid search takes too long** — reduce the grid in Section 8 to two architectures and one value each for dropout and L2. The search is illustrative; the exact winner depends on your dataset anyway.

**TFLite int8 accuracy drops significantly** — this usually means the representative dataset in Section 16 was too small or unrepresentative. Increase from 200 to the full training set: `for sample in X_train:`.

**`AllocateTensors()` fails on Arduino** — increase `TENSOR_ARENA_SIZE`. Try 16384 (16 KB). The Nano 33 BLE Sense has 256 KB RAM so space is not the bottleneck — start high and tune down.

---

## Notes on the "Unknown Object" Problem

Both deployed models will always predict one of the trained classes regardless of what is in front of the sensor. Point the sensor at a wall and it will confidently name a fruit.

This is not a bug. Classification models operate in a closed world — they only know what they were shown during training. If you want the model to say "not a fruit", collect 30–50 samples of the sensor pointing at neutral surfaces (desk, wall, air) and label them as a `none` class. Retrain. The model will then have a class to assign when nothing recognisable is present.

---

## Reproducing Results

All random operations in both notebooks use `SEED = 42`. Set once at the top, propagated to NumPy, scikit-learn, and TensorFlow. Running the notebooks top-to-bottom on the same data will produce identical numbers every time.

The only source of non-determinism is GPU parallelism in TensorFlow. If exact reproducibility across machines matters, add `os.environ['TF_DETERMINISTIC_OPS'] = '1'` before importing TensorFlow. This slows down GPU training but removes the last source of variance.
