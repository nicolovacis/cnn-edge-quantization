# CNN Edge Quantization: Brevitas + FINN on Ultra96v2

## What is this repository?

This project explores quantization-aware training and hardware synthesis to deploy CNNs for classification and regression on resource-constrained embedded FPGAs (Ultra96v2).  
The target application is a light-based sensor that determines surface suitability based on optical ring reflection, with models designed to either:
- classify if the image is reliable for traditional estimation,
- or regress the diameter of the reflection for alignment analysis.

The full pipeline uses Brevitas to train quantized models and FINN to compile them into synthesizable hardware architectures.

## Repository Structure

```bash
cnn-edge-quantization/
├── brevitasTraining/
│   ├── Classification/
│   │   ├── CreateDataset.ipynb
│   │   ├── Size128_64/
│   │   │   └── Brevitas model training notebook
│   │   └── Size300/
│   │       ├── Brevitas, PyTorch, and TensorFlow notebooks
│   ├── Regression/
│   │   ├── CreateDatasetBoard.ipynb
│   │   ├── Size64/
│   │   ├── Size128/
│   │   └── Size300/
├── finnAccellerator/
│   └── advanced/
│       ├── Final FINN notebooks for synthesis and experiments
│       └── driver_example.py
└── README.md
```

## How to Use

### 1. Train and Export ONNX

Inside `brevitasTraining/`:

- Run `CreateDataset.ipynb` (Classification) or `CreateDatasetBoard.ipynb` (Regression) to prepare test datasets for board testing.
- For Classification:
  - Dataset creation is handled by an external notebook.
  - Each input size folder contains a Brevitas training notebook. Train and export to ONNX.
- For Regression:
  - Dataset creation is embedded inside each training notebook.
  - `CreateDatasetBoard.ipynb` produces the test dataset used on board.

Exported ONNX models are the input to FINN.

### 2. Build with FINN

Inside `finnAccellerator/advanced/`:

- Use any `4_advanced_builder_settings_*.ipynb` notebook depending on task and architecture.
- These notebooks handle ONNX transformation and build the FINN hardware pipeline.
- Includes a Max FIFO notebook used to try fixing FIFO depths manually (did not work reliably on board).

Use the final output folder to retrieve the deployable bitfile.

### 3. Deploy on Ultra96v2 FPGA

Modify `driver.py` as follows:

```python
from pynq import Device
device = Device.active_device
```

Then run inference:

#### Classification
```bash
sudo -E python3 driver.py --exec_mode execute \
  --bitfile ../bitfile/finn-accel.bit \
  --inputfile /home/xilinx/pynq/finn_cnn/dataset/classification/test_input_600samples_reverse_255values_size128.npy \
  --outputfile result.npy --batchsize 600
```

#### Regression
```bash
sudo -E python3 driver.py --exec_mode execute \
  --bitfile ../bitfile/finn-accel.bit \
  --inputfile /home/xilinx/pynq/finn_cnn/dataset/regression/X_test_size128_400samples_255datasetNew_uint8.npy \
  --outputfile result.npy --batchsize 400
```

#### Throughput test (Regression)
```bash
sudo -E python3 driver.py --exec_mode throughput_test \
  --bitfile ../bitfile/finn-accel.bit \
  --inputfile /home/xilinx/pynq/finn_cnn/dataset/regression/X_test_size64_400samples_255datasetNew_uint8.npy \
  --outputfile result.npy --batchsize 400
```

Alternative path example:
```bash
sudo -E python3 driver.py --exec_mode throughput_test \
  --bitfile /usr/local/share/pynq-venv/lib/python3.10/site-packages/pynq/finn_cnn/model/regression/toTest/.../bitfile/finn-accel.bit \
  --inputfile /home/xilinx/pynq/finn_cnn/dataset/regression/X_test_size64_400samples_255datasetNew_uint8.npy \
  --outputfile result.npy --batchsize 400
```

## Debugging Tips and Known Issues

- Use `QuantReLU` instead of standard ReLU to avoid export issues.
- BatchNorm must not be quantized.
- Softmax must be excluded from the ONNX model (apply outside the hardware block).
- Do not apply quantization after a Flatten layer.
- Use this order of passes in FINN:
  `streamline → manage_gap → streamline`
- Conv2D and Transpose layers are handled by `streamline`, not `manage_layer`.
- Use Netron to check the exported model structure and block removals.

### FIFO Issues

- On Ultra96v2, FINN often overestimates FIFO sizes.
- Using `split_large_fifos=True` is not sufficient to fix this.
- PE=1 and SIMD=1 are recommended.
- Using a single final dense neuron may crash with:
  ```
  FileNotFoundError: .../verilator_fifosim.../results.txt
  ```
  Solution: use at least 2 neurons in the output layer.

### Max FIFO

- Only worked when using `QuantIdentity`.
- Still resulted in hardware/software mismatch on board.

## Best Working Configurations

- Classification models are robust. 64x64 input, 8-bit weights, and filters 8–24 worked well (F1 > 0.87).
- Regression models need careful tuning:
  - Config A: 64x64, 8-bit weights, filters 8–24 → avg error = 34.77 µm
  - Config B: 64x64, 4-bit weights, filters 8–24 → avg error = 54.69 µm

## Summary

This repository provides a complete workflow for deploying CNNs on embedded FPGA with:
- Quantization-aware training using Brevitas
- Dataset preparation and export to ONNX
- Hardware synthesis using FINN
- Final bitstream generation and execution on Ultra96v2
- Hardware-aware hypertuning and performance trade-off analysis
