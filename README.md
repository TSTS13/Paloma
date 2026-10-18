# Paloma: Phase-Conditioned Residual Modulation for Time Series Forecasting (NeurIPS 2026)

If you find this repository useful, please consider giving it a ⭐ Star to show your support. Thank you!

🚩 News (2026.09) Paloma has been accepted as a Spotlight poster in NeurIPS 2026.

##  🚀 Getting Started

### Environment Requirements

*   Python 3.8+
*   PyTorch 1.10+
*   Other dependencies (see `requirements.txt`)

### Installation

```bash
conda create -n Paloma python=3.8
conda activate Paloma
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Training scripts
The experimental results can be reproduced using the following scripts:

```bash
bash ./scripts/ETTm1.sh
bash ./scripts/ETTh1.sh
```




