# C Implementation of Isolation Forest

High-performance anomaly detection algorithm - Isolation Forest implemented in C

## Features

- Multi-threaded
- Scikit-learn compatibility

## Getting Start

```bash
git clone https://github.com/yourusername/isolation-forest.git
cd isolation-forest
make
```

## Test

```bash
# Generate test data
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-learn pandas
pip install ipykernel

# run on Notebook
gen-data-test-iforest-scikit-learn.ipynb

# test
make test
```
