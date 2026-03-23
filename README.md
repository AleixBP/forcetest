### 1. Create base environment with FEniCS

```bash
conda create -n fenimage -c conda-forge fenics==2019.1.0 matplotlib scipy jupyter mshr pip
conda activate fenimage
pip install hippylib==3.1.0
conda install scikit-image
```

## 2. Run

Find example run in multiscale_algorithms.py
