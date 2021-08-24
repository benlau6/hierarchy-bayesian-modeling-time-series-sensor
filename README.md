# hierarchy-bayesian-modeling-time-series-sensor

## Environment create
```
conda create -c msys2 -c conda-forge -n pm python=3.8 libpython mkl-service m2w64-toolchain numba python-graphviz scipy ipykernel
pip install pymc3 pydot-ng
python -m ipykernel install --user --name=pm
```

## Model
1. hierarchy bayesian student-t model for regression and change point detection
2. gaussian process for regression

## License

This project is licensed under the terms of the MIT license.
