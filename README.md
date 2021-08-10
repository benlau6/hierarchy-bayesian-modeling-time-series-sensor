# hierarchy-bayesian-modeling-time-series-sensor

## Environment create
```
conda env create -f pm.yaml
```
or alternatively, run
```
conda create -c msys2 -c conda-forge -n pm python=3.8 mkl-service libpython m2w64-toolchain statsmodels pandas s3fs seaborn boto3 fsspec ipykernel
pip install pymc3 pyarrow
python -m ipykernel install --user --name=pm
```

## Model
1. hierarchy bayesian student-t model for regression and change point detection
2. gaussian process for regression

## License

This project is licensed under the terms of the MIT license.
