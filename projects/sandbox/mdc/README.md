To generate ML MDC dataset, start by cloning [this fork](https://github.com/alecgunny/ml-mock-data-challenge-1)

```
git clone git@github.com:alecgunny/ml-mock-data-challenge-1.git
git checkout aframe
```

Then build the conda environemnt

```
conda env create -f environment.yaml
```

Then run the following script to generate the datasets (presuming that `DATA_DIR` has the same meaning it does when you run the `sandbox` project here):

```
# skip the first 10 days since we're training on those
conda run -n ml-mdc python generate_data.py \
    -d 4 \
    -i $DATA_DIR/injections.hdf \
    -f $DATA_DIR/foreground.hdf \
    -b $DATA_DIR/background.hdf \
    -s 42 \
    --start-offset $((3600 * 24 * 10)) \
    --duration $((3600 * 24 * 30)) \
    --verbose
```
