# Export
Exporting trained `Aframe` models to accelerated TensorRT executables for inference

## Environment
The `export` project is manged by `poetry`.

In the root of the `export` project, run 
```bash
apptainer build $AFRAME_CONTAINER_ROOT/export.sif apptainer.def
```
to build the `export` container.

This project can also be installed as a `poetry` environment with

```
poetry install
```

## Scripts
TODO: explain preprocessor / snapshotter / whitener
