# Export
Exporting trained `Aframe` models to accelerated TensorRT executables for inference

## Environment
The `export` project is manged by `uv`.

In the root of the `export` project, run 
```bash
apptainer build $AFRAME_CONTAINER_ROOT/export.sif apptainer.def
```
to build the `export` container.

This project can also be installed as a `uv` environment with

```
uv sync
```
