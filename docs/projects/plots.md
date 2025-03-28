# Plots
Producing visualizations of `Aframe` performance

## Environment
The `plots` project environment is manged by `uv`.

In the root of the `plots` project, run 
```bash
apptainer build $AFRAME_CONTAINER_ROOT/plots.sif apptainer.def
```
to build the `plots` container.

This project can also be installed locally via 

```
uv sync
```
