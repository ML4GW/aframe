"""Snakemake rules for pipeline plots and metrics.

Rules:
  sensitive_volume   compute sensitive volume vs. FAR and produce a plot
"""

import json
import os

plots_dir = run_dir / "plots"
plots_log_dir = log_dir / "plots"

PLOTS_CONTAINER = os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), "plots.sif")


rule sensitive_volume:
    """Compute sensitive volume vs. FAR and plot against LVK pipelines."""
    input:
        background=str(infer_dir / "background.hdf5"),
        foreground=str(infer_dir / "foreground.hdf5"),
        rejected_params=str(test_waveforms / "rejected_parameters.hdf5"),
    output:
        sv_data=str(plots_dir / "sensitive_volume.hdf5"),
        sv_plot=str(plots_dir / "sensitive_volume.html"),
    log:
        str(plots_log_dir / "sensitive_volume.log"),
    container:
        PLOTS_CONTAINER
    params:
        ifos=_fmt_list(config["ifos"]),
        mass_combos=json.dumps(config["mass_combos"]),
        source_prior=config["source_prior"],
        output_dir=str(plots_dir),
        dt=config.get("dt") or "null",
    shell:
        "sensitive-volume"
        " --background {input.background}"
        " --foreground {input.foreground}"
        " --rejected_params {input.rejected_params}"
        " --ifos '{params.ifos}'"
        " --mass_combos '{params.mass_combos}'"
        " --source_prior {params.source_prior}"
        " --output_dir {params.output_dir}"
        " --dt {params.dt}"
        " &> {log}"
