"""Snakemake rules for pipeline plots and metrics.

Rules:
  fetch_gwtc3_injections   copy the GWTC-3 injection set into the run dir
  fetch_veto_segments      query veto segments for the analyzed span
  sensitive_volume         compute sensitive volume vs. FAR and produce a plot

The fetch rules are localrules, so sensitive_volume reads only declared
inputs and needs no network access.
"""

import json
import os

plots_dir = run_dir / "plots"
plots_log_dir = log_dir / "plots"

PLOTS_CONTAINER = os.path.join(os.getenv("AFRAME_CONTAINER_ROOT", ""), "plots.sif")

VETOS = config.get("vetos")


localrules:
    fetch_gwtc3_injections,
    fetch_veto_segments,


rule fetch_gwtc3_injections:
    """Copy the GWTC-3 sensitivity injection set from Zenodo (via cache)."""
    output:
        str(plots_dir / "gwtc3_injections.hdf5"),
    log:
        str(plots_log_dir / "fetch_gwtc3_injections.log"),
    container:
        PLOTS_CONTAINER
    script:
        "scripts/fetch_gwtc3_injections.py"


rule fetch_veto_segments:
    """Query DQSegDB for the veto segments covering the background."""
    input:
        background=str(infer_dir / "background.hdf5"),
    output:
        str(plots_dir / "veto_segments.json"),
    log:
        str(plots_log_dir / "fetch_veto_segments.log"),
    container:
        PLOTS_CONTAINER
    params:
        vetos=VETOS,
        ifos=config["ifos"],
        segment_server=config["segment_server"],
    script:
        "scripts/fetch_veto_segments.py"


rule sensitive_volume:
    """Compute sensitive volume vs. FAR and plot against LVK pipelines."""
    input:
        **({"veto_segments": str(plots_dir / "veto_segments.json")} if VETOS else {}),
        background=str(infer_dir / "background.hdf5"),
        foreground=str(infer_dir / "foreground.hdf5"),
        rejected_params=str(test_waveforms / "rejected_parameters.hdf5"),
        injection_file=str(plots_dir / "gwtc3_injections.hdf5"),
    output:
        sv_data=str(plots_dir / "sensitive_volume.hdf5"),
        sv_plot=str(plots_dir / "sensitive_volume.html"),
        gwtc3_sv=str(plots_dir / "gwtc-3_pipeline_sv.hdf5"),
    log:
        str(plots_log_dir / "sensitive_volume.log"),
    container:
        PLOTS_CONTAINER
    resources:
        **rule_resources("sensitive_volume"),
    params:
        ifos=_fmt_list(config["ifos"]),
        mass_combos=json.dumps(config["mass_combos"]),
        source_prior=config["source_prior"],
        output_dir=lambda wc, output: str(Path(output.sv_data).parent),
        dt=config.get("dt") or "null",
        # Omitted entirely when unset. If we pass an empty list,
        # we still do a query.
        vetos=lambda wc, input: (
            f" --vetos '{_fmt_list(VETOS)}' --veto_segments {input.veto_segments}"
            if VETOS
            else ""
        ),
    shell:
        "sensitive-volume"
        " --background {input.background}"
        " --foreground {input.foreground}"
        " --rejected_params {input.rejected_params}"
        " --injection_file {input.injection_file}"
        " --ifos '{params.ifos}'"
        " --mass_combos '{params.mass_combos}'"
        " --source_prior {params.source_prior}"
        " --output_dir {params.output_dir}"
        " --dt {params.dt}"
        "{params.vetos}"
        " &> {log}"
