"""Top-level Snakefile

Usage, from the root directory:
    snakemake -n                                   # dry-run
    snakemake --profile pipeline/profiles/condor   # HTCondor
    snakemake --profile pipeline/profiles/local    # local execution (dev)

The pipeline config is loaded from pipeline/config/config.yaml by
default. To use a different config, copy the original, make
modifications, and run:

    snakemake --configfile my_run.yaml --profile pipeline/profiles/condor
"""


configfile: "pipeline/config/config.yaml"


rule all:
    default_target: True
    input:
        [],
