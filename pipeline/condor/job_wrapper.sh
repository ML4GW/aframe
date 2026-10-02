#!/bin/bash
# The executable of every condor job under the ldg-transfer profile. The
# htcondor executor runs it inside the job's image, in the job's scratch
# directory, with the arguments of the snakemake command that runs the job.
#
# Scratch holds the run's copy of the code and the job's input files at
# their paths relative to the run directory, since nothing is shared
# with the submit node.

# $HOME may not exist or be writable, so point it at scratch, where the
# libraries' caches under it (astropy, gwpy, pycbc, matplotlib, triton)
# then go. Torch inductor's default is /tmp/torchinductor_<user>, and looking
# up the user fails on nodes whose user has no passwd entry, so it's set too.
export HOME=$PWD
export TORCHINDUCTOR_CACHE_DIR=$PWD/.cache/torchinductor

# The image installs the projects against /opt/aframe, which holds the code
# from when it was built. PYTHONPATH comes first, so the run's copy wins.
for package in "$PWD"/code/projects/*/ "$PWD"/code/libs/*/; do
    PYTHONPATH=$package${PYTHONPATH:+:$PYTHONPATH}
done
export PYTHONPATH

# Condor returns the whole logs directory, which must exist
mkdir -p logs

python -m snakemake "$@"
status=$?

# When a job fails, snakemake removes its outputs and condor holds a job
# whose outputs are missing, which hides the failure until the executor's
# held-timeout. Create them empty instead, so the job ends with snakemake's
# status at once. Snakemake on the submit node removes them once they're
# back.
if [ $status -ne 0 ]; then
    python - <<'EOF'
import os
from pathlib import Path

# Condor writes the job's attributes to this file, one per line, including
# the files it will send back: TransferOutput = "data/a.hdf5,logs"
for line in Path(os.environ["_CONDOR_JOB_AD"]).read_text().splitlines():
    key, _, value = line.partition(" = ")
    if key == "TransferOutput":
        for output in value.strip('"').split(","):
            output = Path(output)
            if not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                output.touch()
EOF
fi
exit $status
