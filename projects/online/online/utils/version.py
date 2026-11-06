import os

# These variables are set in the image by the CI, so any search should use
# an image built by the CI, not one built by `build_containers`.
PIPELINE_VERSION = (
    f"{os.environ.get('AFRAME_VERSION', 'unknown version')} "
    f"(commit {os.environ.get('AFRAME_COMMIT', 'unknown')})"
)
