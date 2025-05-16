# Online Deployment
Scripts for deploying `Aframe` and `AMPLFI` models for online analysis  

Please see our documentation for more information.

# Connecting to `aframe` node with the shared `aframe` account
If you are someone who has access to the shared `aframe` account on the LDG and need to run the search,
you can connect to the dedicated `aframe` node on the CIT cluster via the following steps:

On your local machine:
```bash
ssh-add <path-to-sshkey>
ssh -A <username>@ldas-grid.ligo.caltech.edu>
```

Once connected:
```bash
ssh aframe@aframe.ldas.cit
```
