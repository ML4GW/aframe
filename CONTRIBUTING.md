# Contributing to Aframe

Thanks for taking the time to contribute! While we appreciate all who wish to contribute, at the moment we're prioritizing contributions from IGWN scientists.

Here are some guidelines on how code in this repo should be structured, as well as best practices for making contributions to it.

## Getting started
> **Note** It is assumed that you have succesfully completed the [ml4gw quickstart](https://github.com/ml4gw/quickstart/) instructions for pre-requisite environment setup and software installation.

To start contributing, create a [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of this repository. You can then clone your fork

```console
git clone git@github.com:albert-einstein/aframe.git
```

(replacing your GitHub username for `albert-einstein`). Then, add your the main (`ML4GW`) repository as a remote reference. A common practice is to rename this remote `upstream`.

```console
git remote add upstream git@github.com:ML4GW/aframe.git
```

Next, check out to a new branch for a _specific_ issue you're trying to solve
```console
git checkout -b new-feature
```

Now you're ready to start adding in all your important fixes!

## Contribution guidelines

### Libraries vs. Projects vs. Tasks/Pipelines
Broadly speaking, Aframe consists of three main sub-directories: `projects` , `libs`, and `aframe`. 

## libs
If the code you're writing is some general-purpose function that gets used in many places, consider implementing it as a library in `libs` that other code (typically, a project) can import and call. To make clear what libraries come from this repository, we've adopted the practice that local library imports should start with `aframe.`, e.g. `from aframe.analysis import matched_filter`.

In practice, this means structuring libraries like:
```
| libs/
|    | my-library/
|    |    | aframe/
|    |    |    | my-library/
|    |    |    |    | __init__.py
|    |    |    |    | ...
|    |    | pyproject.toml
|    |    | poetry.lock
```

## projects
Code that produces _artifacts_ of some specific experiment (training data, optimized models, analysis plots, etc.) should be implemented as a project in the `projects` directory. Projects should be kept modular and specific to the artifact they are designed to generate, with light-weight environments.

## tasks/pipelines
The `aframe` sub-directory utilizes [`luigi`](https://luigi.readthedocs.io/en/stable/) and a higher level wrapper [`law`](https://github.com/riga/law) to construct complex, automated, end-to-end pipelines from different projects. `law` introduces environment sandboxing, which allows different tasks to be run in individual environments. These environments can be Apptainer images, or simple python environments. Also, `law` has abstractions for running tasks via [Condor](https://htcondor.readthedocs.io/en/latest/) which makes distributing tasks on HPC clusters trivial. If you think your contribution would benefit from being included in a broader pipeline, consider implementing a `law.Task` wrapper around it. It is recommended that you read the `law` and `luigi` docs linked above to get familiar with these concepts.

### Testing
For any code that you contribute, make sure to add unit tests which explicitly state and validate expectations about the behavior of your code. Tests should be placed in a `tests` subdirectory of each library and project, and should be structured similarly to the library code itself but with `test_` prepended to all the names.

### Pre-commit hooks
To keep the code style consistent and neat throughout the repo, we implement [pre-commit hooks](https://pre-commit.com/) to statically lint and style-check any code that wants to get added to the upstream `main` branch. `pre-commit` is already installed in the root `Aframe` environment, so you can run 

```console
poetry run pre-commit install
```

to install the hooks.

Now any attempts to commit new code will require these tests to past first (and even do some reformatting for you if possible). To run the hooks on existing code, you can run 
```console
poetry run pre-commit run --all
```

### Non-automatable style guidelines
- Annotate function arguments and returns as specifically as possible
- Adopt [Google docstring](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) formatting (this will eventually be used by Sphinx autodoc, so consistency is important)

## Adding your code
Once you've added all the code required to solve the issue you set out for, and have tested it and run it through the pre-commit hooks, you're ready to add it to the upstream repo! To do this, push the branch you've been working on back up to _your_ fork

```console
git push -u origin new-unit-tests
```

Now submit a new [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) from your branch to upstream `main`, describing what your new code does and potentially linking to any issues it addresses. This will kick off CI workflows for unit testing and style checks, as well as a review from your fellow Aframe contributors.

If any of the CI tests fail or your reviewers request changes to the code, you can keep working on your same branch locally and push the relevant fixes and changes up once they're ready, which will kick this process off again. One common issue here is that changes have been made to the `main` branch by another contributor that you don't have in your repo. If this is the case, try pulling the changes into your local branch via `git pull upstream/main`.

Once all the checks pass and at least one reviewer approves your code, your changes will be merged into the main repository for all the world to use!

Occasionally, you may have a branch that is not ready to pass all the checks required for merging, but is still useful to make others aware of the work that is going on (for e.g. Project board planning). If this is the case, consider opening a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/), then updating its status once the code is complete.
