# Prebuilt wheels

Drop-off point for binary wheels that cannot be built inside the training image
because they need a CUDA toolkit that `python:3.12-slim` does not carry.

Currently this means exactly one package: **FlashAttention 3**
(`flash_attn_3-3.0.0b1-cp312-cp312-linux_x86_64.whl`).

The preferred path downloads a checksum-pinned community wheel for
the project's exact Linux x86-64 / CUDA 12.8 / Torch 2.8.0 ABI:

```bash
scripts/fetch_flash_attn3_wheel.sh
```

The download is resumable and its SHA-256 is verified before it is moved into
this directory. It comes from the community-maintained
`windreamer/flash-attention3-wheels` project, not an official Dao-AILab
release. The filename records upstream source commit `c8abdd`; its pinned
SHA-256 is `6a00f0ba0fd063f228809260c64225cd83aa43ee1c99f7d718f8f104e7e2fd86`.
To avoid that third-party binary or to change any ABI pin, compile the official
pinned source instead:

```bash
scripts/build_flash_attn3_wheel.sh
```

The build keeps downloads, its source checkout, and completed Ninja object
files under `.cache/flash-attn3/`. If compilation fails, rerunning the same
command reuses that cache instead of downloading everything and compiling every
successful object again. The latest complete compiler log is also retained as
`latest-build.log` inside the pin-specific cache directory printed by the
script. Use `MAX_JOBS=1` if the compiler is killed because the build host is
short on RAM; the default is 2.

Then build the Hopper-only image variant:

```bash
IMAGE_TAG=cu128-fa3-py312 INSTALL_FLASH_ATTN3=1 docker compose build trainer
```

The wheels themselves are git-ignored — they are hundreds of megabytes of
architecture- and ABI-specific binary. Transfer them to the offline host next to
the image tarball, or rebuild them there if it has a toolkit. See
`docs/OFFLINE_DEPLOYMENT.md` §6.7.

A wheel is only valid for the exact Torch, Python, CUDA and glibc combination it
was compiled against. Rebuild it whenever the Torch pin in `pyproject.toml`
changes; a mismatched wheel installs cleanly and then fails at import with an
undefined-symbol error.

Because the build host is not the Hopper host, verify the resulting image before
exporting it:

```bash
docker run --rm -v "$PWD/scripts:/scripts:ro" \
    translategemma:cu128-fa3-py312 python /scripts/verify_flash_attn3.py
```

That catches glibc and Torch-ABI mismatches without a GPU. The kernel-execution
checks report as `SKIP` there and run for real when the same script is invoked
on the H100.
