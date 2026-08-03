# Prebuilt wheels

Drop-off point for binary wheels that cannot be built inside the training image
because they need a CUDA toolkit that `python:3.12-slim` does not carry.

Currently this means exactly one package: **FlashAttention 3**
(`flash_attn_3-3.0.0b1-cp312-cp312-linux_x86_64.whl`).

Build it with:

```bash
scripts/build_flash_attn3_wheel.sh
```

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
