"""Verify a prebuilt FlashAttention 3 wheel inside the image that carries it.

The wheel is compiled on a build host that is deliberately not the machine that
runs it: the build needs a CUDA toolkit and internet, the execution needs a
Hopper GPU and neither. That split means the two most damaging failure modes --
a glibc mismatch and a Torch C++ ABI mismatch -- are invisible until the image
reaches the offline host, where a fix costs a full transfer cycle.

This script separates the checks by what they actually require, so the ones that
can run on the build host do:

    Tier 1 (no GPU needed)   loading the compiled extension. Catches the glibc
                             and Torch-ABI errors, which is most of the risk.
    Tier 2 (Hopper needed)   executing a kernel and comparing it against SDPA.

Run it in both places:

    docker run --rm translategemma:cu128-fa3-py312 \
        python /workspace/scripts/verify_flash_attn3.py       # build host, tier 1
    docker compose run --rm trainer \
        python scripts/verify_flash_attn3.py                  # H100 host, both

Exit status is 0 only if every check that could run passed. Skipped tier-2
checks are reported as SKIP and do not fail the run, so the same invocation is
meaningful on both machines.
"""

from __future__ import annotations

import importlib
import sys
import traceback
from types import ModuleType

# FA3's package name moved: the upstream hopper/ directory historically
# installed `flash_attn_interface`, while the 3.0.0b1 distribution installs
# `flash_attn_3`. Transformers probes for `flash_attn_3`, so that name is the
# one that matters for model.attn_implementation, but accept either here and
# report which was found -- a wheel exposing only the old name explains an
# ImportError from transformers that otherwise looks like a missing install.
CANDIDATE_MODULES = ("flash_attn_3", "flash_attn_interface")

# Hopper. FA3 emits sm_90a kernels only; anything below cannot launch them.
REQUIRED_CAPABILITY = (9, 0)

_failures: list[str] = []
_skips: list[str] = []


def report(status: str, name: str, detail: str = "") -> None:
    line = f"[{status:4}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)


def fail(name: str, detail: str) -> None:
    _failures.append(name)
    report("FAIL", name, detail)


def skip(name: str, detail: str) -> None:
    _skips.append(name)
    report("SKIP", name, detail)


def check_import() -> ModuleType | None:
    """Tier 1. Loads the compiled .so and resolves its symbols against torch."""
    for name in CANDIDATE_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        except OSError as exc:
            # A glibc mismatch surfaces here, not as ImportError:
            #   /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.38' not found
            fail("import", f"{name} present but its extension will not load: {exc}")
            return None
        report("ok", "import", f"{name} from {getattr(module, '__file__', '?')}")
        if name != "flash_attn_3":
            fail(
                "module name",
                f"found {name}, but transformers probes for flash_attn_3; "
                "model.attn_implementation='flash_attention_3' will not find it",
            )
        return module

    fail("import", f"none of {CANDIDATE_MODULES} importable -- wheel not installed")
    return None


def check_symbols(module: ModuleType) -> None:
    """Tier 1. Forces the CUDA extension itself to load.

    Importing the Python package does not always pull in the binary; an
    undefined-symbol error from a Torch ABI mismatch can hide until first use.
    Touching the entry point here makes it surface without a GPU.
    """
    entry = getattr(module, "flash_attn_func", None)
    if entry is None:
        fail("symbols", "module exposes no flash_attn_func")
        return
    report("ok", "symbols", "flash_attn_func resolved")


def check_torch_pairing() -> object | None:
    """Tier 1. The wheel is valid only for the Torch it was compiled against."""
    try:
        import torch
    except ImportError as exc:
        fail("torch", f"torch not importable: {exc}")
        return None
    report(
        "ok",
        "torch",
        f"{torch.__version__}, CUDA {torch.version.cuda}, "
        f"python {sys.version_info.major}.{sys.version_info.minor}",
    )
    return torch


def check_transformers_probe() -> None:
    """Tier 1 where it can be, tier 2 where it cannot.

    This is the exact predicate transformers evaluates when
    model.attn_implementation is 'flash_attention_3'. On some versions it also
    inspects the current device, so a False here on a non-Hopper build host is
    expected and not a failure.
    """
    try:
        from transformers.utils import is_flash_attn_3_available
    except ImportError as exc:
        skip("transformers probe", f"not available in this transformers: {exc}")
        return

    available = bool(is_flash_attn_3_available())
    if available:
        report("ok", "transformers probe", "is_flash_attn_3_available() is True")
        return

    import torch

    if not torch.cuda.is_available():
        skip(
            "transformers probe",
            "False, but no GPU visible -- this probe is device-sensitive; "
            "re-run on the H100 host",
        )
    else:
        fail(
            "transformers probe",
            "is_flash_attn_3_available() is False on a visible GPU -- "
            "model.attn_implementation='flash_attention_3' will raise",
        )


def check_device(torch) -> bool:
    """Gate for tier 2."""
    if not torch.cuda.is_available():
        skip("device", "no CUDA device visible -- tier 2 checks not run here")
        return False

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability < REQUIRED_CAPABILITY:
        skip(
            "device",
            f"{name} is sm_{capability[0]}{capability[1]}; FA3 needs "
            f"sm_{REQUIRED_CAPABILITY[0]}{REQUIRED_CAPABILITY[1]}+ -- keep sdpa here",
        )
        return False

    report("ok", "device", f"{name}, sm_{capability[0]}{capability[1]}")
    return True


def check_kernel(module: ModuleType, torch) -> None:
    """Tier 2. Launches a real kernel and checks it against SDPA.

    A wheel can import cleanly and still have no kernel for this architecture.
    Comparing against SDPA rather than just checking for an exception also
    catches a numerically wrong build, which is otherwise indistinguishable from
    a slightly worse training run.
    """
    batch, seqlen, heads, head_dim = 2, 256, 8, 128
    dtype = torch.bfloat16

    try:
        q, k, v = (
            torch.randn(
                batch, seqlen, heads, head_dim, device="cuda", dtype=dtype
            )
            for _ in range(3)
        )
        out = module.flash_attn_func(q, k, v, causal=True)
        # Some builds return (out, softmax_lse).
        if isinstance(out, tuple):
            out = out[0]
    except RuntimeError as exc:
        # "no kernel image is available for execution on the device" lands here
        # when the wheel was compiled for a different architecture.
        fail("kernel", f"launch failed: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - surface anything the build throws
        fail("kernel", f"unexpected error: {exc}")
        return

    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)

    max_diff = (out.float() - reference.float()).abs().max().item()
    # bf16 attention over 256 positions: agreement is to a few thousandths, not
    # to float tolerance. A wrong build is off by orders of magnitude, so this
    # bound separates the two cases without being flaky.
    if max_diff > 2e-2:
        fail("kernel", f"output disagrees with SDPA, max abs diff {max_diff:.4g}")
    else:
        report("ok", "kernel", f"matches SDPA, max abs diff {max_diff:.4g}")


def main() -> int:
    print("FlashAttention 3 wheel verification\n")

    torch = check_torch_pairing()
    module = check_import()

    if module is not None:
        check_symbols(module)
    if torch is not None:
        check_transformers_probe()

    if module is not None and torch is not None and check_device(torch):
        check_kernel(module, torch)

    print()
    if _failures:
        print(f"FAILED: {', '.join(_failures)}")
        return 1
    if _skips:
        print(
            f"PASSED, with {len(_skips)} check(s) skipped: {', '.join(_skips)}.\n"
            "Re-run on the Hopper host to exercise them."
        )
        return 0
    print("PASSED: all checks ran and succeeded.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - a crash here is itself the finding
        traceback.print_exc()
        sys.exit(1)
