#!/usr/bin/env python3
"""Build a servable overlay of a Gemma 3 checkpoint for vLLM 0.13.0.

Works around a bug in vLLM 0.13.0's `patch_rope_parameters`
(vllm/transformers_utils/config.py). Under Transformers v4 it does:

    if rope_theta is not None:
        config.rope_parameters["rope_theta"] = rope_theta      # line ~326
    ...
    if set(config.rope_parameters.keys()).issubset(ALLOWED_LAYER_TYPES):
        for layer_params in config.rope_parameters.values():   # line ~349
            patch_rope_parameters_dict(layer_params)
    else:
        patch_rope_parameters_dict(config.rope_parameters)

Transformers 4.57 writes Gemma 3's RoPE per layer type:

    "rope_parameters": {
        "full_attention":    {"rope_type": "linear", "factor": 8.0},
        "sliding_attention": {"rope_type": "default"}
    }

and also writes a sibling `rope_theta`. The injection puts `rope_theta` at the
TOP level of that nested dict, so the subset test fails, the whole nested dict is
treated as one flat parameter block, and validation dies with
"rope_parameters should have a 'rope_type' key" -- before a single weight loads.

Removing the sibling `rope_theta` does not help: it is a class default on
`Gemma3TextConfig`, so vLLM's `getattr` finds 1e6 whether or not the JSON
carries it, and re-injects. Under Transformers v4 that makes vLLM's nested
branch unreachable for Gemma 3 -- the config must be FLAT:

    "rope_parameters": {"rope_type": "linear", "factor": 8.0},
    "rope_scaling":    {"rope_type": "linear", "factor": 8.0}

This is the shape Gemma 3 configs had before Transformers 4.57, and the one
vLLM's Gemma 3 implementation reads. The flat block describes the GLOBAL
(full-attention) layers; the sliding layers take their frequency from
`rope_local_base_freq`, which is a separate field and is left untouched.

Nothing is modified in place: the output directory hardlinks every file of the
source checkpoint except config.json, which is rewritten. Delete the output
directory to revert, and re-run this after upgrading vLLM to check whether the
shim is still needed.

Usage:

    python scripts/vllm_rope_shim.py SRC_CHECKPOINT DEST_DIR
    python scripts/vllm_rope_shim.py /models/merged /models/merged-vllm --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Mirrors transformers.configuration_utils.ALLOWED_LAYER_TYPES, which is what
# vLLM tests the top-level keys against.
ALLOWED_LAYER_TYPES = (
    "full_attention",
    "sliding_attention",
    "chunked_attention",
    "linear_attention",
)

# Gemma 3's sliding-window layers use their own base frequency, kept in a
# separate field rather than in the rope block.
LOCAL_FREQ_FIELD = "rope_local_base_freq"
# The layer type whose rope block becomes the flat one: Gemma 3's global
# attention layers, which are the ones the scaling factor applies to.
GLOBAL_LAYER_TYPE = "full_attention"


def patch_text_config(text_config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return the patched text config and a log of what changed."""
    changes: list[str] = []
    rope_parameters = text_config.get("rope_parameters")

    if not isinstance(rope_parameters, dict):
        raise SystemExit(
            "No dict-valued rope_parameters in the text config; this checkpoint is "
            "not in the nested per-layer form the shim exists for. Serve it directly."
        )

    layer_types = set(rope_parameters) & set(ALLOWED_LAYER_TYPES)
    if not layer_types:
        raise SystemExit(
            f"rope_parameters keys {sorted(rope_parameters)} contain no layer type; "
            "this is already a flat rope block and vLLM handles it."
        )
    if extra := set(rope_parameters) - set(ALLOWED_LAYER_TYPES):
        # Already-injected keys from a previous shim run, or a config shape this
        # script has not seen. Either way, do not guess.
        raise SystemExit(
            f"rope_parameters mixes layer types with other keys {sorted(extra)}. "
            "Refusing to patch; inspect the config by hand."
        )

    global_block = rope_parameters.get(GLOBAL_LAYER_TYPE)
    if not isinstance(global_block, dict) or "rope_type" not in global_block:
        raise SystemExit(
            f"rope_parameters[{GLOBAL_LAYER_TYPE!r}] is missing or has no rope_type; "
            "the source config is not the shape this shim knows how to flatten."
        )

    # Flatten to the pre-4.57 shape. Deleting the sibling rope_theta does NOT
    # work: it is a class default on Gemma3TextConfig, so vLLM's getattr finds
    # 1e6 regardless and re-injects it at the top level of whatever dict is
    # there. Under Transformers v4 that makes vLLM's nested branch unreachable
    # for Gemma 3, so the config itself has to be flat.
    #
    # The flat block describes the GLOBAL (full-attention) layers only, which is
    # how Gemma 3 configs looked before 4.57 and what vLLM's Gemma 3
    # implementation reads. The sliding layers are unaffected: their frequency
    # comes from rope_local_base_freq, a separate field left untouched here.
    flat = dict(global_block)
    text_config["rope_parameters"] = flat
    changes.append(f"rope_parameters flattened to {json.dumps(flat)} (from {GLOBAL_LAYER_TYPE})")

    # Set both spellings. vLLM's v4 branch does `config.rope_parameters =
    # rope_scaling` when rope_scaling is present, so this wins outright; and if
    # Transformers nulls rope_scaling on load, the flat rope_parameters above is
    # already correct. Whichever path runs, the result is the same flat block.
    text_config["rope_scaling"] = dict(flat)
    changes.append("rope_scaling set to the same flat block")

    dropped = sorted(set(rope_parameters) - {GLOBAL_LAYER_TYPE})
    if dropped:
        changes.append(
            f"per-layer entries {dropped} dropped -- sliding layers use "
            f"{LOCAL_FREQ_FIELD}={text_config.get(LOCAL_FREQ_FIELD)}"
        )

    return text_config, changes


def patch_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    # Multimodal Gemma 3 nests the language model under text_config; a text-only
    # export keeps the same fields at the top level.
    if isinstance(config.get("text_config"), dict):
        config["text_config"], changes = patch_text_config(config["text_config"])
    else:
        config, changes = patch_text_config(config)
    return config, changes


def _link_or_copy(src: Path | str, dest: Path | str) -> None:
    """Hardlink a file, resolving symlinks first so cache blobs are linked too."""
    source = Path(src).resolve()
    Path(dest).hardlink_to(source)


def build_overlay(src: Path, dest: Path, config: dict[str, Any], force: bool) -> None:
    if dest.exists():
        if not force:
            raise SystemExit(f"{dest} already exists; pass --force to rebuild it.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for child in sorted(src.iterdir()):
        if child.name == "config.json":
            continue
        target = dest / child.name
        # Hardlinks, not symlinks. The overlay is mounted into a container at a
        # path that has nothing to do with its host path, so an absolute symlink
        # dangles there and a relative one only survives if the source happens to
        # be mounted too. A hardlink is the same inode: it costs no disk, needs
        # no second mount, and cannot dangle.
        #
        # Falls back to an absolute symlink across filesystems (or for the
        # symlinked blobs of a HuggingFace cache), which then does require the
        # source tree to be mounted at the same path inside the container.
        try:
            if child.is_dir():
                shutil.copytree(child, target, copy_function=_link_or_copy)
            else:
                _link_or_copy(child, target)
        except OSError as error:
            print(f"  ! {child.name}: {error}; falling back to a symlink", file=sys.stderr)
            target.symlink_to(child.resolve())

    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="Source checkpoint directory.")
    parser.add_argument("dest", type=Path, help="Overlay directory to create.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the patched rope block and stop."
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing overlay.")
    args = parser.parse_args(argv)

    config_path = args.src / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"No config.json in {args.src}.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config, changes = patch_config(config)

    text_config = config.get("text_config", config)
    rope_block = {key: value for key, value in text_config.items() if "rope" in key}
    print("Patched rope block:")
    print(json.dumps(rope_block, indent=2))
    print("\nChanges:")
    for change in changes:
        print(f"  - {change}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    build_overlay(args.src, args.dest, config, args.force)
    print(f"\nOverlay written to {args.dest} (weights hardlinked, config.json rewritten).")
    print("Serve that path with vLLM. Delete the directory to revert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
