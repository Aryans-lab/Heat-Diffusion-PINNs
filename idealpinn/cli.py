"""Command line entry point.

    python -m idealpinn list
    python -m idealpinn run <experiment> [--preset quick|paper] [--set key=value ...]
    python -m idealpinn plot <experiment> [--preset quick|paper]
    python -m idealpinn all [--preset quick|paper] [--only theory|neural] [--force]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import time
import traceback

from .experiments import REGISTRY
from .utils import environment, out_dir, save_json

THEORY = ["bias_vs_weight", "misspecification", "uv_divergence"]


def _load(name: str):
    if name not in REGISTRY:
        raise SystemExit(f"unknown experiment '{name}'. Available: {', '.join(REGISTRY)}")
    return importlib.import_module(f"idealpinn.experiments.{name}")


def _run(name: str, preset: str, overrides: list[str], out_base=None) -> None:
    mod = _load(name)
    if preset not in mod.PRESETS:
        raise SystemExit(f"preset '{preset}' not defined for {name}: {list(mod.PRESETS)}")
    cfg = dict(mod.PRESETS[preset])
    for item in overrides or []:
        key, _, val = item.partition("=")
        if key not in cfg:
            raise SystemExit(f"'{key}' is not a config key of {name}: {list(cfg)}")
        try:
            cfg[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            cfg[key] = val
    out = out_dir(name, preset, out_base)
    t0 = time.time()
    print(f"=== {name} [{preset}] -> {out}")
    mod.run(cfg, out)
    save_json(out / "run_info.json", {"experiment": name, "preset": preset, "config": cfg,
                                      "seconds": time.time() - t0, "environment": environment()})
    mod.plot(out)
    print(f"=== {name} done in {time.time() - t0:.0f} s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="idealpinn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for cmd in ("run", "plot"):
        p = sub.add_parser(cmd)
        p.add_argument("experiment")
        p.add_argument("--preset", default="quick")
        p.add_argument("--out", default=None, help="output directory (default results/<exp>/<preset>)")
        if cmd == "run":
            p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                           help="override config entries, e.g. --set n_experiments=1000")
    p = sub.add_parser("all")
    p.add_argument("--preset", default="quick")
    p.add_argument("--only", choices=["theory", "neural"], default=None)
    p.add_argument("--force", action="store_true", help="rerun experiments that already have results")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for name in REGISTRY:
            mod = _load(name)
            doc = (mod.__doc__ or "").strip().splitlines()[0]
            print(f"{name:20s} {doc}   presets: {', '.join(mod.PRESETS)}")
    elif a.cmd == "run":
        _run(a.experiment, a.preset, a.set, a.out)
    elif a.cmd == "plot":
        _load(a.experiment).plot(out_dir(a.experiment, a.preset, a.out))
    else:
        names = REGISTRY if a.only is None else \
            [n for n in REGISTRY if (n in THEORY) == (a.only == "theory")]
        failed = []
        for name in names:
            done = (out_dir(name, a.preset) / "run_info.json").exists()
            if done and not a.force:
                print(f"=== {name} [{a.preset}] already done, skipping (use --force to rerun)")
                continue
            try:
                _run(name, a.preset, [])
            except Exception:                                # keep going; report at the end
                traceback.print_exc()
                failed.append(name)
        print("=== summary:", "all experiments finished" if not failed else f"FAILED: {failed}")
        if failed:
            raise SystemExit(1)

if __name__ == "__main__":
    main()
