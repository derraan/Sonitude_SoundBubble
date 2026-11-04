"""CLI: python -m tools.measure …"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.measure")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("trial", help="Run one algorithm trial (ONNX RT + stimulus [+ HATS])")
    t.add_argument("--config-json", type=Path, help="MeasureConfig / session fields as JSON")
    t.add_argument("--acoustic-profile", choices=["windows_pico", "pi_hats"], default="windows_pico")
    t.add_argument("--host", default="windows_testbench")
    t.add_argument("--measure-root", type=Path, default=REPO_ROOT / "measure")
    t.add_argument("--run-id", default="")
    t.add_argument("--stimulus", type=Path, required=False)
    t.add_argument("--stimulus-out-device", type=int, required=False)
    t.add_argument("--pipeline-in-device", type=int, default=None)
    t.add_argument("--pipeline-out-device", type=int, default=None)
    t.add_argument("--hats-in-device", type=int, default=None)
    t.add_argument("--runtime", choices=["cpp", "python"], default="cpp")
    t.add_argument("--zoo-dir", type=str, default=str(REPO_ROOT / "edge" / "zoo"))
    t.add_argument("--bubble-radius", type=float, default=1.5)
    t.add_argument("--model", type=str, default="")
    t.add_argument("--contract-path", type=str, default="")
    t.add_argument("--soundbubble-rt", type=str, default="")
    t.add_argument("--input-gain-db", type=float, default=0.0)
    t.add_argument("--output-gain-db", type=float, default=0.0)
    t.add_argument("--mix-dry", type=float, default=0.0)
    t.add_argument("--distance-m", type=float, default=1.5)
    t.add_argument("--azimuth-deg", type=float, default=0.0)
    t.add_argument("--elevation-deg", type=float, default=0.0)
    t.add_argument("--channel-map", type=str, default="0,1,2,3,4,5")

    p = sub.add_parser("parity", help="Offline cpp vs python wet diff → TrialStore kind=parity")
    p.add_argument("--raw-8ch", type=Path, required=True)
    p.add_argument("--measure-root", type=Path, default=REPO_ROOT / "measure")
    p.add_argument("--run-id", default=None)
    p.add_argument("--out-dir", type=Path, default=None, help="Legacy direct folder (skips TrialStore ids)")
    p.add_argument("--zoo-dir", type=str, default=str(REPO_ROOT / "edge" / "zoo"))
    p.add_argument("--bubble-radius", type=float, default=1.5)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--contract-path", type=str, default="")
    p.add_argument("--soundbubble-rt", type=str, default="")
    p.add_argument("--input-gain-db", type=float, default=0.0)
    p.add_argument("--output-gain-db", type=float, default=0.0)
    p.add_argument("--mix-dry", type=float, default=0.0)

    a = sub.add_parser("analyze", help="Suppression / polar / FR from taps (+ HATS)")
    a.add_argument("--measure-root", type=Path, required=True)
    a.add_argument("--run-id", default=None)
    a.add_argument("--trial", type=Path, default=None)
    a.add_argument("--acoustic-profile", default=None)
    a.add_argument("--host", default=None)
    a.add_argument("--runtime", default=None)
    a.add_argument("--kind", default="algorithm")
    a.add_argument("--out-dir", type=Path, default=None)
    a.add_argument("--out-csv", type=Path, default=None)
    a.add_argument("--no-plots", action="store_true")

    rp = sub.add_parser("raw-polar", help="Hardware-only stimulus play + Pico multi-ch capture")
    rp.add_argument("--acoustic-profile", choices=["windows_pico", "pi_hats"], default="windows_pico")
    rp.add_argument("--host", default="windows_testbench")
    rp.add_argument("--measure-root", type=Path, default=REPO_ROOT / "measure")
    rp.add_argument("--run-id", default="")
    rp.add_argument("--stimulus", type=Path, required=True)
    rp.add_argument("--stimulus-out-device", type=int, required=True)
    rp.add_argument("--pipeline-in-device", type=int, required=True)
    rp.add_argument("--input-sr", type=int, default=24000)
    rp.add_argument("--input-channels", type=int, default=8)
    rp.add_argument("--distance-m", type=float, default=1.5)
    rp.add_argument("--azimuth-deg", type=float, default=0.0)
    rp.add_argument("--elevation-deg", type=float, default=0.0)

    sm = sub.add_parser("smoke", help="Validate argv build without opening audio devices")
    sm.add_argument("--runtime", choices=["cpp", "python"], default="python")
    sm.add_argument("--zoo-dir", type=str, default=str(REPO_ROOT / "edge" / "zoo"))
    sm.add_argument("--bubble-radius", type=float, default=1.5)
    sm.add_argument("--soundbubble-rt", type=str, default="")

    args = parser.parse_args(argv)

    if args.cmd == "parity":
        from tools.measure.parity import main as parity_main

        av = ["--raw-8ch", str(args.raw_8ch), "--zoo-dir", args.zoo_dir, "--bubble-radius", str(args.bubble_radius)]
        if args.out_dir:
            av += ["--out-dir", str(args.out_dir)]
        else:
            av += ["--measure-root", str(args.measure_root)]
        if args.run_id:
            av += ["--run-id", args.run_id]
        if args.model:
            av += ["--model", args.model]
        if args.contract_path:
            av += ["--contract-path", args.contract_path]
        if args.soundbubble_rt:
            av += ["--soundbubble-rt", args.soundbubble_rt]
        av += [
            "--input-gain-db",
            str(args.input_gain_db),
            "--output-gain-db",
            str(args.output_gain_db),
            "--mix-dry",
            str(args.mix_dry),
        ]
        return parity_main(av)

    if args.cmd == "analyze":
        from tools.measure.analyze import main as analyze_main

        av = ["--measure-root", str(args.measure_root)]
        if args.run_id:
            av += ["--run-id", args.run_id]
        if args.trial:
            av += ["--trial", str(args.trial)]
        if args.acoustic_profile:
            av += ["--acoustic-profile", args.acoustic_profile]
        if args.host:
            av += ["--host", args.host]
        if args.runtime:
            av += ["--runtime", args.runtime]
        if args.kind:
            av += ["--kind", args.kind]
        if args.out_dir:
            av += ["--out-dir", str(args.out_dir)]
        if args.out_csv:
            # Compat: treat as aggregate out-dir parent hint via analyze_run out_dir
            av += ["--out-dir", str(Path(args.out_csv).parent)]
        if args.no_plots:
            av += ["--no-plots"]
        return analyze_main(av)

    if args.cmd == "smoke":
        from tools.measure.runtime import RuntimeLaunch

        launch = RuntimeLaunch(
            runtime=args.runtime,
            pipeline_in_device=0,
            pipeline_out_device=1,
            zoo_dir=args.zoo_dir,
            bubble_radius_m=args.bubble_radius,
            use_zoo=True,
            soundbubble_rt_path=args.soundbubble_rt,
            tap_dir=str(REPO_ROOT / "measure" / "_smoke_tap"),
            channel_map="",
        )
        try:
            argv_built = launch.build_argv()
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 1
        print(json.dumps({"ok": True, "argv": argv_built}, indent=2))
        return 0

    if args.cmd == "raw-polar":
        from tools.measure.core import MeasureConfig, TrialGeometry, run_raw_polar_trial
        from tools.measure.runtime import RuntimeLaunch

        cfg = MeasureConfig(
            acoustic_profile=args.acoustic_profile,
            host=args.host,
            measure_root=str(args.measure_root),
            run_id=args.run_id,
            kind="raw_polar",
            stimulus_path=str(args.stimulus),
            stimulus_out_device=args.stimulus_out_device,
            geometry=TrialGeometry(
                distance_m=args.distance_m,
                azimuth_deg=args.azimuth_deg,
                elevation_deg=args.elevation_deg,
            ),
            runtime=RuntimeLaunch(
                runtime="python",
                pipeline_in_device=args.pipeline_in_device,
                input_sr=args.input_sr,
                input_channels=args.input_channels,
                use_zoo=False,
            ),
        )
        doc = run_raw_polar_trial(cfg)
        print(
            json.dumps(
                {"trial_id": doc.get("trial_id"), "run_id": doc.get("run_id"), "error": doc.get("error")},
                indent=2,
            )
        )
        return 1 if doc.get("error") else 0

    # trial
    from tools.measure.core import MeasureConfig, TrialGeometry, config_from_session_dict, run_algorithm_trial
    from tools.measure.runtime import Gains, RuntimeLaunch

    if args.config_json:
        raw = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        cfg = config_from_session_dict(raw)
    else:
        if args.stimulus is None or args.stimulus_out_device is None:
            parser.error("trial requires --stimulus and --stimulus-out-device (or --config-json)")
        gains = Gains(
            input_gain_db=args.input_gain_db,
            output_gain_db=args.output_gain_db,
            mix_dry=args.mix_dry,
        )
        # Keep empty channel_map out of argv unless the operator set one (mic-index canon).
        cmap = args.channel_map.strip()
        if cmap in ("0,1,2,3,4,5",):
            # CLI default is identity for convenience; still pass it for Class B day-1.
            pass
        rt = RuntimeLaunch(
            runtime=args.runtime,
            pipeline_in_device=args.pipeline_in_device,
            pipeline_out_device=args.pipeline_out_device,
            channel_map=cmap,
            gains=gains,
            zoo_dir=args.zoo_dir if not args.model else "",
            bubble_radius_m=args.bubble_radius,
            use_zoo=not bool(args.model),
            model_path=args.model,
            contract_path=args.contract_path,
            soundbubble_rt_path=args.soundbubble_rt,
        )
        cfg = MeasureConfig(
            acoustic_profile=args.acoustic_profile,
            host=args.host,
            measure_root=str(args.measure_root),
            run_id=args.run_id,
            stimulus_path=str(args.stimulus),
            stimulus_out_device=args.stimulus_out_device,
            hats_in_device=args.hats_in_device,
            geometry=TrialGeometry(
                distance_m=args.distance_m,
                azimuth_deg=args.azimuth_deg,
                elevation_deg=args.elevation_deg,
            ),
            runtime=rt,
        )
    doc = run_algorithm_trial(cfg)
    print(json.dumps({"trial_id": doc.get("trial_id"), "run_id": doc.get("run_id"), "error": doc.get("error")}, indent=2))
    return 1 if doc.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
