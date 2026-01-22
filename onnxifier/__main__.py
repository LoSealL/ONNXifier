"""
Copyright (C) 2025 The ONNXIFIER Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=missing-function-docstring

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from . import ONNXIFIER_OPSET, PassManager, convert_graph
from .checker import check_accuracy
from .logger import set_level

USAGE = "onnxify input_model.onnx [output_model.onnx]"


def parse_args():
    parser = argparse.ArgumentParser(
        prog="onnxify",
        usage=USAGE,
        description="onnxify command-line api",
    )
    parser.add_argument(
        "-a",
        "--activate",
        nargs="*",
        help="select passes to be activated, activate L1, L2 and L3 passes if not set.",
    )
    parser.add_argument(
        "-r",
        "--remove",
        nargs="*",
        help="specify passes to be removed from activated passes.",
    )
    parser.add_argument(
        "-n",
        "--no-passes",
        action="store_true",
        help="do not run any optimizing passes, just convert the model",
    )
    parser.add_argument(
        "--print",
        nargs="?",
        const="all",
        help="print the name of all optimizing passes",
    )
    parser.add_argument(
        "--format",
        choices=("protobuf", "textproto", "json", "onnxtxt"),
        default=None,
        help="onnx file format",
    )
    parser.add_argument(
        "-s",
        "--infer-shapes",
        action="store_true",
        help="infer model shapes",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="specify a json-format config file for passes",
    )
    parser.add_argument(
        "-u",
        "--uncheck",
        action="store_false",
        help="no checking output model",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check optimized model with random inputs",
    )
    parser.add_argument(
        "--checker-backend",
        choices=("onnx", "openvino", "onnxruntime"),
        default="onnxruntime",
        help="backend for accuracy checking, defaults to openvino",
    )
    parser.add_argument(
        "-v",
        "--opset-version",
        type=int,
        help=f"target opset version, defaults to {ONNXIFIER_OPSET.version}",
    )
    parser.add_argument(
        "-vv",
        "--log-level",
        nargs="?",
        const="DEBUG",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="specify the level of log messages to be printed, defaults to INFO",
    )
    parser.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="recursively optimize nested functions",
    )
    parser.add_argument(
        "--nodes",
        nargs="*",
        help="specify a set of node names to apply passes only on these nodes",
    )

    return parser.parse_known_args()


def parse_unknown_args(args: Sequence[str]) -> tuple[str, str, dict, list[str]]:
    """Parse unknown args as pass inputs.

    input_model [output_name] [--key value] ...

    Return:
        str: input model name
        str: output model name
        dict: pass inputs
        list[str]: invalid args
    """

    input_name = ""
    output_name = ""
    configs: dict = {}
    invalid_args: list[str] = []
    value = None
    for i, arg in enumerate(args):
        if arg.startswith("--"):
            if "=" not in arg:
                key = arg[2:]
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    value = args[i + 1]
                else:
                    configs[key] = True
                    continue
            else:
                key, value = arg[2:].split("=", 1)
            if key in configs:
                if isinstance(configs[key], list):
                    configs[key].append(value)
                else:
                    configs[key] = [configs[key], value]
            else:
                configs[key] = value
        elif value is not None:
            value = None
            continue
        elif input_name == "":
            input_name = arg
        elif output_name == "":
            output_name = arg
        else:
            invalid_args.append(arg)
    domain_configs: dict[str, dict] = defaultdict(dict)
    for k, v in configs.items():
        if ":" in k:
            domain, key = k.split(":", 1)
        else:
            domain = ""
            key = k
        if domain not in domain_configs:
            domain_configs[domain] = {}
        try:
            domain_configs[domain][key] = json.loads(v)
        except Exception:  # pylint: disable=broad-except
            print(f'[W] use "{v}" directly as it is not a json string')
            domain_configs[domain][key] = v
    return input_name, output_name, domain_configs, invalid_args


def read_configs_from_args_or_file(args, argv) -> tuple[Path, Path, Optional[dict]]:
    """Read configuration from command-line arguments or a config file.

    Args:
        args: Parsed command-line arguments.
        argv: Remaining command-line arguments.

    Returns:
        Path: The path to the input model.
        Path: The path to the output model.
        Optional[dict]: A dictionary containing the configuration, or None if
            no configuration is found.
    """
    configs = None
    input_url, output_url, configs, errors = parse_unknown_args(argv)
    if not input_url:
        print("Usage: " + USAGE)
        raise RuntimeError("missing input model")
    if errors:
        print("Usage: " + USAGE)
        raise RuntimeError(f"invalid args: {' '.join(errors)}")
    input_model = Path(input_url).expanduser()
    if not output_url:
        output_url = input_model.stem + "_o2o"
    output_model = Path(output_url).expanduser()

    if args.activate and len(args.activate) == 1 and "" in configs:
        configs[args.activate[0]].update(configs.pop(""))

    if args.config_file:
        if configs:
            print("[W] configs from command line will be ignored")
        config_file = Path(args.config_file).expanduser()
        with open(config_file, encoding="utf-8") as file:
            configs = json.load(file)
    return input_model, output_model, configs


def main():
    args, argv = parse_args()
    if args.log_level:
        set_level(args.log_level)

    if args.print:
        match args.print:
            case "all":
                PassManager.print_all()
            case "l1":
                PassManager.print_l1()
            case "l2":
                PassManager.print_l2()
            case "l3":
                PassManager.print_l3()
            case _:
                PassManager.print(args.print)
        exit(0)
    try:
        input_model, output_model, configs = read_configs_from_args_or_file(args, argv)
    except RuntimeError as ex:
        print(f"[E] {ex}")
        exit(1)
    # TODO: need to check configs?

    graph = convert_graph(
        model=input_model,
        passes=[] if args.no_passes else args.activate,
        exclude=args.remove,
        onnx_format=args.format,
        configs=configs,
        target_opset=args.opset_version,
        recursive=args.recursive,
        specify_node_names=args.nodes,
    )
    output_model = graph.save(
        output_model,
        format=args.format,
        infer_shapes=args.infer_shapes,
        check=args.uncheck,
    )
    print(f"model saved to {output_model}")
    if args.check and isinstance(output_model, Path):
        error_maps = check_accuracy(
            input_model,
            output_model,
            backend=args.checker_backend,
        )
        for k, v in error_maps.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
