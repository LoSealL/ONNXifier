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

import json
from pathlib import Path
from unittest.mock import ANY, MagicMock, mock_open, patch

import pytest

from onnxifier.__main__ import main, parse_args, parse_unknown_args


class TestParseArgs:
    """Test parse_args function"""

    def test_parse_args_basic(self):
        """Test basic argument parsing"""
        with patch("sys.argv", ["onnxifier", "model.onnx"]):
            _, argv = parse_args()
            assert argv == ["model.onnx"]

    def test_parse_args_with_all_options(self):
        """Test parsing with all command line options"""
        test_args = [
            "onnxifier",
            "input.onnx",
            "-a",
            "pass1",
            "pass2",
            "-r",
            "pass3",
            "-n",
            "--print",
            "all",
            "--format",
            "protobuf",
            "-s",
            "-c",
            "config.json",
            "-u",
            "--check",
            "--checker-backend",
            "onnxruntime",
            "-v",
            "18",
            "-vv",
            "DEBUG",
        ]

        with patch("sys.argv", test_args):
            args, argv = parse_args()
            assert args.activate == ["pass1", "pass2"]
            assert args.remove == ["pass3"]
            assert args.no_passes is True
            assert args.print == "all"
            assert args.format == "protobuf"
            assert args.infer_shapes is True
            assert args.config_file == "config.json"
            assert args.uncheck is False
            assert args.check is True
            assert args.checker_backend == "onnxruntime"
            assert args.opset_version == 18
            assert args.log_level == "DEBUG"
            assert argv == ["input.onnx"]

    def test_parse_args_print_without_value(self):
        """Test --print flag without value"""
        with patch("sys.argv", ["onnxifier", "--print"]):
            args, _argv = parse_args()
            assert args.print == "all"

    def test_parse_args_log_level_without_value(self):
        """Test -vv flag without value"""
        with patch("sys.argv", ["onnxifier", "model.onnx", "-vv"]):
            args, argv = parse_args()
            assert args.log_level == "DEBUG"
            assert argv == ["model.onnx"]

    def test_parse_args_with_unknown_args(self):
        """Test parsing with unknown arguments"""
        test_argv = ["onnxifier", "model.onnx", "output.onnx", "--custom-arg", "value"]
        with patch("sys.argv", test_argv):
            _args, argv = parse_args()
            assert argv == ["model.onnx", "output.onnx", "--custom-arg", "value"]


class TestParseUnknownArgs:
    """Test parse_unknown_args function"""

    def test_parse_unknown_args_output_model_only(self):
        """Test parsing with only output model name"""
        args = ["output.onnx"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == "output.onnx"
        assert output_name == ""
        assert configs == {}
        assert invalid_args == []

    def test_parse_unknown_args_with_configs(self):
        """Test parsing with configuration arguments"""
        args = ["--key1=value1", "--key2=value2", "--domain:key3=value3"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == ""
        assert output_name == ""
        assert configs[""]["key1"] == "value1"
        assert configs[""]["key2"] == "value2"
        assert configs["domain"]["key3"] == "value3"
        assert invalid_args == []

    def test_parse_unknown_args_with_model_and_configs(self):
        """Test parsing with both model name and configurations"""
        args = ["output.onnx", "--config1", "val1", "--config2=val2"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == "output.onnx"
        assert output_name == ""
        assert configs[""]["config1"] == "val1"
        assert configs[""]["config2"] == "val2"
        assert invalid_args == []

    def test_parse_unknown_args_multiple_values_same_key(self):
        """Test parsing with multiple values for the same key"""
        args = ["--key=value1", "--key=value2"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == ""
        assert output_name == ""
        assert configs[""]["key"] == ["value1", "value2"]
        assert invalid_args == []

    def test_parse_unknown_args_boolean_flag(self):
        """Test parsing boolean flags"""
        args = ["--flag1", "--flag2=false"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == ""
        assert output_name == ""
        assert configs[""]["flag1"] is True
        assert configs[""]["flag2"] is False
        assert invalid_args == []

    def test_parse_unknown_args_with_input_and_output(self):
        """Test parsing with input and output models"""
        args = ["input.onnx", "output.onnx"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == "input.onnx"
        assert output_name == "output.onnx"
        assert configs == {}
        assert invalid_args == []

    def test_parse_unknown_args_incomplete_flag(self):
        """Test parsing with single flag at end"""
        args = ["--flag"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == ""
        assert output_name == ""
        assert configs[""]["flag"] is True
        assert invalid_args == []

    def test_parse_unknown_args_config_value_to_list(self):
        """Test parsing where config value becomes a list"""
        args = ["--key=value1", "--key=value2", "--key=value3"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == ""
        assert output_name == ""
        assert configs[""]["key"] == ["value1", "value2", "value3"]
        assert invalid_args == []

    def test_parse_unknown_args_with_invalid_args(self):
        """Test parsing with invalid arguments (more than 2 non-config args)"""
        args = ["input.onnx", "output.onnx", "extra1.onnx", "extra2.onnx"]
        input_name, output_name, configs, invalid_args = parse_unknown_args(args)
        assert input_name == "input.onnx"
        assert output_name == "output.onnx"
        assert configs == {}
        assert invalid_args == ["extra1.onnx", "extra2.onnx"]


class TestMain:
    """Test main function"""

    @patch("onnxifier.__main__.PassManager")
    def test_main_print_all_passes(self, mock_pass_manager):
        """Test main function with --print all"""
        with patch("sys.argv", ["onnxifier", "--print", "all"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_pass_manager.print_all.assert_called_once()

    @patch("onnxifier.__main__.PassManager")
    def test_main_print_l1_passes(self, mock_pass_manager):
        """Test main function with --print l1"""
        with patch("sys.argv", ["onnxifier", "--print", "l1"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_pass_manager.print_l1.assert_called_once()

    @patch("onnxifier.__main__.PassManager")
    def test_main_print_l2_passes(self, mock_pass_manager):
        """Test main function with --print l2"""
        with patch("sys.argv", ["onnxifier", "--print", "l2"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_pass_manager.print_l2.assert_called_once()

    @patch("onnxifier.__main__.PassManager")
    def test_main_print_l3_passes(self, mock_pass_manager):
        """Test main function with --print l3"""
        with patch("sys.argv", ["onnxifier", "--print", "l3"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_pass_manager.print_l3.assert_called_once()

    @patch("onnxifier.__main__.PassManager")
    def test_main_print_custom_pass(self, mock_pass_manager):
        """Test main function with --print custom pass name"""
        with patch("sys.argv", ["onnxifier", "--print", "custom_pass"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_pass_manager.print.assert_called_once_with("custom_pass")

    @patch("onnxifier.__main__.set_level")
    @patch("onnxifier.__main__.convert_graph")
    def test_main_log_level_setting(self, mock_convert_graph, mock_set_level):
        """Test main function sets log level when specified"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("output.onnx")
        mock_convert_graph.return_value = mock_graph

        with patch("sys.argv", ["onnxifier", "model.onnx", "-vv", "INFO"]):
            main()
            mock_set_level.assert_called_once_with("INFO")

    @patch("onnxifier.__main__.convert_graph")
    @patch("builtins.print")
    def test_main_basic_conversion(self, mock_print, mock_convert_graph):
        """Test basic model conversion"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("model_o2o.onnx")
        mock_convert_graph.return_value = mock_graph

        with patch("sys.argv", ["onnxifier", "model.onnx"]):
            main()

        mock_convert_graph.assert_called_once_with(
            model=Path("model.onnx").expanduser(),
            passes=None,
            exclude=None,
            onnx_format=None,
            configs=ANY,  # defaultdict(dict) when no configs
            target_opset=None,
            recursive=False,
            specify_node_names=None,
        )
        mock_graph.save.assert_called_once_with(
            Path("model_o2o").expanduser(),
            format=None,
            infer_shapes=False,
            check=True,
        )
        mock_print.assert_called_with("model saved to model_o2o.onnx")

    @patch("onnxifier.__main__.convert_graph")
    def test_main_with_output_model(self, mock_convert_graph):
        """Test conversion with specified output model"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("output.onnx")
        mock_convert_graph.return_value = mock_graph

        with patch("sys.argv", ["onnxifier", "input.onnx", "output.onnx"]):
            main()

        mock_convert_graph.assert_called_once_with(
            model=Path("input.onnx").expanduser(),
            passes=None,
            exclude=None,
            onnx_format=None,
            configs=ANY,  # Using ANY to match defaultdict
            target_opset=None,
            recursive=False,
            specify_node_names=None,
        )
        mock_graph.save.assert_called_once_with(
            Path("output.onnx").expanduser(),
            format=None,
            infer_shapes=False,
            check=True,
        )

    @patch("onnxifier.__main__.convert_graph")
    def test_main_with_no_passes(self, mock_convert_graph):
        """Test conversion with --no-passes option"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("output.onnx")
        mock_convert_graph.return_value = mock_graph

        with patch("sys.argv", ["onnxifier", "model.onnx", "-n"]):
            main()

        mock_convert_graph.assert_called_once_with(
            model=Path("model.onnx").expanduser(),
            passes=[],
            exclude=None,
            onnx_format=None,
            configs=ANY,  # defaultdict(dict) when no configs
            target_opset=None,
            recursive=False,
            specify_node_names=None,
        )

    @patch("onnxifier.__main__.convert_graph")
    @patch("builtins.print")
    def test_main_with_activate_and_remove_passes(
        self, _mock_print, mock_convert_graph
    ):
        """Test conversion with activate and remove passes"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("output.onnx")
        mock_convert_graph.return_value = mock_graph

        test_argv = ["onnxifier", "model.onnx", "-a", "pass1", "pass2", "-r", "pass3"]
        with patch("sys.argv", test_argv):
            main()

        mock_convert_graph.assert_called_once_with(
            model=Path("model.onnx").expanduser(),
            passes=["pass1", "pass2"],
            exclude=["pass3"],
            onnx_format=None,
            configs=ANY,  # defaultdict(dict) when no configs
            target_opset=None,
            recursive=False,
            specify_node_names=None,
        )

    @patch("onnxifier.__main__.convert_graph")
    @patch("builtins.print")
    def test_main_with_format_and_infer_shapes(self, _mock_print, mock_convert_graph):
        """Test conversion with format and infer shapes options"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("output.onnx")
        mock_convert_graph.return_value = mock_graph

        test_argv = ["onnxifier", "model.onnx", "--format", "protobuf", "-s"]
        with patch("sys.argv", test_argv):
            main()

        mock_convert_graph.assert_called_once_with(
            model=Path("model.onnx").expanduser(),
            passes=None,
            exclude=None,
            onnx_format="protobuf",
            configs=ANY,  # defaultdict(dict) when no configs
            target_opset=None,
            recursive=False,
            specify_node_names=None,
        )
        mock_graph.save.assert_called_once_with(
            Path("model_o2o").expanduser(),
            format="protobuf",
            infer_shapes=True,
            check=True,
        )

    @patch("onnxifier.__main__.convert_graph")
    @patch("builtins.print")
    def test_main_with_opset_version(self, _mock_print, mock_convert_graph):
        """Test conversion with target opset version"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("output.onnx")
        mock_convert_graph.return_value = mock_graph

        with patch("sys.argv", ["onnxifier", "model.onnx", "-v", "18"]):
            main()

        mock_convert_graph.assert_called_once_with(
            model=Path("model.onnx").expanduser(),
            passes=None,
            exclude=None,
            onnx_format=None,
            configs=ANY,  # defaultdict(dict) when no configs
            target_opset=18,
            recursive=False,
            specify_node_names=None,
        )

    @patch("onnxifier.__main__.convert_graph")
    @patch("builtins.print")
    def test_main_with_uncheck_option(self, _mock_print, mock_convert_graph):
        """Test conversion with --uncheck option"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("output.onnx")
        mock_convert_graph.return_value = mock_graph

        with patch("sys.argv", ["onnxifier", "model.onnx", "-u"]):
            main()

        mock_graph.save.assert_called_once_with(
            Path("model_o2o").expanduser(),
            format=None,
            infer_shapes=False,
            check=False,
        )

    @patch("onnxifier.__main__.convert_graph")
    @patch("onnxifier.__main__.check_accuracy")
    @patch("builtins.print")
    def test_main_with_check_accuracy(
        self, mock_print, mock_check_accuracy, mock_convert_graph
    ):
        """Test conversion with accuracy checking"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = Path("output.onnx")
        mock_convert_graph.return_value = mock_graph
        mock_check_accuracy.return_value = {"error1": 0.001, "error2": 0.002}

        test_argv = [
            "onnxifier",
            "model.onnx",
            "--check",
            "--checker-backend",
            "openvino",
        ]
        with patch("sys.argv", test_argv):
            main()

        mock_check_accuracy.assert_called_once_with(
            Path("model.onnx").expanduser(),
            Path("output.onnx"),
            backend="openvino",
        )
        # Check that error messages are printed
        mock_print.assert_any_call("error1: 0.001")
        mock_print.assert_any_call("error2: 0.002")

    def test_main_with_config_file(self):
        """Test conversion with config file"""
        config_data = {"domain1": {"key1": "value1"}, "domain2": {"key2": "value2"}}

        with (
            patch("sys.argv", ["onnxifier", "model.onnx", "-c", "config.json"]),
            patch("builtins.open", mock_open(read_data=json.dumps(config_data))),
            patch("onnxifier.__main__.convert_graph") as mock_convert_graph,
            patch("builtins.print"),
        ):

            mock_graph = MagicMock()
            mock_graph.save.return_value = Path("output.onnx")
            mock_convert_graph.return_value = mock_graph

            main()

            mock_convert_graph.assert_called_once_with(
                model=Path("model.onnx").expanduser(),
                passes=None,
                exclude=None,
                onnx_format=None,
                configs=config_data,
                target_opset=None,
                recursive=False,
                specify_node_names=None,
            )

    def test_main_unknown_argument_error(self):
        """Test main function with unknown arguments raises error"""
        with (
            patch("sys.argv", ["onnxifier", "model.onnx", "invalid1", "invalid2"]),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

    def test_main_multiple_unknown_arguments_error(self):
        """Test main function with multiple unknown arguments raises error"""
        test_argv = ["onnxifier", "input.onnx", "output.onnx", "invalid1", "invalid2"]
        with (
            patch("sys.argv", test_argv),
            pytest.raises(SystemExit, match="1"),
        ):
            main()

    def test_main_activate_single_pass_with_config_merge(self):
        """Test main function merges configs for single activated pass"""
        test_argv = ["onnxifier", "model.onnx", "-a", "pass1", "--key", "value"]
        with (
            patch("sys.argv", test_argv),
            patch("onnxifier.__main__.convert_graph") as mock_convert_graph,
            patch("builtins.print"),
        ):

            mock_graph = MagicMock()
            mock_graph.save.return_value = Path("output.onnx")
            mock_convert_graph.return_value = mock_graph

            main()

            # Check that the config was merged into the pass1 domain
            expected_configs = {"pass1": {"key": "value"}}
            mock_convert_graph.assert_called_once_with(
                model=Path("model.onnx").expanduser(),
                passes=["pass1"],
                exclude=None,
                onnx_format=None,
                configs=expected_configs,
                target_opset=None,
                recursive=False,
                specify_node_names=None,
            )

    @patch("onnxifier.__main__.convert_graph")
    @patch("onnxifier.__main__.check_accuracy")
    @patch("builtins.print")
    def test_main_check_accuracy_with_non_path_output(
        self, _mock_print, mock_check_accuracy, mock_convert_graph
    ):
        """Test that accuracy check is skipped when output is not a Path"""
        mock_graph = MagicMock()
        mock_graph.save.return_value = "output.onnx"  # String instead of Path
        mock_convert_graph.return_value = mock_graph

        with patch("sys.argv", ["onnxifier", "model.onnx", "--check"]):
            main()

        # check_accuracy should not be called when output is not a Path
        mock_check_accuracy.assert_not_called()

    def test_main_handles_path_expansion(self):
        """Test that main function properly expands user paths"""
        with (
            patch("sys.argv", ["onnxifier", "~/model.onnx", "~/output.onnx"]),
            patch("onnxifier.__main__.convert_graph") as mock_convert_graph,
            patch("builtins.print"),
        ):

            mock_graph = MagicMock()
            mock_graph.save.return_value = Path("output.onnx")
            mock_convert_graph.return_value = mock_graph

            main()

            # Verify that expanduser() was called on the paths
            _args, kwargs = mock_convert_graph.call_args
            assert kwargs["model"] == Path("~/model.onnx").expanduser()

            save_args, _save_kwargs = mock_graph.save.call_args
            assert save_args[0] == Path("~/output.onnx").expanduser()

    def test_main_config_no_output_model(self):
        """Test main function with configs but no output model specified"""
        test_argv = ["onnxifier", "input.onnx", "--custom=value"]
        with (
            patch("sys.argv", test_argv),
            patch("onnxifier.__main__.convert_graph") as mock_convert_graph,
            patch("builtins.print"),
        ):

            mock_graph = MagicMock()
            mock_graph.save.return_value = Path("output.onnx")
            mock_convert_graph.return_value = mock_graph

            main()

            # Should use default output name when no model specified but configs exist
            mock_graph.save.assert_called_once()
            save_args = mock_graph.save.call_args[0]
            # The output should be input_o2o since no output model was specified
            assert save_args[0] == Path("input_o2o").expanduser()

    def test_main_recursive(self):
        """Test main function with recursive flag"""
        with (
            patch("sys.argv", ["onnxifier", "model.onnx", "-R"]),
            patch("onnxifier.__main__.convert_graph") as mock_convert_graph,
            patch("builtins.print"),
        ):

            mock_graph = MagicMock()
            mock_graph.save.return_value = Path("output.onnx")
            mock_convert_graph.return_value = mock_graph

            main()

            # Check that the config was merged into the pass1 domain
            mock_convert_graph.assert_called_once_with(
                model=Path("model.onnx").expanduser(),
                passes=None,
                exclude=None,
                onnx_format=None,
                configs=ANY,
                target_opset=None,
                recursive=True,
                specify_node_names=None,
            )

    def test_main_specify_node_names(self):
        """Test main function with specify node names option"""
        test_argv = [
            "onnxifier",
            "model.onnx",
            "--nodes",
            "node1",
            "node2",
            "node3",
        ]
        with (
            patch("sys.argv", test_argv),
            patch("onnxifier.__main__.convert_graph") as mock_convert_graph,
            patch("builtins.print"),
        ):

            mock_graph = MagicMock()
            mock_graph.save.return_value = Path("output.onnx")
            mock_convert_graph.return_value = mock_graph

            main()

            # Check that the specified node names are passed correctly
            mock_convert_graph.assert_called_once_with(
                model=Path("model.onnx").expanduser(),
                passes=None,
                exclude=None,
                onnx_format=None,
                configs=ANY,
                target_opset=None,
                recursive=False,
                specify_node_names=["node1", "node2", "node3"],
            )
