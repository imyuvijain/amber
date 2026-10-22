from typer.testing import CliRunner

from amber.cli import app


def test_cli_help_lists_usage():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0 and "Usage" in result.output
