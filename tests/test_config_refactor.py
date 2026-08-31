import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from config._sandbox_policy import SandboxPolicy
from config._tool_config import ToolConfigs


def write_configs(tmp_path: Path):
    (tmp_path / 'tools.toml').write_text('''
[web-search]
host_url = "http://localhost"

[curl]
permission_timeout = 10

[bash]

[subprocess]

[browser]
''')
    (tmp_path / 'sandbox.toml').write_text('''
mode = "ONLY_SOURCE"
extra_readonly_binds = ["~/test"]
''')


def test_homogenous_load(tmp_path):
    write_configs(tmp_path)
    tools = ToolConfigs.load(tmp_path)
    sandbox = SandboxPolicy.load(tmp_path)

    assert tools.web_search.host_url == 'http://localhost'
    assert tools.curl.permission_timeout == 10
    assert sandbox.extra_ro_binds[0].name == 'test'
