from pathlib import Path

from citra.config import SandboxPolicy, ToolConfigs


def write_configs(tmp_path: Path):
    (tmp_path / 'tools.toml').write_text('''
[web-search]
host_url = "http://localhost"

[bash]

[subprocess]

[browser]
''')
    (tmp_path / 'sandbox.toml').write_text('''
extra_readonly_binds = ["~/test"]
''')


def test_homogenous_load(tmp_path):
    write_configs(tmp_path)
    tools = ToolConfigs.load(tmp_path)
    sandbox = SandboxPolicy.load(tmp_path)

    assert tools.web_search.host_url == 'http://localhost'
    assert tools.bash.permission_timeout == 30
    assert sandbox.extra_ro_binds[0].name == 'test'
