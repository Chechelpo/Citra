# Citra installation

## Make `start.sh` executable and add Citra to `PATH`

From the Citra project root:

```bash
chmod +x start.sh
```

Create a `citra` command in a directory already on your `PATH`:

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/start.sh" ~/.local/bin/citra
```

Ensure `~/.local/bin` is on your `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

You can then launch Citra from anywhere with:

```bash
citra
```

## Agent Runtime prerequisites

Citra executes tools through Bubblewrap. Install `bwrap` with the system
package manager and verify that user namespaces are permitted by the host:

```bash
bwrap --version
```

Each Citra process creates one disposable
`citra-process-<pid>-<nonce>` directory beneath the configured temporary
workspace (or the system temporary directory). The complete project copy,
shared dependency environment, caches, home, and temporary state live there
until Citra exits. Normal and hard shutdown both remove the directory; a later
startup conservatively removes verified stale roots left by an abnormal exit.

Runtime/tool assets are provisioned copy-first under a hard byte budget and
fall back to explicit read-only binds when their declaration permits it.
`env`, `cache`, and `tmp` limits are soft guardrails for Citra-controlled
allocations, not kernel quotas. See [`config-template.toml`](config-template.toml)
for `[runtime.storage]`, `[runtime.environment]`, override, and cleanup settings.

## Operating modes

Citra asks you to select an operating mode immediately after startup and
before it creates the workspace or sandbox. Enter a displayed number or mode
name. Press Enter without a selection to use the configured default.

Copy [`mode-template.toml`](mode-template.toml) to
`.citra/config/mode.toml` and set the default by name:

```toml
default = "greenfield"
```

The selected mode owns the sandbox level, including whether Citra uses a
disposable project workspace or works directly on the authoritative source.
Operator `[sandbox]` settings in `tools.toml` extend the mode: extra read-only
and writable binds are appended to the mode's binds, and
`global_network_disallow = true` can further restrict network access. The
operator config cannot replace the mode's `SandboxMode` or re-enable network
access denied by a mode.

Durable conversation memory remains independently configurable:

```toml
[memory]
enabled = true
```

Setting `memory.enabled = false` removes durable memory tools and their prompt
instructions while preserving ordinary conversation history.


## OpenSERP setup

Create a persistent Docker Compose service:

```bash
sudo mkdir -p /opt/openserp
cd /opt/openserp
```

Create `/opt/openserp/docker-compose.yml`:

```yaml
services:
  openserp:
    image: karust/openserp:latest
    container_name: openserp

    command:
      - serve
      - -a
      - 0.0.0.0
      - -p
      - "7000"

    ports:
      - "127.0.0.1:7000:7000"

    restart: unless-stopped
```

Start the service:

```bash
cd /opt/openserp
docker compose up -d
```

Verify that it is running:

```bash
docker compose ps
curl http://127.0.0.1:7000/health
```

The health endpoint should return JSON with:

```json
{
  "status": "healthy"
}
```

Ensure Docker starts automatically at boot:

```bash
sudo systemctl enable --now docker
```

OpenSERP will then restart automatically because the container uses:

```yaml
restart: unless-stopped
```

### Updating OpenSERP

```bash
cd /opt/openserp
docker compose pull
docker compose up -d
```

### Logs

```bash
cd /opt/openserp
docker compose logs -f openserp
```

Citra should be configured to use:

```text
http://127.0.0.1:7000
```

Keep OpenSERP bound to `127.0.0.1` unless it specifically needs to be reachable from another machine.

## Mermaid renderer setup

Citra uses Mermaid CLI (`mmdc`) to render diagrams.

Install it globally:

```bash
sudo npm install -g @mermaid-js/mermaid-cli
mmdc --version
```
Verify the installation:

```bash
mmdc --version
```

If you don't have a chromium headless isntall it:

```bash
npx puppeteer browsers install chrome-headless-shell
```

Test rendering:

```bash
echo 'flowchart LR; A --> B' > /tmp/test.mmd
mmdc -i /tmp/test.mmd -o /tmp/test.svg
```

If the SVG is created successfully, Citra diagram rendering is ready.

Update Mermaid CLI with:

```bash
sudo npm install -g @mermaid-js/mermaid-cli@latest
```
