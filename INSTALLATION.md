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
