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
