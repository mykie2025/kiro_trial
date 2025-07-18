# Neo4j Docker Compose Configuration

This document provides instructions for using Docker Compose to run Neo4j for the Context Persistence Solutions project.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- Docker Compose (included with Docker Desktop)

## Configuration

The Neo4j Docker Compose configuration uses environment variables from the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |
| `NEO4J_BOLT_PORT` | Bolt protocol port | `7687` |
| `NEO4J_HTTP_PORT` | HTTP port for Neo4j Browser | `7474` |
| `NEO4J_HTTPS_PORT` | HTTPS port | `7473` |
| `NEO4J_HEAP_INITIAL` | Initial heap size | `512M` |
| `NEO4J_HEAP_MAX` | Maximum heap size | `1G` |

## Persistent Volumes

The Docker Compose configuration mounts the following volumes for data persistence:

- `./neo4j_data`: Neo4j data directory
- `./neo4j_logs`: Neo4j logs directory
- `./neo4j_import`: Neo4j import directory for data import operations

## Usage

### Starting Neo4j

To start Neo4j in the background:

```bash
docker compose up -d
```

### Checking Neo4j Status

To check if Neo4j is running:

```bash
docker compose ps
```

### Viewing Neo4j Logs

To view the Neo4j logs:

```bash
docker compose logs neo4j
```

To follow the logs in real-time:

```bash
docker compose logs -f neo4j
```

### Stopping Neo4j

To stop Neo4j:

```bash
docker compose down
```

To stop Neo4j and remove all data volumes (WARNING: This will delete all data):

```bash
docker compose down -v
```

## Accessing Neo4j

### Neo4j Browser

Once Neo4j is running, you can access the Neo4j Browser at:

```
http://localhost:7474
```

Login credentials:
- Username: `neo4j` (or the value of `NEO4J_USERNAME` in your `.env` file)
- Password: `password` (or the value of `NEO4J_PASSWORD` in your `.env` file)

### Bolt Connection

Applications can connect to Neo4j using the Bolt protocol at:

```
neo4j://localhost:7687
```

## Integration with Python Code

The Neo4j Docker Compose configuration is compatible with the existing Python code in the project. The `.env` file already contains the necessary configuration for both Docker Compose and the Python code.

## Troubleshooting

### Container Won't Start

If the container won't start, check the logs:

```bash
docker compose logs neo4j
```

### Connection Issues

If you can't connect to Neo4j, make sure:
1. The container is running (`docker compose ps`)
2. The ports are correctly mapped (`docker compose port neo4j 7474` and `docker compose port neo4j 7687`)
3. Your firewall isn't blocking the connections

### Data Persistence Issues

If you're experiencing data persistence issues:
1. Check that the volume directories exist and have proper permissions
2. Verify that the volumes are correctly mounted (`docker compose exec neo4j ls -la /data`)

## Health Check

The Docker Compose configuration includes a health check that verifies Neo4j is running properly. You can check the health status with:

```bash
docker compose ps
```

The health status will be displayed in the "Status" column. 