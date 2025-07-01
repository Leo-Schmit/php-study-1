# Makefile Commands

## Quick Start

- **Docker** and **docker-compose** must be installed.
- For the `make collect` command, you **must** set the `GITHUB_TOKEN` environment variable (for GitHub GraphQL access).

Example (Linux/Mac):

```sh
export GITHUB_TOKEN=your_token_here
```

## Main Commands

| Command            | Description                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------     |
| **results**        | Generate statistics from collected data (`results.py`).                                      |
| **collect**        | Run the main analysis script (`collect.py`) inside the container. **Requires GITHUB_TOKEN**. |
| **shell**          | Open an interactive bash shell in the container (for manual work or debugging).              |
| **build**          | Build the Docker image with all required dependencies.                                       |

## Usage

Example command usage:

```sh
export GITHUB_TOKEN=your_token_here   # Required for make collect
make results                          # Generate statistics from results/data.json
make collect                          # Collect the main analysis (requires GITHUB_TOKEN) and create results/data.json
make shell                            # Open a bash shell in the container for debug
make build                            # Build the Docker image
```
