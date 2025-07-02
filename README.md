# Makefile Commands

## Quick Start

- **Docker** and **docker-compose** must be installed.
- For the `make collect` command, you **must** set the `GITHUB_TOKEN` environment variable (for GitHub GraphQL access).

Example (Linux/Mac):

```sh
export GITHUB_TOKEN=your_token_here
```

### MIN\_R\_S Parameter

The `MIN_R_S` parameter defines the cutoff value for the correlation coefficient. By default:

- `MIN_R_S` is set to `0.35`.

You can override this default when running the `make results` command:

```sh
make results MIN_R_S=<your_threshold>
```

For example, to include only values with r_s ≥ 0.5:

```sh
make results MIN_R_S=0.5
```


## Main Commands

| Command     | Description                                                                                           |
| ----------- | ----------------------------------------------------------------------------------------------------- |
| **results** | Generate statistics from collected data (`results.py`). Use the `MIN_R_S` parameter to filter output. |
| **collect** | Run the main data collection script (`collect.py`) inside the container. **Requires GITHUB\_TOKEN**.  |
| **shell**   | Open an interactive bash shell in the container (for manual work or debugging).                       |
| **build**   | Build the Docker image with all required dependencies.                                                |

## Usage

Example command usage:

```sh
export GITHUB_TOKEN=your_token_here   # Required for make collect
make results                          # Generate statistics from results/data.json
make collect                          # Collect the main analysis (requires GITHUB_TOKEN) and create results/data.json
make shell                            # Open a bash shell in the container for debug
make build                            # Build the Docker image
```
