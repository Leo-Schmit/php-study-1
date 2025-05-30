# Makefile Commands

## Quick Start

* **Docker** and **docker-compose** must be installed.
* For the `make run` command, you **must** set the `GITHUB_TOKEN` environment variable (for GitHub API access).

Example (Linux/Mac):

```sh
export GITHUB_TOKEN=your_token_here
```

## Main Commands

| Command            | Description                                                                               |
| ------------------ | ----------------------------------------------------------------------------------------- |
| **results**         | Generate statistics from collected data (`results.py`).                          |
| **run**            | Run the main analysis script (`run.py`) inside the container. **Requires GITHUB\_TOKEN**. |
| **run-php-parser** | Run the `php-parser.php` script using PHP-Parser for code analysis.                       |
| **run-php-ast**    | Run the `php-ast.php` script using the php-ast extension for code analysis.               |
| **shell**          | Open an interactive bash shell in the container (for manual work or debugging).           |
| **build**          | Build the Docker image with all required dependencies.                                    |

## Usage

Example command usage:

```sh
export GITHUB_TOKEN=your_token_here   # Required for make run
make results                          # Generate statistics from combined_repo_data.json
make run                              # Run the main analysis (requires GITHUB_TOKEN) and create combined_repo_data.json
make run-php-parser                   # Run PHP-Parser example
make run-php-ast                      # Run php-ast example
make shell                            # Open a bash shell in the container for debug
make build                            # Build the Docker image
```
