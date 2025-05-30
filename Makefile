.PHONY: build run shell php-run python-run

build:
	docker-compose build --progress=plain

run: build
	docker-compose run --rm analyzer bash -c "GITHUB_TOKEN=$$GITHUB_TOKEN /app/venv/bin/python /app/run.py"

shell: build
	docker-compose run --rm analyzer bash

run-php-parser: build
	docker-compose run --rm analyzer php /app/php-parser.php

run-php-ast: build
	docker-compose run --rm analyzer php /app/php-ast.php

results: build
	docker-compose run --rm analyzer /app/venv/bin/python /app/results.py
