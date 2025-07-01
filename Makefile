.PHONY: build run shell php-run python-run

MIN_R_S ?= 0.35

build:
	docker-compose build --progress=plain

collect: build
	docker-compose run --rm analyzer bash -c "GITHUB_TOKEN=$$GITHUB_TOKEN /app/venv/bin/python /app/collect.py"

shell: build
	docker-compose run --rm analyzer bash

results: build
	docker-compose run --rm analyzer /app/venv/bin/python /app/results.py --min_r_s $(MIN_R_S)
