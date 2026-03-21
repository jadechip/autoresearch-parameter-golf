ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
TRAIN_SHARDS ?= 80
RUN_ID ?= baseline_5090_5min
WATCH_PATH ?= ./runs/autoresearch_5090/index/latest.json
RESULTS_TSV ?= ./runs/autoresearch_5090/results.tsv
PACKAGE_TRACK ?= track_non_record_16mb
PACKAGE_NAME ?= Example Submission
PACKAGE_AUTHOR ?= Your Name
PACKAGE_GITHUB_ID ?= your-github-id
PACKAGE_BLURB ?= Replace this with a short summary.

.PHONY: install install-cpu smoke-data smoke-train smoke download-data train-5090 train-h100-1x train-h100-8x eval-h100-1x eval-h100-8x autoresearch-baseline init-autoresearch-session codex-autoresearch-loop watch-codex-loop watch-latest monitor-latest tensorboard-autoresearch compare-autoresearch package-submission

install:
	bash scripts/bootstrap.sh

install-cpu:
	bash scripts/bootstrap.sh --no-tokenizer

smoke-data:
	uv run pgolf-prepare smoke-data --output_dir ./smoke_data

smoke-train:
	uv run pgolf-train --config_json ./smoke_data/smoke_config.json --max_wallclock_seconds 30

smoke: smoke-data smoke-train

download-data:
	TRAIN_SHARDS=$(TRAIN_SHARDS) bash scripts/download_official_fineweb.sh

train-5090:
	bash scripts/runpod_5090_train.sh

train-h100-1x:
	bash scripts/run_h100_1x_train.sh

train-h100-8x:
	bash scripts/run_h100_8x_train.sh

eval-h100-1x:
	bash scripts/run_h100_1x_eval.sh

eval-h100-8x:
	bash scripts/run_h100_8x_eval.sh

autoresearch-baseline:
	RUN_ID=$(RUN_ID) bash scripts/run_autoresearch_experiment.sh

init-autoresearch-session:
	bash scripts/init_autoresearch_session.sh

codex-autoresearch-loop:
	bash scripts/run_codex_autoresearch_loop.sh

watch-codex-loop:
	bash scripts/watch_codex_autoresearch.sh

watch-latest:
	bash scripts/run_tensorboard_autoresearch.sh

monitor-latest:
	bash scripts/run_tensorboard_autoresearch.sh

tensorboard-autoresearch:
	bash scripts/run_tensorboard_autoresearch.sh

compare-autoresearch:
	uv run pgolf-compare-runs --results_tsv $(RESULTS_TSV)

package-submission:
	uv run pgolf-package-submission --train_results_json $(RESULTS_JSON) --track $(PACKAGE_TRACK) --name "$(PACKAGE_NAME)" --author "$(PACKAGE_AUTHOR)" --github_id "$(PACKAGE_GITHUB_ID)" --blurb "$(PACKAGE_BLURB)"
