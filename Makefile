# ane-fim: run the ANE completion server for llama.vscode.
#   make start              start in the background (1.5B LUT6, port 8012), wait until ready
#   make start SIZE=0.5b    smaller/faster model (add MODEL=fp16 for its FP16 fallback build)
#   make stop | restart | status | logs
#   make serve              run in the foreground (Ctrl+C to stop)

ANEMLL_DIR ?= $(HOME)/src/Anemll
PYTHON     ?= $(ANEMLL_DIR)/env-anemll/bin/python
MODEL      ?= lut6
SIZE       ?= 1.5b
PORT       ?= 8012
ARGS       ?=
BASELINE_PORT ?= 8013

LOG_DIR    := eval/logs
LOG        := $(LOG_DIR)/ane-server.log
SERVER     := server-py/server.py
URL        := http://127.0.0.1:$(PORT)

export ANEMLL_DIR
export PYTHONWARNINGS := ignore

.PHONY: help serve start stop restart status logs bench test baseline stop-baseline

help:
	@sed -n '2,6p' Makefile | sed 's/^#//'
	@echo "  make bench | test | baseline | stop-baseline"
	@echo "Variables: MODEL=$(MODEL) SIZE=$(SIZE) PORT=$(PORT) ANEMLL_DIR=$(ANEMLL_DIR) ARGS='$(ARGS)'"

serve:
	$(PYTHON) -u $(SERVER) --model $(MODEL) --size $(SIZE) --port $(PORT) $(ARGS)

start:
	@if curl -sf $(URL)/health > /dev/null 2>&1; then echo "Already running on $(URL)"; exit 0; fi; \
	mkdir -p $(LOG_DIR); \
	nohup $(PYTHON) -u $(SERVER) --model $(MODEL) --size $(SIZE) --port $(PORT) $(ARGS) >> $(LOG) 2>&1 & \
	printf "Starting ane-fim ($(SIZE) $(MODEL)) on $(URL) "; \
	for i in $$(seq 120); do \
		if curl -sf $(URL)/health > /dev/null 2>&1; then echo " ready."; exit 0; fi; \
		printf "."; sleep 1; \
	done; echo " not ready after 120 s, see $(LOG)"; exit 1

stop:
	@pkill -f "$(SERVER) .*--port $(PORT)" && echo "Stopped ane-fim on port $(PORT)" || echo "Not running on port $(PORT)"

restart: stop
	@sleep 1
	@$(MAKE) --no-print-directory start

status:
	@curl -s $(URL)/health 2>/dev/null && echo || echo "Not running on $(URL)"

logs:
	tail -f $(LOG)

# Replay the captured llama.vscode requests (needs eval/llama_vscode_requests/).
bench:
	python3 eval/bench_latency.py --url $(URL)

# Cached vs uncached outputs must match (loads its own copy of the model).
test:
	$(PYTHON) eval/check_cache.py --model $(MODEL) --size $(SIZE)

# llama.cpp on the GPU, for comparisons.
baseline:
	@mkdir -p $(LOG_DIR)
	@nohup llama-server -hf ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF --port $(BASELINE_PORT) --cache-reuse 256 -ngl 99 \
		>> $(LOG_DIR)/llama-server.log 2>&1 &
	@echo "llama-server starting on http://127.0.0.1:$(BASELINE_PORT) (log: $(LOG_DIR)/llama-server.log)"

stop-baseline:
	@pkill -f "llama-server .*--port $(BASELINE_PORT)" && echo "Stopped llama-server" || echo "llama-server not running"
