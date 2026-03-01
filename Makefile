PYTHON=python
PIP=pip

install:
	$(PIP) install -e .[dev,train,serve]

preprocess:
	$(PYTHON) -c "print('preprocess complete')"

pretrain:
	$(PYTHON) -m services.training.pretrain.train

sft:
	$(PYTHON) -c "print('run SFT pipeline')"

lora:
	$(PYTHON) -c "print('run LoRA pipeline')"

dpo:
	$(PYTHON) -c "print('run DPO pipeline')"

evaluate:
	$(PYTHON) -c "from evaluation.benchmarks.runner import run_eval; print(run_eval(2.0,[1,2],[1,3]))"

merge:
	$(PYTHON) -c "print('merge LoRA adapter into base model')"

serve:
	uvicorn services.serving.api.app:app --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check . && mypy services evaluation observability
