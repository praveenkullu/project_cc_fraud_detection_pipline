.PHONY: install notebook train test test-unit test-integration lint serve docker-up docker-down benchmark retrain seed-chargebacks register-models migrate

install:
	pip install -r requirements.txt

notebook:
	jupyter notebook notebooks/

train:
	python -m src.train

retrain:
	python -m scripts.retrain

seed-chargebacks:
	python -m scripts.seed_chargebacks

register-models:
	python -m scripts.register_models

migrate:
	docker compose exec db psql -U fraud_user -d fraud_db -f /docker-entrypoint-initdb.d/002_model_promotions.sql

serve:
	uvicorn src.app:app --reload --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/ -v --cov=src --cov-report=term-missing --ignore=tests/test_integration.py

test-integration:
	pytest tests/test_integration.py -v --no-cov

lint:
	python -m flake8 src/ tests/ --max-line-length=100

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

benchmark:
	python scripts/benchmark.py
