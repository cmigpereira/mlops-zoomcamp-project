setup:
	pipenv install --dev
	pre-commit install

quality_checks:
	isort .
	black .
	pylint --recursive=y .

unit_tests:
	pytest unit_tests/tests.py

integration_tests:
	bash integration_tests/run.sh

build: quality_checks unit_tests integration_tests
	docker compose -f docker-compose-airflow.yml build
	docker compose -f docker-compose-airflow.yml up airflow-init
	docker compose -f docker-compose-airflow.yml up -d
	sleep 20
	docker exec airflow-deploy airflow dags trigger create_datasets
	sleep 20
	docker exec airflow-deploy airflow dags trigger initial_deploy
	sleep 20
	docker compose -f docker-compose-airflow.yml down
	docker-compose -f docker-compose-service.yml up --build -d