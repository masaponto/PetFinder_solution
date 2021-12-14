.PHONY: build run
build:
	poetry run python encode.py
	cat .build/script.py > output.py
run:
	pip install -r src/requirements.txt
	python -m src.main