.PHONY: build
build:
	poetry run python encode.py
	cat .build/script.py