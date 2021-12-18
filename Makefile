.PHONY: build run
build:
	poetry run python encode.py
	cat .build/script.py > output.py
run:
	docker run --rm --runtime nvidia --shm-size '2gb' -v `pwd`:/works -v `pwd`/data:/kaggle/input -w /works gcr.io/kaggle-gpu-images/python:latest /bin/bash -c \
		"pip install -r src/requirements.txt && python -m src.main"