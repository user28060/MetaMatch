FORMAT_DIR=MetaMatch/src

format:
	poetry run autoflake -r --in-place --remove-all-unused-imports --remove-unused-variables $(FORMAT_DIR)
	poetry run isort $(FORMAT_DIR)
	poetry run black $(FORMAT_DIR)

lint:
	poetry run flake8 $(FORMAT_DIR)

fix: format lint
