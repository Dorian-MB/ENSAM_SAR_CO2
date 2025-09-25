
install:
	poetry install


tree:
	git ls-files | tree --fromfile

py-tree:
	git ls-files | grep ".py" | tree --fromfile


st: 
	poetry run streamlit run streamlit_app.py


black:
	poetry run black .

ruff:
	poetry run ruff format .


# compter le nombre de ligne du projet
count-lines:
	find . -name "*.py" -type f -exec wc -l {} + | tail -1

tokei:
	tokei .

cloc:
	cloc .
