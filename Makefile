
tree:
	git ls-files | tree --fromfile

py-tree:
	git ls-files | grep ".py" | tree --fromfile

