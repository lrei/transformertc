pylint:
	pylint --exit-zero transformertc

mypy:
	mypy transformertc

pipreqs:
	pipreqs transformertc --savepath requirements.exact.txt

doc:
	cd docs %% make html

