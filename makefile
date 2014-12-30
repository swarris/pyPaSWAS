PROJECT = pypaswas
VIRTUALENV = venv
VENVACTIVATE = $(VIRTUALENV)/bin/activate
SLOCCOUNT = sloccount.txt
PYLINT = pylint.txt
PEP8 = pep8.txt
NOSE = nosetests.xml
COVERAGE = coverage.xml
EPYDOC = $(PROJECT)-epydoc
DISTRIBUTE = dist MANIFEST


PACKAGE = pyPaSWAS

all: clean $(VIRTUALENV) $(SLOCCOUNT) $(COVERAGE) $(PYLINT) $(PEP8) $(EPYDOC) $(DISTRIBUTE)

clean:
	-rm -r $(VIRTUALENV) $(SLOCCOUNT) $(PYLINT) $(PEP8) $(NOSE) $(COVERAGE) $(EPYDOC) $(DISTRIBUTE)

$(VIRTUALENV): $(VENVACTIVATE)
$(VENVACTIVATE): build-requirements.txt
	test -d $(VIRTUALENV) || virtualenv $(VIRTUALENV)
	. $@; \
	pip install -Ur $^; \
	pip install -Ur requirements.txt; \
	touch $@

$(SLOCCOUNT): $(PACKAGE)
	sloccount --duplicates --details $^ > $@

$(PYLINT): $(PACKAGE)
	-. $(VENVACTIVATE); \
	pylint --max-line-length=120 --disable="E0602,W0511,I0011,C0301" -f parseable --include-ids=y $^ > $@

$(PEP8): $(PACKAGE)
	-. $(VENVACTIVATE); \
	pep8 --ignore="E501,E262" $^ > $@

$(NOSE): $(PACKAGE)
	-. $(VENVACTIVATE); \
	nosetests --traverse-namespace $^ --verbosity=3 --with-coverage --with-xunit --xunit-file=$(NOSE)

$(COVERAGE): $(NOSE)
	-. $(VENVACTIVATE); \
	coverage xml -o $@

$(EPYDOC): $(PACKAGE)
	-. $(VENVACTIVATE); \
	epydoc -v --graph all --output $@ --name $(PROJECT) --url https://trac.nbic.nl/$(PROJECT)/ $^

$(DISTRIBUTE): $(PACKAGE)
	python setup.py sdist bdist
	rm -rf build

install: $(VIRTUALENV) $(DISTRIBUTE)
	. $(VENVACTIVATE); \
	cp dist/*.tar.gz .; \
	sh install.sh Y; \
	python -c "import pyPaSWAS.pypaswasall"
