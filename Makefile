
VENV=.venv
BIN=$(VENV)/bin
export PYTHON=$(BIN)/python


# ----------------------------------------
# ENVIRONMENT SETUP
# ----------------------------------------

venv:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install pip-tools --index-url https://pypi.org/simple/
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install transformers torch sentence-transformers datasets