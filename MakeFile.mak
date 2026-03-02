## 🛠️ Makefile for easy local commands

This Makefile assumes Python 3.12 and a `.venv` environment.

```makefile
# MediBot Makefile

PYTHON = python
VENV = .venv
ACTIVATE = . $(VENV)/Scripts/activate

# Create virtual environment
venv:
    $(PYTHON) -m venv $(VENV)

# Install dependencies
install:
    $(ACTIVATE) && pip install -r requirements.txt

# Run the app
run:
    $(ACTIVATE) && python app.py

# Format code (optional if using black)
format:
    $(ACTIVATE) && black .

# Clean temporary files
clean:
    rm -rf __pycache__ */__pycache__ .pytest_cache

# Reset environment completely
reset:
    rm -rf $(VENV)
