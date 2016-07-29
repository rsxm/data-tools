# data-tools

Utilities to extract, clean and analyse data from various sources, especially dirty Excel files.

Detailed documentation is in the "docs" directory.

## Quick start

1. Clone this repository::

    git clone https://github.com/jr-minnaar/data-tools
    cd data-tools

1. Create a python virtual environment and activate::

    pyvenv-3.5 venv
    source env/bin/activate

2. Update pip and innstall development dependencies::

    pip install -U pip
    pip install -r requirements.txt

3.  Copy example configuration and edit the file to match data to be extracted:

    cp example.config.py config/config.py

4. Add the config file name to the main loop in extractor.py

5. Run the extractor by double clicking the run.command (Mac) or run.bat (windows), or using python:

    python extractor.py

