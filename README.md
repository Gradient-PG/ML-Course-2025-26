# Gradient ML Course 2025/26

This repository contains hands-on regular **.py** files as well as **Jupyter Notebooks** designed to help you learn and experiment with key AI and machine learning concepts through practical examples.

---

## Getting Started

### I. Install the required tools

You will need **Python** and **Jupyter Notebook** (or JupyterLab).
**Python** can be installed from: [https://www.python.org/downloads/](https://www.python.org/downloads/).
Make sure you install pip as well (it should install itself along with python).

#### 1. Create a venv (optional but recommended on windows, mandatory for linux and mac):

```bash
python -m venv .venv
```

or like this in some cases:

```bash
python3 -m venv .venv
```

This should create a folder named .venv

#### 2. Activate the venv

windows version

```bash
./.venv/Scripts/activate.exe
```

linux/mac version

```bash
. ./.venv/bin/activate
```

After this something like `(.venv)` should appear on the left side of your terminal

#### 3. Install the required packages (e.g. `jupyter`):

```bash
pip install jupyter
```

or in some cases:

```bash
pip3 install jupyter
```

You can check how to install or verify a specific python package on pypi, e.g.:
[https://pypi.org/project/jupyter/](https://pypi.org/project/jupyter/)

This is also the way to solve **`ModuleNotFoundError`** for pretty much any module

---

### II. Running a Python file

You can run a .py file like this:

```bash
python file.py
```

or in some cases

```bash
python3 file.py
```

But most of the time, using whatever run function is integrated into your IDE (editor, like vscode or Pycharm) is much more comfortable

---

### III. Running a Notebook

You can start Jupyter locally by running:

```bash
jupyter notebook
```

This will open a local web interface where you can browse and run any notebook in this course.

---

## Recommended Setup: PyCharm

We strongly recommend using **PyCharm’s built-in Jupyter Notebook**, which provides:

* Better code navigation and debugging tools
* Integrated virtual environments
* Seamless execution of notebook cells directly within the IDE

Students have **free access to JetBrains Premium tools**, including PyCharm Professional.
You can claim your free educational license here:
[https://www.jetbrains.com/community/education/#students](https://www.jetbrains.com/community/education/#students)

After activation, simply open a notebook file (`.ipynb`) in PyCharm — it will automatically switch to notebook mode.

## Other Recommended Setup: Vscode

Vscode is free for everyone, a bit simpler, more versatile and more extensible, but requires a bit more work to setup
It can be downloaded here: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)

Vscode relies more on extensions e.g. to edit jupyter notebooks and python in general, you will need those:

[https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy)
[https://marketplace.visualstudio.com/items?itemName=ms-python.python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
[https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
[https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-python-envs](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-python-envs)
[https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
[https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

---

## Structure

Each notebook or .py file in this course is designed to help you **practice and explore topics discussed during the lectures**.
They contain various **tasks**, **exercises**, and **code sections to complete**, allowing you to apply theoretical knowledge in a hands-on way.

Typical notebook contents include:

* **Lecture-related topics** - concepts directly connected to what was covered in class
* **Theory overview** - a short summary of key ideas and background
* **Code examples** - working snippets that illustrate the discussed concepts
* **Exercises to complete** - missing code fragments, functions, or logic you need to fill in
* **Practice tasks** - small applied challenges or exploratory problems related to the lecture topics

---

### Learn by doing

The best way to learn AI (and programming in general) is to **build, test, and break things**.
Have fun exploring!
