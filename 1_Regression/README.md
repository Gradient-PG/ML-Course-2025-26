# How to Run This Lecture

This repository contains a Manim lecture presentation. Follow the steps below to get started.

## 1. Clone the Repository or Download the File

You can either clone the entire repository:

```bash
git clone <repository_url>
cd <repository_name>
````

or simply download the `slides.py` file to your working directory.

## 2. Install Required Packages

First, make sure you have Python installed (Python 3.9+ recommended).

Then, install Manim:

```bash
pip install manim
```

Next, install Manim Slides with full PySide6 support:

```bash
pip install -U "manim-slides[pyside6-full]"
```

## 3. Render the Slides

Render the lecture slides with the following command:

```bash
manim-slides render slides.py
```

This will generate all the slides in your local output folder.

## 4. Display the Lecture

Once rendered, you can display the slides using:

```bash
manim-slides
```

Use the keyboard arrows to navigate through the lecture.

## Notes

* Make sure you are in the directory containing `slides.py` when running the commands.
* For more information about Manim Slides, visit: [https://manim.community/slides](https://manim.community/slides)
