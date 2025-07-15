This project is meant as a personal learning experience.  I've often considered trying to track grocery prices for personal data analysis.  Here I am attempting to train an AI to detect grocery store prices within images of store displays.

Goals:
1. Reliably detect a grocery store price label within an image.
2. Reliably extract the price from the detected label.
3. Store externally provided product identity with detected price.


## Setup
1. Create virtual environment: `python -m venv yolov5-venv`
2. Activate: run/source `yolov5-venv/bin/activate`
3. Install dependencies: `pip install -r yolov5-venv-requirements.txt
