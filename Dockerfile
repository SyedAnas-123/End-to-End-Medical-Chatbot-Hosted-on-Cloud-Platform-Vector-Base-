FROM python:3.11-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]



#python:3.11.7-slim-buster means:
# You’re using Python 3.11.7.
# slim-buster is a lightweight Debian-based version, so your image stays small (faster builds and deploys).


# Sets the working directory inside the container to /app.
# Any COPY, RUN, or CMD after this will happen in this directory.


# Copies your entire local project (current folder: .) into the container's /app directory.
# That includes your code files, requirements.txt, etc.

#It runs python3 app.py inside the /app directory
