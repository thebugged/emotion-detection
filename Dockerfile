FROM python:3.10.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1


RUN pip install --upgrade pip
COPY ./requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app


CMD ["streamlit", "run", "main.py"]
