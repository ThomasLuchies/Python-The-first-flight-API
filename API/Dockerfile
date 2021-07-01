FROM python:3.8.0-buster

#Directory for the api
WORKDIR API/

#install dependancies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY /API .

CMD ["python", "main.py"]
