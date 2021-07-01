FROM python:3.8

#Directory for the api
WORKDIR src/

#install dependancies
RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 --no-cache install -r requirements.txt

COPY /src .

EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD ["main.py"]
