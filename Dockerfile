FROM python:3.8
WORKDIR /myapp
COPY . /myapp
RUN apt update
RUN apt install -y gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config python3-pip python3-dev
RUN pip install -r requirements.txt
EXPOSE 80
CMD gunicorn -w 2 app:app -b 0.0.0.0:8080