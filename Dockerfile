FROM python:3.8

WORKDIR /app

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

EXPOSE 5000

ADD ./static ./static
ADD ./templates/ ./templates
ADD ./uploaded ./uploaded
ADD simpsons.h5 simpsons.h5
ADD sale.pkl sale.pkl
ADD app.py app.py

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
