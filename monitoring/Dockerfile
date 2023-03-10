#
FROM python:3.10.4

#
WORKDIR /monitoring

#
COPY ./ ./

#
ENV VIRTUAL_ENV=./monitoring-venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

#
RUN pip install --no-cache-dir -r ./requirements.txt

#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]