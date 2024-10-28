FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip --no-cache-dir install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

COPY ["*.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "subscription_serving:app"]