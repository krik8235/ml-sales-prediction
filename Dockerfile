# use an official Python runtime
FROM public.ecr.aws/lambda/python:3.12

# set the working directory in the container
WORKDIR /app

# to handle lightgbm
RUN dnf install -y libgomp

# copy the entire repository (except .dockerignore) contents into the container at /app
COPY . /app/

# install deps
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python" ]

CMD [ "-m", "awslambdaric", "app.handler" ]


# no need for containarlized lambda func
EXPOSE 8080