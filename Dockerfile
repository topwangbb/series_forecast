FROM python:3.6.8

RUN mkdir -p /app/app
RUN mkdir -p /app/bin
RUN mkdir -p /app/logs

ADD ./app /app/app
ADD ./bin /app/bin

RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip3 install -r /app/app/requirements.txt

ENV PYTHONUNBUFFERED=1

RUN chmod +s /app/bin/chowner
RUN chmod +x /app/bin/*
WORKDIR /app/bin

EXPOSE 8888
CMD ["bash","sarima_start.sh"]