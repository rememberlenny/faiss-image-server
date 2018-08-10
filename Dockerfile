FROM daangn/faiss

ENV GRPC_PYTHON_VERSION 1.14.0
RUN python -m pip install --upgrade pip
RUN pip install grpcio==${GRPC_PYTHON_VERSION} grpcio-tools==${GRPC_PYTHON_VERSION}

RUN pip install tensorflow==1.9.0
RUN pip install pillow==5.2.0

RUN mkdir -p /app
WORKDIR /app

#ONBUILD COPY requirements.txt /usr/src/app/
#ONBUILD RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gevent==1.3.5

# https://tensorflow.blog/2017/05/12/tf-%EC%84%B1%EB%8A%A5-%ED%8C%81-winograd-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%84%A4%EC%A0%95/
ENV TF_ENABLE_WINOGRAD_NONFUSED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

ENTRYPOINT ["python"]
CMD ["server.py"]

HEALTHCHECK --interval=3s --timeout=2s \
  CMD ls /tmp/status || exit 1

RUN apt-get -qq update && apt-get -qq install wget
RUN mkdir nets && cd nets && \
      wget -q https://github.com/tensorflow/models/raw/master/research/slim/nets/__init__.py && \
      wget -q https://github.com/tensorflow/models/raw/master/research/slim/nets/inception_utils.py && \
      wget -q https://github.com/tensorflow/models/raw/master/research/slim/nets/inception_v4.py

RUN pip install -q scikit-learn==0.19.2
RUN pip install -q scipy==1.1.0
RUN pip install -q boto3
RUN pip install -q click

# for click library
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY *.py /app/
