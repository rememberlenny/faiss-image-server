import logging
import grpc
import numpy as np
from time import time

import image_embedding_pb2 as pb2
import image_embedding_pb2_grpc as pb2_grpc

class ImageEmbeddingService:
    KEEP_STUB_SEC = 60

    def __init__(self, host, load_balancing=False, timeout=10):
        self.host = host
        self.load_balancing = load_balancing
        self._timeout = timeout
        self._stub = self._get_stub()

        response = self._stub.Dimension(pb2.Empty(), timeout=1)
        self._dim = response.dim

    def _get_stub(self):
        if self.load_balancing:
            self._channel = grpc.insecure_channel(self.host, [("grpc.lb_policy_name", "round_robin")])
        else:
            self._channel = grpc.insecure_channel(self.host)
        return pb2_grpc.ImageEmbeddingStub(self._channel)

    def dim(self):
        return self._dim

    def get_embedding(self, url, is_retry=False):
        try:
            response = self._stub.Embedding(pb2.EmbeddingRequest(url=url), timeout=self._timeout)
        except grpc.RpcError as e:
            msg = str(e)
            if 'Forbidden' in msg:
                return None
            if not is_retry:
                logging.warn("embedding error: %s", msg)
                self._stub = self._get_stub()
                return self.get_embedding(url, is_retry=True)
            raise e
        return np.array(response.embedding, dtype=np.float32)

    def info(self):
        response = self._stub.Info(pb2.Empty(), timeout=1)
        return response.message

    def stop(self):
        self._channel.close()
