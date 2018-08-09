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
        self._stub = None
        self._last_stub_time = 0
        self._timeout = timeout

        response = self._get_stub().Dimension(pb2.Empty())
        self._dim = response.dim

    def _get_stub(self):
        if (time() - self._last_stub_time) < self.KEEP_STUB_SEC:
            return self._stub
        if self.load_balancing:
            channel = grpc.insecure_channel(self.host, [("grpc.lb_policy_name", "round_robin")])
        else:
            channel = grpc.insecure_channel(self.host)
        self._stub = pb2_grpc.ImageEmbeddingStub(channel)
        self._last_stub_time = time()
        return self._stub

    def dim(self):
        return self._dim

    def get_embedding(self, url, is_retry=False):
        try:
            response = self._get_stub().Embedding(pb2.EmbeddingRequest(url=url), timeout=self._timeout)
        except grpc.RpcError as e:
            msg = str(e)
            if 'Forbidden' in msg:
                return None
            if not is_retry:
                logging.warn("embedding error: %s", msg)
                self._last_stub_time = 0
                return self.get_embedding(url, is_retry=True)
            raise e
        return np.array(response.embedding, dtype=np.float32)
