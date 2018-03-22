import grpc
import numpy as np

import image_embedding_pb2 as pb2
import image_embedding_pb2_grpc as pb2_grpc

class ImageEmbeddingService:
    def __init__(self, host):
        channel = grpc.insecure_channel(host)
        stub = pb2_grpc.ImageEmbeddingStub(channel)
        response = stub.Dimension(pb2.Empty())
        self._dim = response.dim
        self._stub = stub

    def dim(self):
        return self._dim

    def get_embedding(self, url):
        response = self._stub.Embedding(pb2.EmbeddingRequest(url=url))
        return np.array(response.embedding, dtype=np.float32)

