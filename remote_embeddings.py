import grpc
import numpy as np

import image_embedding_pb2 as pb2
import image_embedding_pb2_grpc as pb2_grpc

class ImageEmbeddingService:
    def __init__(self, host, load_balancing=False):
        if load_balancing:
            channel = grpc.insecure_channel(host, [("grpc.lb_policy_name", "round_robin")])
        else:
            channel = grpc.insecure_channel(host)
        stub = pb2_grpc.ImageEmbeddingStub(channel)
        response = stub.Dimension(pb2.Empty())
        self._dim = response.dim
        self._stub = stub

    def dim(self):
        return self._dim

    def get_embedding(self, url):
        try:
            response = self._stub.Embedding(pb2.EmbeddingRequest(url=url))
        except grpc.RpcError as e:
            if 'Forbidden' in str(e):
                return None
            raise e
        return np.array(response.embedding, dtype=np.float32)

