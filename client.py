from __future__ import print_function
import os
from time import time

import click
import grpc

import faissimageindex_pb2 as pb2
import faissimageindex_pb2_grpc as pb2_grpc

@click.group()
def cli():
    pass

@click.command()
@click.option('--host', default='localhost:50051', help='host:port')
def info(host):
    with grpc.insecure_channel(host) as channel:
        stub = pb2_grpc.ImageIndexStub(channel)
        response = stub.Info(pb2.Empty())
        print(response.message)

@click.command()
@click.option('--host', default='localhost:50051', help='host:port')
@click.option('--count', default=5, help='results limit count')
@click.argument('id')
def search(host, id, count):
    with grpc.insecure_channel(host) as channel:
        stub = pb2_grpc.ImageEmbeddingStub(channel)
        response = stub.Search(pb2.SearchRequest(id=id, count=5))
        print("%s, %s" % (response.ids, response.scores))

@click.command()
@click.option('--host', default='localhost:50051', help='host:port')
@click.argument('id')
@click.argument('url')
def add(host, id, url):
    with grpc.insecure_channel(host) as channel:
        stub = pb2_grpc.ImageEmbeddingStub(channel)
        response = stub.Add(pb2.AddRequest(id=id, created_at_ts=time(), url=url))
        print(response.message)

@click.command()
@click.option('--host', default='localhost:50051', help='host:port')
@click.argument('id')
def remove(host, id):
    with grpc.insecure_channel(host) as channel:
        stub = pb2_grpc.ImageEmbeddingStub(channel)
        response = stub.Remove(pb2.IdRequest(id=id))
        print(response.message)

cli.add_command(info)
cli.add_command(search)
cli.add_command(add)
cli.add_command(remove)

if __name__ == '__main__':
  cli()
