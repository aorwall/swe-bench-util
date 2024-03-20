import logging
import os
from dataclasses import dataclass
from typing import List

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from swe_bench_util.retriever import CodeSnippetRetriever, CodeSnippet
from swe_bench_util.splitters.code_splitter import CodeSplitterV2


@dataclass
class IngestionPipelineSetup:
    name: str
    transformations: List[TransformComponent]
    embed_model: BaseEmbedding
    dimensions: int


class LlamaIndexCodeSnippetRetriever(CodeSnippetRetriever):

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    @classmethod
    def from_pipeline_setup(cls,
                            pipeline_setup: IngestionPipelineSetup,
                            path: str,
                            perist_dir: str = None):
        db = chromadb.PersistentClient(path=f"{perist_dir}/chroma_db")
        chroma_collection = db.get_or_create_collection("files")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        required_exts = [".py"]

        reader = SimpleDirectoryReader(
            input_dir=path,
            filename_as_id=True,
            required_exts=required_exts,
            recursive=True,
        )

        docs = reader.load_data()

        pipeline = IngestionPipeline(
            transformations=pipeline_setup.transformations + [pipeline_setup.embed_model],
            docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
            docstore=SimpleDocumentStore(),
            vector_store=vector_store
        )

        if os.path.exists(perist_dir):
            try:
                pipeline.load(perist_dir)
            except Exception as e:
                logging.error(f"Failed to load pipeline from {perist_dir}: {e}")

        nodes = pipeline.run(documents=docs, show_progress=True)
        print(f"Found {len(nodes)} new nodes")
        pipeline.persist(perist_dir)

        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=pipeline_setup.embed_model,
        )

        return cls(
            retriever=vector_index.as_retriever(similarity_top_k=250)
        )

    def retrieve(self, query: str) -> List[CodeSnippet]:
        result = self.retriever.retrieve(query)
        return [CodeSnippet(
            path=node.node.metadata['file_path'],
            content=node.node.get_content(),
        ) for node in result]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline_setup = IngestionPipelineSetup(
        name="text-embedding-3-small--code-splitter-v2-512",
        transformations=[
            CodeSplitterV2(max_tokens=512, language="python")
        ],
        embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
        dimensions=1536
    )

    retriever = LlamaIndexCodeSnippetRetriever.from_pipeline_setup(
        pipeline_setup=pipeline_setup,
        path="/tmp/repos/scikit-learn",
        perist_dir="/tmp/repos/scikit-learn-storage",
    )
