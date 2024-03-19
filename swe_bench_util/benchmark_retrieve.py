import logging
from typing import List

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

import faiss

from swe_bench_util.cached_embedding import CachedEmbedding





def benchmark_retrieve(path: str, query: str, expected_files: list[str]):

    storage_context = StorageContext.from_defaults()

    embed_model = CachedEmbedding(persist_dir=".cache", embed_model=OpenAIEmbedding(model="text-embedding-3-small"))

    required_exts = [".py"]

    reader = SimpleDirectoryReader(
        input_dir=path,
        required_exts=required_exts,
        recursive=True,
    )

    docs = reader.load_data()

    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(),
            embed_model,
        ]
    )

    nodes = pipeline.run(documents=docs, show_progress=True)
    print(f"Found {len(nodes)}")

    embed_model.persist()

    vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=250,
    )

    result = retriever.retrieve(query)

    for i, node in enumerate(result):
        file_path = node.node.metadata['file_path']
        if file_path in expected_files:
            print(f"{i}: {file_path} - {node.score}")
            expected_files.remove(file_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    benchmark_retrieve(
        path="/tmp/princeton-nlp/SWE-bench-dev/sqlfluff/a820c139ccbe6d1865d73c4a459945cd69899f8f",
        query="""Enable quiet mode/no-verbose in CLI for use in pre-commit hook
There seems to be only an option to increase the level of verbosity when using SQLFluff [CLI](https://docs.sqlfluff.com/en/stable/cli.html), not to limit it further.

It would be great to have an option to further limit the amount of prints when running `sqlfluff fix`, especially in combination with deployment using a pre-commit hook. For example, only print the return status and the number of fixes applied, similar to how it is when using `black` in a pre-commit hook:
![image](https://user-images.githubusercontent.com/10177212/140480676-dc98d00b-4383-44f2-bb90-3301a6eedec2.png)

This hides the potentially long list of fixes that are being applied to the SQL files, which can get quite verbose.""",
        expected_files=[
            "/tmp/princeton-nlp/SWE-bench-dev/sqlfluff/a820c139ccbe6d1865d73c4a459945cd69899f8f/src/sqlfluff/cli/commands.py",
            "/tmp/princeton-nlp/SWE-bench-dev/sqlfluff/a820c139ccbe6d1865d73c4a459945cd69899f8f/src/sqlfluff/cli/formatters.py",
            "/tmp/princeton-nlp/SWE-bench-dev/sqlfluff/a820c139ccbe6d1865d73c4a459945cd69899f8f/src/sqlfluff/core/linter/linted_dir.py"]
    )
