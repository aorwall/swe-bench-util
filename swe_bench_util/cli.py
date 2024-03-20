"""This module provides the CLI."""
import json
import os
import subprocess
import sys
from typing import Optional, List

import typer
from datasets import load_dataset
from llama_index.core.node_parser import TokenTextSplitter, CodeSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

from swe_bench_util import __app_name__, __version__
from swe_bench_util.llama_index_retriever import IngestionPipelineSetup, LlamaIndexCodeSnippetRetriever
from swe_bench_util.retriever import CodeSnippetRetriever
from swe_bench_util.splitters.code_splitter import CodeSplitterV2

app = typer.Typer()
get_app = typer.Typer()
run_app = typer.Typer()
app.add_typer(get_app, name="get")
app.add_typer(run_app, name="run")

devin_scikit_learn_pass = [
    "scikit-learn__scikit-learn-10297",
    "scikit-learn__scikit-learn-10870",
    "scikit-learn__scikit-learn-10986",
    "scikit-learn__scikit-learn-11578",
    "scikit-learn__scikit-learn-12973",
    "scikit-learn__scikit-learn-13496",
    "scikit-learn__scikit-learn-14496",
    "scikit-learn__scikit-learn-15100",
    "scikit-learn__scikit-learn-15119",
    "scikit-learn__scikit-learn-15512",
    "scikit-learn__scikit-learn-19664"
]

ingestion_pipelines = [
    IngestionPipelineSetup(
        name="text-embedding-3-small--text-splitter-512",
        transformations=[
            TokenTextSplitter(chunk_size=512)
        ],
        embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
        dimensions=1536
    ),
    IngestionPipelineSetup(
        name="text-embedding-3-small--code-splitter-1500",
        transformations=[
            CodeSplitter(max_chars=1500, language="python")
        ],
        embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
        dimensions=1536
    ),
    IngestionPipelineSetup(
        name="text-embedding-3-small--code-splitter-v2-256",
        transformations=[
            CodeSplitterV2(max_tokens=256, language="python")
        ],
        embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
        dimensions=1536
    )

]




def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()

def maybe_clone(repo_url, repo_dir):
    if not os.path.exists(f"{repo_dir}/.git"):
        # Clone the repo if the directory doesn't exist
        result = subprocess.run(['git', 'clone', repo_url, repo_dir], check=True, text=True, capture_output=True)

        if result.returncode == 0:
            print(f"Repo '{repo_url}' was cloned to '{repo_dir}'", file=sys.stderr)
        else:
            print(f"Failed to clone repo '{repo_url}' to '{repo_dir}'", file=sys.stderr)
            raise typer.Exit(code=1)
    else:
        print(f"Repo '{repo_url}' already exists in '{repo_dir}'", file=sys.stderr)

def checkout_commit(repo_dir, commit_hash):
    subprocess.run(['git', 'reset', '--hard', commit_hash], cwd=repo_dir, check=True)

def write_file(path, text):
    with open(path, 'w') as f:
        f.write(text)
        print(f"File '{path}' was saved", file=sys.stderr)

def write_json(path, name, data):
    json_str = json.dumps(data, indent=2)
    json_path = f"{path}/{name}.json"
    write_file(json_path, json_str)

def format_markdown_code_block(text):
    text = text.replace('```', '\\`\\`\\`')
    return f"```\n{text}\n```"

def write_markdown(path, name, data):
    template_fields = [
        "instance_id"
        "repo",
        "base_commit",
        "problem_statement"
    ]
    text = f"""# {data['instance_id']}

* repo: {data['repo']}
* base_commit: {data['base_commit']}

## problem_statement
{data['problem_statement']}
"""
    for k, v in data.items():
        if k not in template_fields:
            text += f"""## {k}\n{format_markdown_code_block(v)}\n\n"""
    md_path = f"{path}/{name}.md"
    write_file(md_path, text)

@get_app.command()
def row(index:int=0, id:str=None, split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    """Download one row"""

    row_data = get_row(index=index, id=id, split=split, dataset_name=dataset_name)
    id = row_data['instance_id']
    write_json('rows', f"{id}", row_data)
    write_markdown('rows', f"{id}", row_data)


def get_row(index:int=0, id:str=None, split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    dataset = load_dataset(dataset_name, split=split)

    if id is not None:
        for i, row_data in enumerate(dataset):
            if row_data['instance_id'] == id:
                index = i
                break
        else:
            print(f"Row with id '{id}' not found in the dataset", file=sys.stderr)
            raise typer.Exit(code=1)

    return dataset[index]

def diff_file_names(text: str) -> list[str]:
    return [
        line[len("+++ b/"):] 
        for line in text.split('\n') 
        if line.startswith('+++')
    ]


@get_app.command()
def oracle(split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    """Download oracle (patched files) for all rows in split"""
    dataset = load_dataset(dataset_name, split=split)
    result = []
    for row_data in dataset:
        patch_files = diff_file_names(row_data['patch'])
        test_patch_files = diff_file_names(row_data['test_patch'])
        result.append({
            "id": row_data['instance_id'],
            "repo": row_data['repo'],
            "created_at": row_data['created_at'],
            "base_commit": row_data['base_commit'],
            "patch_files": patch_files,
            "test_patch_files": test_patch_files,
            "problem_statement": row_data['problem_statement'],
        })
    write_json('rows', "oracle", result)


def get_case(id: str):
    with open(f'rows/oracle.json') as f:
        oracle_json = json.load(f)

    for row in oracle_json:
        if row['id'] == id:
            return row


def calculate_precision_recall(recommended_files: List[str], patch_files: List[str]):
    true_positives = set(recommended_files) & set(patch_files)
    precision = len(true_positives) / len(recommended_files) if len(recommended_files) > 0 else 0
    recall = len(true_positives) / len(patch_files) if len(patch_files) > 0 else 0

    return precision, recall


def benchmark_retrieve(retriever: CodeSnippetRetriever, query: str, patch_files: list[str]):
    result = retriever.retrieve(query)
    files = [snippet.path for snippet in result]
    recommended_files = []
    for file in files:
        if file not in recommended_files:
            recommended_files.append(file)

    precision, recall = calculate_precision_recall(recommended_files, patch_files)
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    printed_files = set()

    for i, file_path in enumerate(recommended_files, start=1):
        if file_path in printed_files:
            continue

        if file_path in patch_files:
            print(f" -> {i}: {file_path}")
            patch_files.remove(file_path)
        else:
            print(f"    {i}: {file_path}")

        printed_files.add(file_path)

        if not patch_files:
            break

    print("\nMissing Patch Files:")
    for i, file_path in enumerate(patch_files, start=1):
        print(f"{i}: {file_path}")


@run_app.command()
def benchmark(id:str=None, split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    row_data = get_case(id=id)
    repo_name = row_data['repo'].split('/')[-1]
    repo = f'git@github.com:{row_data["repo"]}.git'
    base_commit = row_data['base_commit']
    path = f'/tmp/repos/{repo_name}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' was created.")
    maybe_clone(repo, path)
    checkout_commit(path, base_commit)

    pipeline_setup = ingestion_pipelines[0]

    retriever = LlamaIndexCodeSnippetRetriever.from_pipeline_setup(
        pipeline_setup=pipeline_setup,
        path=path,
        perist_dir=f"/tmp/repos/{repo_name}-storage/{pipeline_setup.name}",
    )

    patch_files = [f"{path}/{file}" for file in row_data["patch_files"]]

    benchmark_retrieve(
            retriever=retriever,
            query=row_data["problem_statement"],
            patch_files=patch_files
    )


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    return