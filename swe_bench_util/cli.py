"""This module provides the CLI."""
import os
import subprocess
from typing import Optional
import json
import sys

import typer

from swe_bench_util import __app_name__, __version__

from datasets import load_dataset

app = typer.Typer()
get_app = typer.Typer()
run_app = typer.Typer()
app.add_typer(get_app, name="get")
app.add_typer(run_app, name="run")

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
    #TODO?         f"git reset --hard {base_commit}",
    subprocess.run(['git', 'checkout', commit_hash], cwd=repo_dir, check=True)

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
def row(index:int=0, split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    """Download one row"""
    dataset = load_dataset(dataset_name, split=split)
    row_data = dataset[index]
    id = row_data['instance_id']
    write_json('rows', f"{id}", row_data)
    write_markdown('rows', f"{id}", row_data)
    
def diff_file_names(text: str) -> list[str]:
    return [
        line[len("+++ b/"):] 
        for line in text.split('\n') 
        if line.startswith('+++')
    ]


@get_app.command()
def oracle(split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    """Down load oracle (patched files) for all rows in split"""
    dataset = load_dataset(dataset_name, split=split)
    result = []
    for row_data in dataset:
        patch_files = diff_file_names(row_data['patch'])
        test_patch_files = diff_file_names(row_data['test_patch'])
        result.append({
            "id": row_data['instance_id'],
            "repo": row_data['repo'],
            "base_commit": row_data['base_commit'],
            "patch_files": patch_files,
            "test_patch_files": test_patch_files 
        })
    write_json('rows', "oracle", result)


@run_app.command()
def retrieve(index:int=0, split: str='dev', dataset_name='princeton-nlp/SWE-bench'):
    dataset = load_dataset(dataset_name, split=split)
    row_data = dataset[index]
    repo_name = row_data['repo'].split('/')[-1]
    repo = f'git@github.com:{row_data["repo"]}.git'
    base_commit = row_data['base_commit']
    path = f'/tmp/{dataset_name}-{split}/{repo_name}/{base_commit}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' was created.")
    maybe_clone(repo, path)
    checkout_commit(path, base_commit)
    pass

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