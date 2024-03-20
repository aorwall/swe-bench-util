from typing import List

from swe_bench_util.retriever import CodeSnippetRetriever


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
