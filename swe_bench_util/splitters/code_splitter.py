"""Code splitter.

Extended version of the code splitter from llama-index.
."""
import re
from collections import namedtuple
from typing import Any, Callable, List, Optional

from llama_index.core import get_tokenizer
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.node_parser.interface import TextSplitter
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.schema import Document
from tree_sitter import Node

DEFAULT_MAX_TOKENS = 1024
DEFAULT_MIN_TOKENS = 50

Span = namedtuple('Span', ['start_byte', 'end_byte'])


class CodeSplitterV2(TextSplitter):
    """Split code using a AST parser.

    Thank you to Kevin Lu / SweepAI for suggesting this elegant code splitting solution.
    https://docs.sweep.dev/blogs/chunking-2m-files
    """

    language: str = Field(
        description="The programming language of the code being split."
    )

    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        description="Maximum number of characters per chunk.",
        gt=0,
    )

    _parser: Any = PrivateAttr()
    _tokenizer: Callable = PrivateAttr()

    def __init__(
        self,
        language: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        parser: Any = None,
        tokenizer: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> None:
        """Initialize a CodeSplitter."""
        from tree_sitter import Parser  # pants: no-infer-dep

        if parser is None:
            try:
                import tree_sitter_languages  # pants: no-infer-dep

                parser = tree_sitter_languages.get_parser(language)
            except ImportError:
                raise ImportError(
                    "Please install tree_sitter_languages to use CodeSplitter."
                    "Or pass in a parser object."
                )
            except Exception:
                print(
                    f"Could not get parser for language {language}. Check "
                    "https://github.com/grantjenks/py-tree-sitter-languages#license "
                    "for a list of valid languages."
                )
                raise
        if not isinstance(parser, Parser):
            raise ValueError("Parser must be a tree-sitter Parser object.")

        self._parser = parser
        self._tokenizer = tokenizer or get_tokenizer()

        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func

        super().__init__(
            language=language,
            max_tokens=max_tokens,
            callback_manager=callback_manager,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            id_func=id_func,
        )


    @classmethod
    def class_name(cls) -> str:
        return "CodeSplitter"

    def _chunk_node(self, node: Any, text: str, last_end: int = 0) -> List[str]:
        new_chunks = []
        current_chunk = ""
        for child in node.children:
            if child.end_byte - child.start_byte > self.max_tokens:
                # Child is too big, recursively chunk the child
                if len(current_chunk) > 0:
                    new_chunks.append(current_chunk)
                current_chunk = ""
                new_chunks.extend(self._chunk_node(child, text, last_end))
            elif (
                len(current_chunk) + child.end_byte - child.start_byte > self.max_tokens
            ):
                # Child would make the current chunk too big, so start a new chunk
                new_chunks.append(current_chunk)
                current_chunk = text[last_end : child.end_byte]
            else:
                current_chunk += text[last_end : child.end_byte]
            last_end = child.end_byte
        if len(current_chunk) > 0:
            new_chunks.append(current_chunk)
        return new_chunks

    def _count_tokens(self, text: str):
        return len(self._tokenizer(text))

    def chunk_tree(
        self,
        tree,
        source_code: bytes
    ) -> list[Span]:
        from tree_sitter import Node

        def chunk_node(node: Node) -> list[Span]:
            chunks: list[Span] = []
            current_chunk = None

            for child in node.children:
                if not current_chunk:
                    if self._ignore_node(child):
                        continue
                    else:
                        current_chunk = Span(child.start_byte, child.start_byte)

                content = source_code[child.start_byte:child.end_byte].decode("utf-8")
                if self._count_tokens(content) > self.max_tokens:
                    chunks.append(current_chunk)
                    current_chunk = Span(child.end_byte, child.end_byte)
                    chunks.extend(chunk_node(child))
                    continue

                content = source_code[current_chunk.start_byte:child.end_byte].decode("utf-8")
                if self._count_tokens(content) > self.max_tokens:
                    chunks.append(current_chunk)
                    current_chunk = Span(child.start_byte, child.end_byte)
                else:
                    current_chunk = Span(current_chunk.start_byte, child.end_byte)

            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        chunks = chunk_node(tree.root_node)

        if len(chunks) < 2:
            return chunks

        for i in range(len(chunks) - 1):
            chunks[i] = Span(chunks[i].start_byte, chunks[i + 1].start_byte)
        chunks[-1] = Span(chunks[-1].start_byte, tree.root_node.end_byte)

        combined_chunks = []
        tokens = 0
        current_chunk = Span(chunks[0].start_byte, chunks[0].start_byte)
        for chunk in chunks:
            current_chunk_content = source_code[current_chunk.start_byte:current_chunk.end_byte].decode("utf-8")
            current_chunk_tokens = self._count_tokens(current_chunk_content)

            chunk_content = source_code[chunk.start_byte:chunk.end_byte].decode("utf-8")
            chunk_tokens = self._count_tokens(chunk_content)

            if current_chunk_tokens + chunk_tokens < self.max_tokens:
                current_chunk = Span(current_chunk.start_byte, chunk.end_byte)
                tokens = current_chunk_tokens + chunk_tokens
                continue

            if chunk_tokens > self.max_tokens > current_chunk_tokens:
                current_chunk = Span(current_chunk.start_byte, chunk.end_byte)
                combined_chunks.append(current_chunk)
                current_chunk = Span(chunk.end_byte, chunk.end_byte)
                tokens = 0
                continue

            combined_chunks.append(current_chunk)
            current_chunk = Span(chunk.start_byte, chunk.end_byte)
            tokens = chunk_tokens

        if len(current_chunk) > 0:
            if tokens < DEFAULT_MIN_TOKENS:
                chunk_content = source_code[combined_chunks[-1].start_byte:current_chunk.end_byte].decode("utf-8")
                if self._count_tokens(chunk_content) < self.max_tokens:
                    combined_chunks[-1] = Span(combined_chunks[-1].start_byte, current_chunk.end_byte)
                else:
                    combined_chunks.append(current_chunk)
            else:
                combined_chunks.append(current_chunk)

        return combined_chunks

    def get_line_number(self, index: int, source_code: str) -> int:
        total_chars = 0
        line_number = 0
        for line_number, line in enumerate(source_code.splitlines(keepends=True), start=1):
            total_chars += len(line)
            if total_chars > index:
                return line_number - 1
        return line_number

    def split_text(self, text: str) -> List[str]:
        """Split incoming code and return chunks using the AST."""
        with self.callback_manager.event(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: [text]}
        ) as event:
            content_bytes = bytes(text, "utf-8")
            tree = self._parser.parse(content_bytes)

            if (
                not tree.root_node.children
                or tree.root_node.children[0].type != "ERROR"
            ):
                spans = self.chunk_tree(tree, content_bytes)

                chunks = []
                for span in spans:
                    chunked_content = content_bytes[span.start_byte:span.end_byte].decode("utf-8")
                    if len(chunked_content) == 0:
                        continue

                    chunks.append(chunked_content)

                event.on_end(
                    payload={EventPayload.CHUNKS: chunks},
                )

                return chunks
            else:
                raise ValueError(f"Could not parse code with language {self.language}.")

        # TODO: set up auto-language detection using something like https://github.com/yoeo/guesslang.

    def _ignore_node(self, node: Node):
        # Skip copyright and license comments in the beginning of the file
        if node.start_byte == 0 and node.type == "comment":
            text = node.text.decode("utf-8")
            if re.search(r"copyright|license|author", text):
                return True

        return False
