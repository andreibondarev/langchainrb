# frozen_string_literal: true

module Langchain
  module Chunker
    # = Chunkers
    # Chunkers are used to split documents into smaller chunks before indexing into vector search databases.
    # Otherwise large documents, when retrieved and passed to LLMs, may hit the context window limits.
    #
    # == Available chunkers
    #
    # - {Langchain::Chunker::RecursiveText}
    # - {Langchain::Chunker::Text}
    # - {Langchain::Chunker::Semantic}
    # - {Langchain::Chunker::Sentence}
    class Base
      LANG_SEPARATORS = {
        ".js" => [
          # Split along function definitions
          "\nfunction ",
          "\nconst ",
          "\nlet ",
          "\nvar ",
          "\nclass ",
          # Split along control flow statements
          "\nif ",
          "\nfor ",
          "\nwhile ",
          "\nswitch ",
          "\ncase ",
          "\ndefault ",
          # Split by the normal type of lines
          "\n\n",
          "\n",
          " ",
          ""
        ],
        ".py" => [
          # First, try to split along class definitions
          "\nclass ",
          "\ndef ",
          "\n\tdef ",
          # Now split by the normal type of lines
          "\n\n",
          "\n",
          " ",
          ""
        ],
        ".rb" => [
          # Split along method definitions
          "\ndef ",
          "\nclass ",
          # Split along control flow statements
          "\nif ",
          "\nunless ",
          "\nwhile ",
          "\nfor ",
          "\ndo ",
          "\nbegin ",
          "\nrescue ",
          # Split by the normal type of lines
          "\n\n",
          "\n",
          " ",
          ""
        ]
      }
    end
  end
end
