# frozen_string_literal: true

module Langchain::Vectorsearch
  # = Vector Databases
  # A vector database a type of database that stores data as high-dimensional vectors, which are mathematical representations of features or attributes. Each vector has a certain number of dimensions, which can range from tens to thousands, depending on the complexity and granularity of the data.
  #
  # == Available vector databases
  #
  # - {Langchain::Vectorsearch::Chroma}
  # - {Langchain::Vectorsearch::Epsilla}
  # - {Langchain::Vectorsearch::Elasticsearch}
  # - {Langchain::Vectorsearch::Hnswlib}
  # - {Langchain::Vectorsearch::Milvus}
  # - {Langchain::Vectorsearch::Pgvector}
  # - {Langchain::Vectorsearch::Pinecone}
  # - {Langchain::Vectorsearch::Qdrant}
  # - {Langchain::Vectorsearch::Weaviate}
  #
  # == Usage
  #
  # 1. Pick a vector database from list.
  # 2. Review its documentation to install the required gems, and create an account, get an API key, etc
  # 3. Instantiate the vector database class:
  #
  #     weaviate = Langchain::Vectorsearch::Weaviate.new(
  #       url:         ENV["WEAVIATE_URL"],
  #       api_key:     ENV["WEAVIATE_API_KEY"],
  #       index_name:  "Documents",
  #       llm:         Langchain::LLM::OpenAI.new(api_key:)
  #     )
  #
  #     # You can instantiate other supported vector databases the same way:
  #     epsilla  = Langchain::Vectorsearch::Epsilla.new(...)
  #     milvus   = Langchain::Vectorsearch::Milvus.new(...)
  #     qdrant   = Langchain::Vectorsearch::Qdrant.new(...)
  #     pinecone = Langchain::Vectorsearch::Pinecone.new(...)
  #     chroma   = Langchain::Vectorsearch::Chroma.new(...)
  #     pgvector = Langchain::Vectorsearch::Pgvector.new(...)
  #
  # == Schema Creation
  #
  # `create_default_schema()` creates default schema in your vector database.
  #
  #     search.create_default_schema
  #
  # (We plan on offering customizable schema creation shortly)
  #
  # == Adding Data
  #
  # You can add data with:
  # 1. `add_data(path:, paths:)` to add any kind of data type
  #
  #     my_pdf = Langchain.root.join("path/to/my.pdf")
  #     my_text = Langchain.root.join("path/to/my.txt")
  #     my_docx = Langchain.root.join("path/to/my.docx")
  #     my_csv = Langchain.root.join("path/to/my.csv")
  #
  #     search.add_data(paths: [my_pdf, my_text, my_docx, my_csv])
  #
  # 2. `add_texts(texts:)` to only add textual data
  #
  #     search.add_texts(
  #       texts: [
  #         "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
  #         "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s"
  #       ]
  #     )
  #
  # == Retrieving Data
  #
  # `similarity_search_by_vector(embedding:, k:)` searches the vector database for the closest `k` number of embeddings.
  #
  #    search.similarity_search_by_vector(
  #      embedding: ...,
  #      k: # number of results to be retrieved
  #    )
  #
  # `vector_store.similarity_search(query:, k:)` generates an embedding for the query and searches the vector database for the closest `k` number of embeddings.
  #
  # search.similarity_search_by_vector(
  #   embedding: ...,
  #   k: # number of results to be retrieved
  # )
  #
  # `ask(question:)` generates an embedding for the passed-in question, searches the vector database for closest embeddings and then passes these as context to the LLM to generate an answer to the question.
  #
  #     search.ask(question: "What is lorem ipsum?")
  #
  class Base
    include Langchain::DependencyHelper
    extend Forwardable

    attr_reader :client, :index_name, :llm

    DEFAULT_METRIC = "cosine"

    # @param llm [Object] The LLM client to use
    def initialize(llm:)
      @llm = llm
    end

    # Method supported by Vectorsearch DB to retrieve a default schema
    def get_default_schema
      raise NotImplementedError, "#{self.class.name} does not support retrieving a default schema"
    end

    # Method supported by Vectorsearch DB to create a default schema
    def create_default_schema
      raise NotImplementedError, "#{self.class.name} does not support creating a default schema"
    end

    # Method supported by Vectorsearch DB to delete the default schema
    def destroy_default_schema
      raise NotImplementedError, "#{self.class.name} does not support deleting a default schema"
    end

    # Method supported by Vectorsearch DB to add a list of texts to the index
    def add_texts(...)
      raise NotImplementedError, "#{self.class.name} does not support adding texts"
    end

    # Method supported by Vectorsearch DB to update a list of texts to the index
    def update_texts(...)
      raise NotImplementedError, "#{self.class.name} does not support updating texts"
    end

    # Method supported by Vectorsearch DB to delete a list of texts from the index
    def remove_texts(...)
      raise NotImplementedError, "#{self.class.name} does not support deleting texts"
    end

    # Method supported by Vectorsearch DB to search for similar texts in the index
    def similarity_search(...)
      raise NotImplementedError, "#{self.class.name} does not support similarity search"
    end

    # Paper: https://arxiv.org/abs/2212.10496
    # Hypothetical Document Embeddings (HyDE)-augmented similarity search
    #
    # @param query [String] The query to search for
    # @param k [Integer] The number of results to return
    # @return [String] Response
    def similarity_search_with_hyde(query:, k: 4)
      hyde_completion = llm.complete(prompt: generate_hyde_prompt(question: query)).completion
      similarity_search(query: hyde_completion, k: k)
    end

    # Method supported by Vectorsearch DB to search for similar texts in the index by the passed in vector.
    # You must generate your own vector using the same LLM that generated the embeddings stored in the Vectorsearch DB.
    def similarity_search_by_vector(...)
      raise NotImplementedError, "#{self.class.name} does not support similarity search by vector"
    end

    # Method supported by Vectorsearch DB to answer a question given a context (data) pulled from your Vectorsearch DB.
    def ask(...)
      raise NotImplementedError, "#{self.class.name} does not support asking questions"
    end

    # HyDE-style prompt
    #
    # @param [String] User's question
    # @return [String] Prompt
    def generate_hyde_prompt(question:)
      prompt_template = Langchain::Prompt.load_from_path(
        # Zero-shot prompt to generate a hypothetical document based on a given question
        file_path: Langchain.root.join("langchain/vectorsearch/prompts/hyde.yaml")
      )
      prompt_template.format(question: question)
    end

    # Retrieval Augmented Generation (RAG)
    #
    # @param question [String] User's question
    # @param context [String] The context to synthesize the answer from
    # @return [String] Prompt
    def generate_rag_prompt(question:, context:)
      prompt_template = Langchain::Prompt.load_from_path(
        file_path: Langchain.root.join("langchain/vectorsearch/prompts/rag.yaml")
      )
      prompt_template.format(question: question, context: context)
    end

    def add_data(paths:, options: {}, chunker: Langchain::Chunker::Text)
      raise ArgumentError, "Paths must be provided" if Array(paths).empty?

      texts = Array(paths)
        .flatten
        .map do |path|
          data = Langchain::Loader.new(path, options, chunker: chunker)&.load&.chunks
          data.map { |chunk| chunk.text }
        end

      texts.flatten!

      add_texts(texts: texts)
    end

    def self.logger_options
      {
        color: :blue
      }
    end
  end
end
