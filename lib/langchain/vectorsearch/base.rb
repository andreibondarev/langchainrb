# frozen_string_literal: true

require "forwardable"

module Langchain::Vectorsearch
  # = Vector Databases
  # A vector database a type of database that stores data as high-dimensional vectors, which are mathematical representations of features or attributes. Each vector has a certain number of dimensions, which can range from tens to thousands, depending on the complexity and granularity of the data.
  #
  # == Available vector databases
  #
  # - {Langchain::Vectorsearch::Chroma}
  # - {Langchain::Vectorsearch::Milvus}
  # - {Langchain::Vectorsearch::Pinecone}
  # - {Langchain::Vectorsearch::Qdrant}
  # - {Langchain::Vectorsearch::Weaviate}
  # - {Langchain::Vectorsearch::Pgvector}
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
  #       llm:         :openai,              # or :cohere, :hugging_face, :google_palm, or :replicate
  #       llm_api_key: ENV["OPENAI_API_KEY"] # API key for the selected LLM
  #     )
  #
  #     # You can instantiate other supported vector databases the same way:
  #     milvus   = Langchain::Vectorsearch::Milvus.new(...)
  #     qdrant   = Langchain::Vectorsearch::Qdrant.new(...)
  #     pinecone = Langchain::Vectorsearch::Pinecone.new(...)
  #     chrome   = Langchain::Vectorsearch::Chroma.new(...)
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

    # Method supported by Vectorsearch DB to create a default schema
    def create_default_schema
      raise NotImplementedError, "#{self.class.name} does not support creating a default schema"
    end

    # Method supported by Vectorsearch DB to add a list of texts to the index
    def add_texts(...)
      raise NotImplementedError, "#{self.class.name} does not support adding texts"
    end

    # Method supported by Vectorsearch DB to search for similar texts in the index
    def similarity_search(...)
      raise NotImplementedError, "#{self.class.name} does not support similarity search"
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

    def_delegators :llm,
      :default_dimension

    def generate_prompt(question:, context:)
      prompt_template = Langchain::Prompt::FewShotPromptTemplate.new(
        prefix: "Context:",
        suffix: "---\nQuestion: {question}\n---\nAnswer:",
        example_prompt: Langchain::Prompt::PromptTemplate.new(
          template: "{context}",
          input_variables: ["context"]
        ),
        examples: [
          {context: context}
        ],
        input_variables: ["question"],
        example_separator: "\n"
      )

      prompt_template.format(question: question)
    end

    def add_data(paths:)
      raise ArgumentError, "Paths must be provided" if paths.to_a.empty?

      texts = Array(paths)
        .flatten
        .map { |path| Langchain::Loader.new(path)&.load&.value }
        .compact

      add_texts(texts: texts)
    end

    def self.logger_options
      {
        color: :blue
      }
    end
  end
end
