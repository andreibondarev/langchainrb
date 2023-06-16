# frozen_string_literal: true

module Langchain::Vectorsearch
  class Pgvector < Base
    #
    # The PostgreSQL vector search adapter
    #
    # Gem requirements: gem "pgvector", "~> 0.2"
    #
    # Usage:
    # pgvector = Langchain::Vectorsearch::Pgvector.new(url:, index_name:, llm:, namespace_column: nil, namespace: nil)
    #

    # The operators supported by the PostgreSQL vector search adapter
    OPERATORS = {
      "cosine_distance" => "cosine",
      "euclidean_distance" => "euclidean"
    }
    DEFAULT_OPERATOR = "cosine_distance"

    attr_reader :db, :operator, :table_name, :namespace_column, :namespace, :documents_table

    # @param url [String] The URL of the PostgreSQL database
    # @param index_name [String] The name of the table to use for the index
    # @param llm [Object] The LLM client to use
    # @param namespace_column [String] The name of the column to use for the namespace
    # @param namespace [String] The namespace to use for the index when inserting/querying
    def initialize(url:, index_name:, llm:, namespace_column: nil, namespace: nil)
      depends_on "sequel"
      require "sequel"
      depends_on "pgvector"
      require "pgvector"

      @db = Sequel.connect(url)

      @table_name = index_name

      @namespace_column = namespace_column || "namespace"
      @namespace = namespace
      @operator = OPERATORS[DEFAULT_OPERATOR]

      super(llm: llm)
    end

    def documents_model
      Class.new(Sequel::Model(table_name.to_sym)) do
        plugin :pgvector, :vectors
      end
    end

    # Add a list of texts to the index
    # @param texts [Array<String>] The texts to add to the index
    # @return [PG::Result] The response from the database
    def add_texts(texts:)
      data = texts.flat_map do |text|
        embedding = llm.embed(text: text)
        {
          :content => text,
          :vectors => ::Pgvector.encode(embedding),
          namespace_column.to_sym => namespace
        }
      end

      documents_model.multi_insert(data, return: :primary_key)
    end

    # Create default schema
    # @return [PG::Result] The response from the database
    def create_default_schema
      db.run "CREATE EXTENSION IF NOT EXISTS vector"
      namespace = namespace_column
      vector_dimension = default_dimension
      db.create_table? table_name.to_sym do
        primary_key :id
        text :content
        column :vectors, "vector(#{vector_dimension})"
        text namespace.to_sym, default: nil
      end
    end

    # Search for similar texts in the index
    # @param query [String] The text to search for
    # @param k [Integer] The number of top results to return
    # @return [Array<Hash>] The results of the search
    def similarity_search(query:, k: 4)
      embedding = llm.embed(text: query)

      similarity_search_by_vector(
        embedding: embedding,
        k: k
      )
    end

    # Search for similar texts in the index by the passed in vector.
    # You must generate your own vector using the same LLM that generated the embeddings stored in the Vectorsearch DB.
    # @param embedding [Array<Float>] The vector to search for
    # @param k [Integer] The number of top results to return
    # @return [Array<Hash>] The results of the search
    def similarity_search_by_vector(embedding:, k: 4)
      db.transaction do # BEGIN
        documents_model
          .nearest_neighbors(:vectors, embedding, distance: operator).limit(k)
          .where(namespace_column.to_sym => namespace)
      end
    end

    # Ask a question and return the answer
    # @param question [String] The question to ask
    # @return [String] The answer to the question
    def ask(question:)
      search_results = similarity_search(query: question)

      context = search_results.map do |result|
        result.content.to_s
      end
      context = context.join("\n---\n")

      prompt = generate_prompt(question: question, context: context)

      llm.chat(prompt: prompt)
    end
  end
end
