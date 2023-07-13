# frozen_string_literal: true

module Langchain::Vectorsearch
  class Pgvector < Base
    #
    # The PostgreSQL vector search adapter
    #
    # Gem requirements: gem "pgvector", "~> 0.2"
    #
    # Usage:
    # pgvector = Langchain::Vectorsearch::Pgvector.new(url:, index_name:, llm:, llm_api_key:)
    #

    # The operators supported by the PostgreSQL vector search adapter
    OPERATORS = {
      "cosine_distance" => "<=>",
      "euclidean_distance" => "<->"
    }
    DEFAULT_OPERATOR = "cosine_distance"

    attr_reader :operator, :quoted_table_name

    # @param url [String] The URL of the PostgreSQL database
    # @param index_name [String] The name of the table to use for the index
    # @param llm [Object] The LLM client to use
    # @param api_key [String] The API key for the Vectorsearch DB (not used for PostgreSQL)
    def initialize(url:, index_name:, llm:, api_key: nil)
      require "pg"
      require "pgvector"

      @client = ::PG.connect(url)
      registry = ::PG::BasicTypeRegistry.new.define_default_types
      ::Pgvector::PG.register_vector(registry)
      @client.type_map_for_results = PG::BasicTypeMapForResults.new(@client, registry: registry)

      @index_name = index_name
      @quoted_table_name = @client.quote_ident(index_name)
      @operator = OPERATORS[DEFAULT_OPERATOR]

      super(llm: llm)
    end

    # Upsert a list of texts to the index
    # @param texts [Array<String>] The texts to add to the index
    # @param ids [Array<Integer>] The ids of the objects to add to the index, in the same order as the texts
    # @return [PG::Result] The response from the database including the ids of
    # the added or updated texts.
    def upsert_texts(texts:, ids:)
      data = texts.zip(ids).flat_map do |(text, id)|
        [id, text, llm.embed(text: text)]
      end
      values = texts.length.times.map { |i| "($#{3 * i + 1}, $#{3 * i + 2}, $#{3 * i + 3})" }.join(",")
      # see https://github.com/pgvector/pgvector#storing
      client.exec_params(
        "INSERT INTO #{quoted_table_name} (id, content, vectors) VALUES
#{values} ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, vectors = EXCLUDED.vectors RETURNING id;",
        data
      )
    end

    # Add a list of texts to the index
    # @param texts [Array<String>] The texts to add to the index
    # @param ids [Array<String>] The ids to add to the index, in the same order as the texts
    # @return [PG::Result] The response from the database including the ids of
    # the added texts.
    def add_texts(texts:, ids: nil)
      if ids.nil? || ids.empty?
        data = texts.flat_map do |text|
          [text, llm.embed(text: text)]
        end
        values = texts.length.times.map { |i| "($#{2 * i + 1}, $#{2 * i + 2})" }.join(",")
        client.exec_params(
          "INSERT INTO #{quoted_table_name} (content, vectors) VALUES #{values} RETURNING id;",
          data
        )
      else
        upsert_texts(texts: texts, ids: ids)
      end
    end

    # Update a list of ids and corresponding texts to the index
    # @param texts [Array<String>] The texts to add to the index
    # @param ids [Array<String>] The ids to add to the index, in the same order as the texts
    # @return [PG::Result] The response from the database including the ids of
    # the updated texts.
    def update_texts(texts:, ids:)
      upsert_texts(texts: texts, ids: ids)
    end

    # Create default schema
    # @return [PG::Result] The response from the database
    def create_default_schema
      client.exec("CREATE EXTENSION IF NOT EXISTS vector;")
      client.exec(
        <<~SQL
          CREATE TABLE IF NOT EXISTS #{quoted_table_name} (
            id serial PRIMARY KEY,
            content TEXT,
            vectors VECTOR(#{default_dimension})
          );
        SQL
      )
    end

    # TODO: Add destroy_default_schema method

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
      result = client.transaction do |conn|
        conn.exec("SET LOCAL ivfflat.probes = 10;")
        query = <<~SQL
          SELECT id, content FROM #{quoted_table_name} ORDER BY vectors #{operator} $1 ASC LIMIT $2;
        SQL
        conn.exec_params(query, [embedding, k])
      end

      result.to_a
    end

    # Ask a question and return the answer
    # @param question [String] The question to ask
    # @return [String] The answer to the question
    def ask(question:)
      search_results = similarity_search(query: question)

      context = search_results.map do |result|
        result["content"].to_s
      end
      context = context.join("\n---\n")

      prompt = generate_prompt(question: question, context: context)

      llm.chat(prompt: prompt)
    end
  end
end
