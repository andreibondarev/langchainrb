# frozen_string_literal: true

module Langchain::Vectorsearch
  class Elasticsearch < Base
    #
    # Wrapper around Elasticsearch vector search capabilities.
    #
    # Setting up Elasticsearch:
    # 1. Get Elasticsearch up and running with Docker: https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html
    # 2. Copy the HTTP CA certificate SHA-256 fingerprint and set the ELASTICSEARCH_CA_FINGERPRINT environment variable
    # 3. Set the ELASTICSEARCH_URL environment variable
    #
    # Gem requirements:
    #     gem "elasticsearch", "~> 8.0.0"
    #
    # Usage:
    #     llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
    #     es = Langchain::Vectorsearch::Elasticsearch.new(
    #       url: ENV["ELASTICSEARCH_URL"],
    #       index_name: "docs",
    #       llm: llm,
    #       es_options: {
    #         transport_options: {ssl: {verify: false}},
    #         ca_fingerprint: ENV["ELASTICSEARCH_CA_FINGERPRINT"]
    #       }
    #     )
    #
    #     es.create_default_schema
    #     es.add_texts(texts: ["..."])
    #     es.similarity_search(text: "...")
    #
    attr_accessor :es_client, :index_name, :options

    def initialize(url:, index_name:, llm:, api_key: nil, es_options: {})
      require "elasticsearch"

      @options = {
        url: url,
        request_timeout: 20,
        log: false
      }.merge(es_options)

      @es_client = ::Elasticsearch::Client.new(**options)
      @index_name = index_name

      super(llm: llm)
    end

    # Add a list of texts to the index
    # @param texts [Array<String>] The list of texts to add
    # @return [Elasticsearch::Response] from the Elasticsearch server
    def add_texts(texts: [])
      body = texts.map do |text|
        [
          {index: {_index: index_name}},
          {input: text, input_vector: llm.embed(text: text).embedding}
        ]
      end.flatten

      es_client.bulk(body: body)
    end

    # Add a list of texts to the index
    # @param texts [Array<String>] The list of texts to update
    # @param texts [Array<Integer>] The list of texts to update
    # @return [Elasticsearch::Response] from the Elasticsearch server
    def update_texts(texts: [], ids: [])
      body = texts.map.with_index do |text, i|
        [
          {index: {_index: index_name, _id: ids[i]}},
          {input: text, input_vector: llm.embed(text: text).embedding}
        ]
      end.flatten

      es_client.bulk(body: body)
    end

    # Remove a list of texts from the index
    # @param ids [Array<Integer>] The list of ids to delete
    # @return [Elasticsearch::Response] from the Elasticsearch server
    def remove_texts(ids: [])
      body = ids.map do |id|
        {delete: {_index: index_name, _id: id}}
      end

      es_client.bulk(body: body)
    end

    # Create the index with the default schema
    # @return [Elasticsearch::Response] Index creation
    def create_default_schema
      es_client.indices.create(
        index: index_name,
        body: default_schema
      )
    end

    # Deletes the default schema
    # @return [Elasticsearch::Response] Index deletion
    def delete_default_schema
      es_client.indices.delete(
        index: index_name
      )
    end

    def default_vector_settings
      {type: "dense_vector", dims: llm.default_dimension}
    end

    def vector_settings
      options[:vector_settings] || default_vector_settings
    end

    def default_schema
      {
        mappings: {
          properties: {
            input: {
              type: "text"
            },
            input_vector: vector_settings
          }
        }
      }
    end

    def default_query(query_vector)
      {
        script_score: {
          query: {match_all: {}},
          script: {
            source: "cosineSimilarity(params.query_vector, 'input_vector') + 1.0",
            params: {
              query_vector: query_vector
            }
          }
        }
      }
    end

    # Ask a question and return the answer
    # @param question [String] The question to ask
    # @param k [Integer] The number of results to have in context
    # @yield [String] Stream responses back one String at a time
    # @return [String] The answer to the question
    def ask(question:, k: 4, &block)
      search_results = similarity_search(query: question, k: k)

      context = search_results.map do |result|
        result[:input]
      end.join("\n---\n")

      prompt = generate_rag_prompt(question: question, context: context)

      messages = [{role: "user", content: prompt}]
      response = llm.chat(messages: messages, &block)

      response.context = context
      response
    end

    # Search for similar texts
    # @param text [String] The text to search for
    # @param k [Integer] The number of results to return
    # @param query [Hash] Elasticsearch query that needs to be used while searching (Optional)
    # @return [Elasticsearch::Response] The response from the server
    def similarity_search(text: "", k: 10, query: {})
      if text.empty? && query.empty?
        raise "Either text or query should pass as an argument"
      end

      if query.empty?
        query_vector = llm.embed(text: text).embedding

        query = default_query(query_vector)
      end

      es_client.search(body: {query: query, size: k}).body
    end

    # Search for similar texts by embedding
    # @param embedding [Array<Float>] The embedding to search for
    # @param k [Integer] The number of results to return
    # @param query [Hash] Elasticsearch query that needs to be used while searching (Optional)
    # @return [Elasticsearch::Response] The response from the server
    def similarity_search_by_vector(embedding: [], k: 10, query: {})
      if embedding.empty? && query.empty?
        raise "Either embedding or query should pass as an argument"
      end

      query = default_query(embedding) if query.empty?

      es_client.search(body: {query: query, size: k}).body
    end
  end
end
