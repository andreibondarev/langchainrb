# frozen_string_literal: true

module Langchain::Vectorsearch
  class Pinecone < Base
    #
    # Wrapper around Pinecone API.
    #
    # Gem requirements:
    #     gem "pinecone", "~> 0.1.6"
    #
    # Usage:
    #     pinecone = Langchain::Vectorsearch::Pinecone.new(environment:, api_key:, index_name:, llm:)
    #

    # Initialize the Pinecone client
    # @param environment [String] The environment to use
    # @param api_key [String] The API key to use
    # @param index_name [String] The name of the index to use
    # @param llm [Object] The LLM client to use
    def initialize(environment:, api_key:, index_name:, llm:, base_uri: nil)
      depends_on "pinecone"

      ::Pinecone.configure do |config|
        config.api_key = api_key
        config.environment = environment
        config.base_uri = base_uri if base_uri
      end

      @client = ::Pinecone::Client.new
      @index_name = index_name

      super(llm: llm)
    end

    # Find records by ids
    # @param ids [Array<Integer>] The ids to find
    # @param namespace String The namespace to search through
    # @return [Hash] The response from the server
    def find(ids: [], namespace: "")
      raise ArgumentError, "Ids must be provided" if Array(ids).empty?

      client.index(index_name).fetch(
        ids: ids,
        namespace: namespace
      )
    end

    # Add a list of texts to the index
    # @param texts [Array<String>] The list of texts to add
    # @param ids [Array<Integer>] The list of IDs to add
    # @param namespace [String] The namespace to add the texts to
    # @param metadata [Hash] The metadata to use for the texts
    # @return [Hash] The response from the server
    def add_texts(texts:, ids: [], namespace: "", metadata: nil)
      vectors = texts.map.with_index do |text, i|
        {
          id: ids[i] ? ids[i].to_s : SecureRandom.uuid,
          metadata: metadata || {content: text},
          values: llm.embed(text: text).embedding
        }
      end

      index = client.index(index_name)

      index.upsert(vectors: vectors, namespace: namespace)
    end

    def add_data(paths:, namespace: "", options: {}, chunker: Langchain::Chunker::Text)
      raise ArgumentError, "Paths must be provided" if Array(paths).empty?

      texts = Array(paths)
        .flatten
        .map do |path|
          data = Langchain::Loader.new(path, options, chunker: chunker)&.load&.chunks
          data.map { |chunk| chunk.text }
        end

      texts.flatten!

      add_texts(texts: texts, namespace: namespace)
    end

    # Update a list of texts in the index
    # @param texts [Array<String>] The list of texts to update
    # @param ids [Array<Integer>] The list of IDs to update
    # @param namespace [String] The namespace to update the texts in
    # @param metadata [Hash] The metadata to use for the texts
    # @return [Array] The response from the server
    def update_texts(texts:, ids:, namespace: "", metadata: nil)
      texts.map.with_index do |text, i|
        # Pinecone::Vector#update ignore args when it is empty
        index.update(
          namespace: namespace,
          id: ids[i].to_s,
          values: llm.embed(text: text).embedding,
          set_metadata: metadata || {content: text}
        )
      end
    end

    # Create the index with the default schema
    # @return [Hash] The response from the server
    def create_default_schema
      client.create_index(
        metric: DEFAULT_METRIC,
        name: index_name,
        dimension: llm.default_dimension
      )
    end

    # Delete the index
    # @return [Hash] The response from the server
    def destroy_default_schema
      client.delete_index(index_name)
    end

    # Get the default schema
    # @return [Pinecone::Vector] The default schema
    def get_default_schema
      index
    end

    # Search for similar texts
    # @param query [String] The text to search for
    # @param k [Integer] The number of results to return
    # @param namespace [String] The namespace to search in
    # @param filter [String] The filter to use
    # @return [Array] The list of results
    def similarity_search(
      query:,
      k: 4,
      namespace: "",
      filter: nil
    )
      embedding = llm.embed(text: query).embedding

      similarity_search_by_vector(
        embedding: embedding,
        k: k,
        namespace: namespace,
        filter: filter
      )
    end

    # Search for similar texts by embedding
    # @param embedding [Array<Float>] The embedding to search for
    # @param k [Integer] The number of results to return
    # @param namespace [String] The namespace to search in
    # @param filter [String] The filter to use
    # @return [Array] The list of results
    def similarity_search_by_vector(embedding:, k: 4, namespace: "", filter: nil)
      index = client.index(index_name)

      query_params = {
        vector: embedding,
        namespace: namespace,
        filter: filter,
        top_k: k,
        include_values: true,
        include_metadata: true
      }.compact

      response = index.query(query_params)
      response.dig("matches")
    end

    # Ask a question and return the answer
    # @param question [String] The question to ask
    # @param namespace [String] The namespace to search in
    # @param k [Integer] The number of results to have in context
    # @param filter [String] The filter to use
    # @yield [String] Stream responses back one String at a time
    # @return [String] The answer to the question
    def ask(question:, namespace: "", filter: nil, k: 4, &block)
      search_results = similarity_search(query: question, namespace: namespace, filter: filter, k: k)

      context = search_results.map do |result|
        result.dig("metadata").to_s
      end
      context = context.join("\n---\n")

      prompt = generate_rag_prompt(question: question, context: context)

      messages = [{role: "user", content: prompt}]
      response = llm.chat(messages: messages, &block)

      response.context = context
      response
    end

    # Pinecone index
    # @return [Object] The Pinecone index
    private def index
      client.index(index_name)
    end
  end
end
