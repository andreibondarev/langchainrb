# frozen_string_literal: true

module Langchain::Vectorsearch
  class Hnswlib < Base
    #
    # Wrapper around HNSW (Hierarchical Navigable Small World) library.
    # HNSWLib is an in-memory vectorstore that can be saved to a file on disk.
    #
    # Gem requirements:
    #     gem "hnswlib", "~> 0.8.1"
    #
    # Usage:
    #     hnsw = Langchain::Vectorsearch::Hnswlib.new(llm:, url:, index_name:)
    #

    attr_reader :client, :path_to_index

    #
    # Initialize the HNSW vector search
    #
    # @param llm [Object] The LLM client to use
    # @param path_to_index [String] The local path to the index file, e.g.: "/storage/index.ann"
    # @return [Langchain::Vectorsearch::Hnswlib] Class instance
    #
    def initialize(llm:, path_to_index:)
      depends_on "hnswlib"
      require "hnswlib"

      super(llm: llm)

      @client = ::Hnswlib::HierarchicalNSW.new(space: DEFAULT_METRIC, dim: llm.default_dimension)
      @path_to_index = path_to_index

      initialize_index
    end

    #
    # Add a list of texts and corresponding IDs to the index
    #
    # @param texts [Array] The list of texts to add
    # @param ids [Array] The list of corresponding IDs (integers) to the texts
    # @return [Boolean] The response from the HNSW library
    #
    def add_texts(texts:, ids:)
      resize_index(texts.size)

      Array(texts).each_with_index do |text, i|
        embedding = llm.embed(text: text)

        client.add_point(embedding, ids[i])
      end

      client.save_index(path_to_index)
    end

    #
    # Search for similar texts
    #
    # @param query [String] The text to search for
    # @param k [Integer] The number of results to return
    # @return [Array] Results in the format `[[id1, distance3], [id2, distance2]]`
    #
    def similarity_search(
      query:,
      k: 4
    )
      embedding = llm.embed(text: query)

      similarity_search_by_vector(
        embedding: embedding,
        k: k
      )
    end

    #
    # Search for the K nearest neighbors of a given vector
    #
    # @param embedding [Array] The embedding to search for
    # @param k [Integer] The number of results to return
    # @return [Array] Results in the format `[[id1, distance3], [id2, distance2]]`
    #
    def similarity_search_by_vector(
      embedding:,
      k: 4
    )
      client.search_knn(embedding, k)
    end

    private

    #
    # Optionally resizes the index if there's no space for new data
    #
    # @param num_of_elements_to_add [Integer] The number of elements to add to the index
    #
    def resize_index(num_of_elements_to_add)
      current_count = client.current_count

      if (current_count + num_of_elements_to_add) > client.max_elements
        new_size = current_count + num_of_elements_to_add

        client.resize_index(new_size)
      end
    end

    #
    # Loads or initializes the new index
    #
    def initialize_index
      if File.exist?(path_to_index)
        client.load_index(path_to_index)

        Langchain.logger.info("Successfully loaded the index at \"#{path_to_index}\"", for: self.class)
      else
        # Default max_elements: 100, but we constantly resize the index as new data is written to it
        client.init_index(max_elements: 100)

        Langchain.logger.info("Creating a new index at \"#{path_to_index}\"", for: self.class)
      end
    end
  end
end
