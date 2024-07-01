# frozen_string_literal: true

module Langchain::Tool
  class GoogleSearch < Base
    #
    # Wrapper around SerpApi's Google Search API
    #
    # Gem requirements:
    #     gem "google_search_results", "~> 2.0.0"
    #
    # Usage:
    #     search = Langchain::Tool::GoogleSearch.new(api_key: "YOUR_API_KEY")
    #     search.execute(input: "What is the capital of France?")
    #
    NAME = "google_search"
    FUNCTIONS = [:execute]

    attr_reader :api_key

    #
    # Initializes the Google Search tool
    #
    # @param api_key [String] Search API key
    # @return [Langchain::Tool::GoogleSearch] Google search tool
    #
    def initialize(api_key:)
      super()

      depends_on "google_search_results"

      @api_key = api_key
    end

    #
    # Executes Google Search and returns hash_results JSON
    #
    # @param input [String] search query
    # @return [Hash] hash_results JSON
    #
    def self.execute_search(input:)
      new.execute_search(input: input)
    end

    # Executes Google Search and returns the result
    #
    # @param input [String] search query
    # @return [String] Answer
    def execute(input:)
      Langchain.logger.info("Executing \"#{input}\"", for: self.class)

      results = execute_search(input: input)

      answer_box = results[:answer_box_list] ? results[:answer_box_list].first : results[:answer_box]
      if answer_box
        return answer_box[:result] ||
            answer_box[:answer] ||
            answer_box[:snippet] ||
            answer_box[:snippet_highlighted_words] ||
            answer_box.reject { |_k, v| v.is_a?(Hash) || v.is_a?(Array) || v.start_with?("http") }
      elsif (events_results = results[:events_results])
        return events_results.take(10)
      elsif (sports_results = results[:sports_results])
        return sports_results
      elsif (top_stories = results[:top_stories])
        return top_stories
      elsif (news_results = results[:news_results])
        return news_results
      elsif (jobs_results = results.dig(:jobs_results, :jobs))
        return jobs_results
      elsif (shopping_results = results[:shopping_results]) && shopping_results.first.key?(:title)
        return shopping_results.take(3)
      elsif (questions_and_answers = results[:questions_and_answers])
        return questions_and_answers
      elsif (popular_destinations = results.dig(:popular_destinations, :destinations))
        return popular_destinations
      elsif (top_sights = results.dig(:top_sights, :sights))
        return top_sights
      elsif (images_results = results[:images_results]) && images_results.first.key?(:thumbnail)
        return images_results.map { |h| h[:thumbnail] }.take(10)
      end

      snippets = []
      if (knowledge_graph = results[:knowledge_graph])
        snippets << knowledge_graph[:description] if knowledge_graph[:description]

        title = knowledge_graph[:title] || ""
        knowledge_graph.each do |k, v|
          if v.is_a?(String) &&
              k != :title &&
              k != :description &&
              !k.to_s.end_with?("_stick") &&
              !k.to_s.end_with?("_link") &&
              !k.to_s.start_with?("http")
            snippets << "#{title} #{k}: #{v}"
          end
        end
      end

      if (first_organic_result = results.dig(:organic_results, 0))
        if (snippet = first_organic_result[:snippet])
          snippets << snippet
        elsif (snippet_highlighted_words = first_organic_result[:snippet_highlighted_words])
          snippets << snippet_highlighted_words
        elsif (rich_snippet = first_organic_result[:rich_snippet])
          snippets << rich_snippet
        elsif (rich_snippet_table = first_organic_result[:rich_snippet_table])
          snippets << rich_snippet_table
        elsif (link = first_organic_result[:link])
          snippets << link
        end
      end

      if (buying_guide = results[:buying_guide])
        snippets << buying_guide
      end

      if (local_results = results.dig(:local_results, :places))
        snippets << local_results
      end

      return "No good search result found" if snippets.empty?
      snippets
    end

    #
    # Executes Google Search and returns hash_results JSON
    #
    # @param input [String] search query
    # @return [Hash] hash_results JSON
    #
    def execute_search(input:)
      ::GoogleSearch
        .new(q: input, serp_api_key: api_key)
        .get_hash
    end
  end
end
