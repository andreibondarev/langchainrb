# frozen_string_literal: true

module Langchain
  module Chunker
    # LLM-powered semantic chunker.
    # Semantic chunking is a technique of splitting texts by their semantic meaning, e.g.: themes, topics, and ideas.
    # We use an LLM to accomplish this. The Anthropic LLM is highly recommended for this task as it has the longest context window (100k tokens).
    #
    # Usage:
    #     Langchain::Chunker::Semantic.new(
    #       text,
    #       llm: Langchain::LLM::Anthropic.new(api_key: ENV["ANTHROPIC_API_KEY"])
    #     ).chunks
    class Semantic < Base
      attr_reader :text, :llm, :prompt_template
      # @param [Langchain::LLM::Base] Langchain::LLM::* instance
      # @param [Langchain::Prompt::PromptTemplate] Optional custom prompt template
      def initialize(text, llm:, prompt_template: nil)
        @text = text
        @llm = llm
        @prompt_template = prompt_template || default_prompt_template
      end

      # Split the text into chunks
      #
      # @param [String] source
      # @return [Array<Langchain::Chunk>]
      def chunks(source: nil)
        prompt = prompt_template.format(text: text)

        # Replace static 50k limit with dynamic limit based on text length (max_tokens_to_sample)
        completion = llm.complete(prompt: prompt, max_tokens_to_sample: 50000).completion
        completion
          .gsub("Here are the paragraphs split by topic:\n\n", "")
          .split("---")
          .map(&:strip)
          .reject(&:empty?)
          .map do |chunk|
            Langchain::Chunk.new(text: chunk, source: source)
          end
      end

      private

      # @return [Langchain::Prompt::PromptTemplate] Default prompt template for semantic chunking
      def default_prompt_template
        Langchain::Prompt.load_from_path(
          file_path: Langchain.root.join("langchain/chunker/prompts/semantic_prompt_template.yml")
        )
      end
    end
  end
end
