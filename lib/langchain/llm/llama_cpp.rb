# frozen_string_literal: true

module Langchain::LLM
  class LlamaCpp < Base
    attr_accessor :model_path, :n_gpu_layers, :n_ctx, :seed
    attr_writer :n_threads

    def initialize(model_path:, n_gpu_layers: 1, n_ctx: 2048, n_threads: 1, seed: -1)
      depends_on "llama_cpp"
      require "llama_cpp"

      @model_path = model_path
      @n_gpu_layers = n_gpu_layers
      @n_ctx = n_ctx
      @n_threads = n_threads
      @seed = seed
    end

    def embed(text:, n_threads: nil)
      # contexts are kinda stateful when it comes to embeddings, so allocate one each time
      context = build_context_params(embeddings: true)
      embedding_input = context.tokenize(text: text, add_bos: true)
      return unless embedding_input.size.positive?

      n_threads ||= self.n_threads

      context.eval(tokens: embedding_input, n_past: 0, n_threads: n_threads)
      context.embeddings
    end

    def complete(prompt:, n_predict: 128, n_threads: nil)
      n_threads ||= self.n_threads
      # contexts do not appear to be stateful when it comes to completion, so re-use the same one
      context = completion_context
      ::LLaMACpp.generate(context, prompt, n_threads: n_threads, n_predict: n_predict)
    end

    private

    def n_threads
      # Use the maximum number of CPU threads available, if not configured
      @n_threads ||= `sysctl -n hw.ncpu`.strip.to_i
    end


    def build_context_params(embeddings: false)
      context_params = ::LLaMACpp::ContextParams.new

      context_params.seed = seed
      context_params.n_ctx = n_ctx
      context_params.n_gpu_layers = n_gpu_layers
      context_params.embeddings = embeddings

      context_params
    end

    def build_model(embeddings: false)
      return @model if defined?(@model)
      @model = ::LLaMACpp::Model.new(model_path: model_path, params: build_context_params(embeddings: embeddings))
    end

    def build_context
      ::LLaMACpp::Context.new(model: build_model)
    end

    def completion_context
      @completion_context ||= build_context
    end

  end
end
