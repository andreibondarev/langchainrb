# frozen_string_literal: true

module Langchain::LLM
  class UnifiedParameters
    include Enumerable

    attr_reader :schema, :aliases, :parameters

    class Null < self
      def initialize(aliases: {}, parameters: {})
        super(schema: {}, aliases: aliases, parameters: parameters)
      end
    end

    def initialize(schema:, aliases: {}, parameters: {})
      @schema = schema || {}
      @aliases = aliases || {}
      @parameters = to_params(parameters.to_h) if !parameters.to_h.empty?
    end

    def to_params(params = {})
      @parameters ||= params.slice(*schema.keys)
      @aliases.each do |new_key, existing_key|
        # favor existing keys in case of conflicts
        @parameters[existing_key] ||= params[new_key]
      end
      @parameters
    end

    def add(schema: {}, aliases: {})
      @schema.merge!(schema)
      @aliases.merge!(aliases || {})
      self
    end

    def to_h
      @parameters.to_h
    end

    def each(&)
      to_params.each(&)
    end

    def <=>(other)
      to_params.<=>(other.to_params)
    end

    def [](key)
      to_params[key]
    end
  end
end
