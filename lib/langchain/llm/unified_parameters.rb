# frozen_string_literal: true

module Langchain::LLM
  class UnifiedParameters
    include Enumerable

    attr_reader :schema, :aliases, :parameters, :ignored, :remapped

    class Null < self
      def initialize(parameters: {})
        super(schema: {}, parameters: parameters)
      end
    end

    def initialize(schema:, parameters: {})
      @schema = schema || {}
      @aliases = {}
      @remapped = {}
      @ignored = Set.new
      @schema.each do |name, param|
        @aliases[name] = Set.new(Array(param[:aliases])) if param[:aliases]
      end
      @parameters = to_params(parameters.to_h) if !parameters.to_h.empty?
    end

    def to_params(params = {})
      # if params are provided, reset any previously initialized
      @parameters = params if !params.empty?
      @parameters = (@parameters || {}).merge!(params).slice(*schema.keys)
      @aliases.each do |field, aliased_keys|
        # favor existing keys in case of conflicts,
        # and check for multiples
        aliased_keys.each do |alias_key|
          @parameters[field] ||= params[alias_key] if value_present?(params[alias_key])
        end
      end
      @schema.each do |field, param_options|
        param_options ||= {}
        default = param_options[:default]
        @parameters[field] ||= default if value_present?(default)
      end
      @remapped.each do |field, renamed_field|
        @parameters[renamed_field] = @parameters[field] if value_present?(@parameters[field])
      end
      @parameters = @parameters.except(*@ignored + @remapped.keys)
    end

    def remap(field_map)
      @remapped ||= {}
      @remapped.merge!(field_map)
      field_map.each do |field, renamed_field|
        @schema[renamed_field] = @schema[field]
      end
    end

    def update(schema = {})
      @schema.merge!(schema)
      schema.each do |name, param|
        if param[:aliases]
          @aliases[name] ||= Set.new
          @aliases[name] << param[:aliases]
        end
      end
      self
    end

    def ignore(*field_names)
      @ignored.merge(field_names)
    end

    def alias_field(field_name, as:)
      @aliases[field_name] ||= Set.new
      @aliases[field_name] << as
    end

    def to_h
      @parameters.to_h
    end

    def each(&block)
      to_params.each(&block)
    end

    def <=>(other)
      to_params.<=>(other.to_params)
    end

    def [](key)
      to_params[key]
    end

    private

    def value_present?(value)
      !value.nil? && (!value.is_a?(Enumerable) || !value.empty?)
    end
  end
end
