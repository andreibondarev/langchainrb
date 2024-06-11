module Langchain
  module Utils
    class HashTransformer
      # Converts a string to camelCase
      def self.camelize_lower(str)
        str.split("_").inject([]) { |buffer, e| buffer.push(buffer.empty? ? e : e.capitalize) }.join
      end

      # Recursively transforms the keys of a hash to camel case
      def self.deep_transform_keys(hash, &block)
        case hash
        when Hash
          hash.each_with_object({}) do |(key, value), result|
            new_key = block.call(key)
            result[new_key] = deep_transform_keys(value, &block)
          end
        when Array
          hash.map { |item| deep_transform_keys(item, &block) }
        else
          hash
        end
      end
    end
  end
end
