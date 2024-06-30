# frozen_string_literal: true

module Langchain
  module Messages
    # Generic messages that can be used in the `chat` method of the LLM adapter
    # It is recommended that `assistants/messages/*_messages.rb` be unified with this one in the future.
    class ChatMessage
      attr_reader :role, :content

      ROLES = [:system, :user, :assistant].freeze

      class << self
        def system(content)
          new(role: :system, content:)
        end

        def user(content)
          new(role: :user, content:)
        end

        def assistant(content)
          new(role: :assistant, content:)
        end
      end

      def initialize(role:, content: "")
        raise ArgumentError, "Role must be one of #{ROLES.join(", ")}" unless ROLES.member?(role.to_sym)

        @role = role.to_sym
        @content = content
      end

      def to_hash
        {role: role.to_s, content:}
      end

      alias_method :to_h, :to_hash

      def system?
        role == :system
      end

      def user?
        role == :user
      end

      def assistant?
        role == :assistant
      end
    end
  end
end
