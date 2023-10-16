# frozen_string_literal: true

module Langchain
  module Processors
    class Code < Text
      EXTENSIONS = [".rb", ".js", ".py"]
    end
  end
end
