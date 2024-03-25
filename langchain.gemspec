# frozen_string_literal: true

require_relative "lib/langchain/version"

Gem::Specification.new do |spec|
  spec.name = "langchainrb"
  spec.version = Langchain::VERSION
  spec.authors = ["Andrei Bondarev"]
  spec.email = ["andrei.bondarev13@gmail.com"]

  spec.summary = "Build LLM-backed Ruby applications with Ruby's Langchain.rb"
  spec.description = "Build LLM-backed Ruby applications with Ruby's Langchain.rb"
  spec.homepage = "https://rubygems.org/gems/langchainrb"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/andreibondarev/langchainrb"
  spec.metadata["changelog_uri"] = "https://github.com/andreibondarev/langchainrb/CHANGELOG.md"
  spec.metadata["documentation_uri"] = "https://rubydoc.info/gems/langchainrb"

  # Specify which files should be added to the gem when it is released.
  spec.files = Dir["LICENSE.txt", "README.md", "CHANGELOG.md", "lib/**/*"]
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]

  # dependencies
  # Not sure if we should require this as it only applies to OpenAI usecase.
  spec.add_dependency "activesupport", ">= 7.0.8"
  spec.add_dependency "baran", "~> 0.1.9"
  spec.add_dependency "colorize", "~> 1.1.0"
  spec.add_dependency "tiktoken_ruby", "~> 0.0.7"
  spec.add_dependency "json-schema", "~> 4"
  spec.add_dependency "zeitwerk", "~> 2.5"
  spec.add_dependency "pragmatic_segmenter", "~> 0.3.0"
  spec.add_dependency "to_bool", "~> 2.0.0"
  spec.add_dependency "matrix"

  # development dependencies
  spec.add_development_dependency "dotenv-rails", "~> 2.7.6"
  spec.add_development_dependency "pry-byebug", "~> 3.10.0"
  spec.add_development_dependency "yard", "~> 0.9.34"
  spec.add_development_dependency "rdiscount", "~> 2.2.7" # for github-flavored markdown in yard
  spec.add_development_dependency "vcr"
  spec.add_development_dependency "webmock"

  # optional dependencies
  spec.add_development_dependency "ai21", "~> 0.2.1"
  spec.add_development_dependency "anthropic", "~> 0.1.0"
  spec.add_development_dependency "aws-sdk-bedrockruntime", "~> 1.1"
  spec.add_development_dependency "chroma-db", "~> 0.6.0"
  spec.add_development_dependency "cohere-ruby", "~> 0.9.8"
  spec.add_development_dependency "docx", "~> 0.8.0"
  spec.add_development_dependency "elasticsearch", "~> 8.2.0"
  spec.add_development_dependency "epsilla-ruby", "~> 0.0.4"
  spec.add_development_dependency "eqn", "~> 1.6.5"
  spec.add_development_dependency "google-apis-aiplatform_v1", "~> 0.7"
  spec.add_development_dependency "google_palm_api", "~> 0.1.3"
  spec.add_development_dependency "google_search_results", "~> 2.0.0"
  spec.add_development_dependency "hnswlib", "~> 0.8.1"
  spec.add_development_dependency "hugging-face", "~> 0.3.4"
  spec.add_development_dependency "milvus", "~> 0.9.2"
  spec.add_development_dependency "llama_cpp", "~> 0.9.4"
  spec.add_development_dependency "nokogiri", "~> 1.13"
  spec.add_development_dependency "mail", "~> 2.8"
  spec.add_development_dependency "mistral-ai"
  spec.add_development_dependency "open-weather-ruby-client", "~> 0.4.0"
  spec.add_development_dependency "pg", "~> 1.5"
  spec.add_development_dependency "pgvector", "~> 0.2.1"
  spec.add_development_dependency "pdf-reader", "~> 1.4"
  spec.add_development_dependency "pinecone", "~> 0.1.6"
  spec.add_development_dependency "replicate-ruby", "~> 0.2.2"
  spec.add_development_dependency "qdrant-ruby", "~> 0.9.4"
  spec.add_development_dependency "roo", "~> 2.10.0"
  spec.add_development_dependency "ruby-openai", "~> 6.3.0"
  spec.add_development_dependency "safe_ruby", "~> 1.0.4"
  spec.add_development_dependency "sequel", "~> 5.68.0"
  spec.add_development_dependency "weaviate-ruby", "~> 0.8.10"
  spec.add_development_dependency "wikipedia-client", "~> 1.17.0"
  spec.add_development_dependency "faraday"
end
