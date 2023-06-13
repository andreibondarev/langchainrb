# frozen_string_literal: true

RSpec.describe Langchain::ContextualLogger do
  let(:logger) { double(:logger) }

  subject { described_class.new(logger) }

  context "without extra context" do
    it "#info handles line without context" do
      expect(logger).to receive(:info).with(
        <<~LINE.strip
          #{"[LangChain.rb]".yellow} #{"Hello World".white}
        LINE
      )
      subject.info("Hello World")
    end

    it "#warn handles line without context" do
      expect(logger).to receive(:warn).with(
        <<~LINE.strip
          #{"[LangChain.rb]".yellow} #{"Hello World".colorize(color: :yellow, mode: :bold)}
        LINE
      )
      subject.warn("Hello World")
    end
  end

  context "with extra context" do
    it "#info handles line without context" do
      expect(logger).to receive(:info).with(
        <<~LINE.strip
          #{"[LangChain.rb]".yellow} #{"[Langchain::Vectorsearch::Pgvector]".blue}: #{"Hello World".white}
        LINE
      )
      subject.info("Hello World", for: Langchain::Vectorsearch::Pgvector)
    end

    it "#warn handles line without context" do
      expect(logger).to receive(:warn).with(
        <<~LINE.strip
          #{"[LangChain.rb]".yellow} #{"[Langchain::Vectorsearch::Pgvector]".blue}: #{"Hello World".colorize(color: :yellow, mode: :bold)}
        LINE
      )
      subject.warn("Hello World", for: Langchain::Vectorsearch::Pgvector)
    end
  end
end
