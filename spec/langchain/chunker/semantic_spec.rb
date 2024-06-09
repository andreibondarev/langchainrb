# frozen_string_literal: true

RSpec.describe Langchain::Chunker::Semantic do
  let(:source) { "spec/fixtures/loaders/random_facts.txt" }
  let(:text) { File.read(source) }
  let(:llm) { Langchain::LLM::OpenAI.new(api_key: "123") }

  subject { described_class.new(text, llm: llm) }

  describe "#chunks" do
    before do
      allow(llm).to receive_message_chain(:complete, :completion).and_return(
        " Here are the paragraphs split by topic:\n\n---\nOn July 1, 1867, Canada became a self-governing dominion of Great Britain and a federation of four provinces: Nova Scotia; New Brunswick; Ontario; and Quebec.\nThe anniversary of this date was called Dominion Day until 1982.\n---\nNicotine is named after the tobacco plant Nicotiana tabacum, which in turn is named after the French ambassador in Portugal, Jean Nicot de Villemain, who sent tobacco and seeds to Paris in 1560, and who promoted their medicinal use.\n---\nWell, it turns out that killer whales have been known to prey on sharks. Orcas have also been known to eat mako sharks and several other species. \nWhen hunting sharks, killer whales always end up flipping the shark upside down, regardless of how the attack starts.\n---\nThe Lincoln cent or Lincoln penny is a cent coin (or penny) (1/100 of a dollar) that has been struck by the United States Mint since 1909.\nThe obverse or heads side was designed by Victor David Brenner, as was the original reverse.\n---\nThe Undertaker and John Cena are the only two Superstars that have entered the Royal Rumble Match at No. 30 and won.\nThese victories also occurred back-to-back in 2007 and 2008, respectively.\n---\nThe Gobi is a large desert region in northern China and southern Mongolia.\nThe desert basins of the Gobi are bounded by the Altai mountains and the grasslands and steppes of Mongolia on the north, by the Tibetan Plateau to the southwest, and by the North China Plain to the southwest."
      )
    end

    it "returns an array of chunks" do
      expect(subject.chunks.count).to eq(6)
      expect(subject.chunks).to all(be_a(Langchain::Chunk))
    end

    it "sets source for each chunk" do
      expect(subject.chunks(source: source).map(&:source).uniq).to eq([source])
    end
  end
end
