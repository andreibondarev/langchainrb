# frozen_string_literal: true

RSpec.describe Langchain::Tool::QuicknodeBitcoin do
  subject {
    described_class.new(url: "https://logical-crimson-mountain.btc-testnet.quiknode.pro/ddfdf444f4f4f4f4f/")
  }

  let(:response_decoderawtransaction) do
    {
      "txid" => "4fe3d5cc2b7f10c92ff1845ae9c183e2c4747b3f8dd5c97aafab76822c3997d5",
      "hash" => "7e00f1730c09cb8b0ae569e6d70f5f274a3f2c5d6358b8c330514f650abdfb32",
      "size" => 226,
      "vsize" => 144,
      "weight" => 574,
      "version" => 1,
      "locktime" => 0,
      "vin" => [],
      "vout" => []
    }
  end

  let(:response_estimatesmartfee) do
    {
      feerate: 0.00099999,
      blocks: 6
    }
  end

  let(:response_getmemoryinfo) do
    {
      loaded: true,
      size: 7095,
      bytes: 1169532,
      usage: 9697072,
      total_fee: 0.21813004,
      maxmempool: 2048000000,
      mempoolminfee: 0.00001,
      minrelaytxfee: 0.00001,
      incrementalrelayfee: 0.00001,
      unbroadcastcount: 0,
      fullrbf: true
    }
  end

  before do
    allow_any_instance_of(Langchain::Tool::QuicknodeBitcoin).to receive(:send_request)
      .with("decoderawtransaction", anything)
      .and_return(response_decoderawtransaction)

    allow_any_instance_of(Langchain::Tool::QuicknodeBitcoin).to receive(:send_request)
      .with("estimatesmartfee", anything)
      .and_return(response_estimatesmartfee)

    allow_any_instance_of(Langchain::Tool::QuicknodeBitcoin).to receive(:send_request)
      .with("getmemoryinfo")
      .and_return(response_getmemoryinfo)
  end

  describe "#decoderawtransaction" do
    let(:hexstring) { "0100000001eaefefbd1f687ef4e861804aed59ef05e743ea85f432cc146f325d759a026ce6010000006a4730440220718954e28983c875858b5a0094df4607ce2e7c6e9ffea47f3876792b01755c1202205e2adc7c32ff64aaef6d26045f96181e8741e560b6f3a8ef2f4ffd2892add656012103142355370728640592109c3d2bf5592020a6b9226303c8bc98ab2ebcadf057abffffffff02005a6202000000001976a914fe7e0711287688b33b9a5c239336c4700db34e6388ac10ca0f24010000001976a914af92ad98c7f77559f96430dfef2a6805b87b24f888ac00000000" }
    let(:expected_result) { response_decoderawtransaction }

    it "decodes the raw transaction and returns chain information" do
      result = subject.decoderawtransaction(hexstring: hexstring)
      expect(result).to eq(expected_result)
    end
  end

  describe "#estimatesmartfee" do
    let(:conf_target) { 6 }
    let(:expected_result) { response_estimatesmartfee }

    it "estimates the smart fee per kilobyte" do
      result = subject.estimatesmartfee(conf_target: conf_target)
      expect(result).to eq(expected_result)
    end
  end

  describe "#getmemoryinfo" do
    let(:expected_result) { response_getmemoryinfo }

    it "returns information regarding memory usage" do
      result = subject.getmemoryinfo
      expect(result).to eq(expected_result)
    end
  end
end
