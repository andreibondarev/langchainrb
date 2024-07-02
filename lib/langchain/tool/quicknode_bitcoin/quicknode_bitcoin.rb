# frozen_string_literal: true

module Langchain::Tool
  class QuicknodeBitcoin < Base
    #
    # Wrapper around QuickNode's Bitcoin Blockchain API
    #
    # Usage:
    #     quicknode_bitcoin = Langchain::Tool::QuicknodeBitcoin.new(url: ENV["QUICKNODE_BITCOIN_URL"])
    #
    NAME = "quicknode_bitcoin"
    ANNOTATIONS_PATH = Langchain.root.join("./langchain/tool/#{NAME}/#{NAME}.json").to_path

    # Initializes a new Quicknode Bitcoin instance.
    #
    # @param url [String] The URL of the Quicknode Bitcoin instance.
    # @return [Langchain::Tool::QuicknodeBitcoin] QuicknodeBitcoin tool
    def initialize(url:)
      @base_url = url
      # Initialize a Net::HTTP object
      @https = Net::HTTP.new(URI(@base_url).host, URI(@base_url).port)
      @https.use_ssl = true
    end

    # Decodes the raw transaction and provides chain information in JSON format.
    #
    # @param hexstring [String] The transaction hex string to be decoded.
    # @param iswitness [Boolean] Indicates whether the transaction hex is a serialized witness transaction. Default is false.
    # @return [Hash] JSON response containing the decoded transaction information
    def decoderawtransaction(hexstring:, iswitness: false)
      send_request("decoderawtransaction", [hexstring, iswitness])
    end

    # Decodes a hex-encoded script.
    #
    # @param hexstring [String] The hex-encoded script to be decoded.
    # @return [Hash] JSON response containing the decoded script information.
    def decodescript(hexstring:)
      send_request("decodescript", [hexstring])
    end

    # Estimates the smart fee per kilobyte to be paid for a transaction and also returns the number of blocks for which the estimate is valid.
    #
    # @param conf_target [Numeric] Confirmation target in blocks.
    # @param estimate_mode [String] The fee estimate mode, defaults to "CONSERVATIVE".
    # @return [Hash] JSON response containing the estimated fee information.
    def estimatesmartfee(conf_target:, estimate_mode: "CONSERVATIVE")
      send_request("estimatesmartfee", [conf_target, estimate_mode])
    end

    # Returns the hash of the tip block in the most-work fully-validated chain.
    #
    # @return [Hash] JSON response containing the hash of the tip block.
    def getbestblockhash
      send_request("getbestblockhash")
    end

    # Returns an object that contains information regarding blockchain processing in different states.
    #
    # @return [Hash] JSON response containing information regarding blockchain processing.
    def getblockchaininfo
      send_request("getblockchaininfo")
    end

    # Returns the height of the fully-validated chain.
    #
    # @return [Hash] JSON response containing the height of the fully-validated chain.
    def getblockcount
      send_request("getblockcount")
    end

    # Returns the hash of the block provided its height.
    #
    # @param height [Numeric] The height index of the block in the blockchain.
    # @return [Hash] JSON response containing the hash of the block at the specified height.
    def getblockhash(height:)
      send_request("getblockhash", [height])
    end

    # Returns the header of the block given its hash.
    #
    # @param blockhash [String] The hash of the block.
    # @param verbose [Boolean] Default is true. Set to true for a JSON object and false for hex-encoded data.
    # @return [Hash] If verbose is true, returns a JSON object containing the block header. Otherwise, returns the hex-encoded data.
    def getblockheader(blockhash:, verbose: true)
      send_request("getblockheader", [blockhash, verbose])
    end

    # Calculates per block statistics for a given window.
    #
    # @param hash_or_height [String, Numeric] The block hash or height of the target block.
    # @param stats [Array] JSON array of values to filter from.
    # @return [Hash] JSON response containing per block statistics for the given block.
    def getblockstats(hash_or_height:, stats: [])
      send_request("getblockstats", [hash_or_height, stats].compact)
    end

    # Returns information about the block.
    #
    # @param blockhash [String] The hash of the block.
    # @param verbosity [Numeric] Default is 1. Set to 0 for hex-encoded data, 1 for a JSON object, and 2 for a JSON object with transaction data.
    # @return [Hash] JSON response containing information about the block.
    def getblock(blockhash:, verbosity: 1)
      send_request("getblock", [blockhash, verbosity])
    end

    # Returns information about all known chaintips in the block tree, including the main chain as well as orphaned branches.
    #
    # @return [Hash] JSON response containing information about all known chaintips in the block tree.
    def getchaintips
      send_request("getchaintips")
    end

    # Calculates data about the total number and rate of transactions in the chain.
    #
    # @param nblocks [Numeric] The window's size in number of blocks. Default is one month.
    # @param blockhash [String] The hash of the block that ends the window. Default is chain tip.
    # @return [Hash] JSON response containing data about the total number and rate of transactions in the chain.
    def getchaintxstats(nblocks: nil, blockhash: nil)
      send_request("getchaintxstats", [nblocks, blockhash].compact)
    end

    # Returns the connection count to other nodes.
    #
    # @return [Hash] JSON response containing the connection count to other nodes.
    def getconnectioncount
      send_request("getconnectioncount")
    end

    # Returns the proof-of-work difficulty as a multiple of the minimum difficulty.
    #
    # @return [Hash] JSON response containing the proof-of-work difficulty.
    def getdifficulty
      send_request("getdifficulty")
    end

    # Returns the status of one or all available indexes actively running in the node.
    #
    # @param index_name [String, nil] Filters results for an index with a specific name.
    # @return [Hash] JSON response containing the status of one or all available indexes actively running in the node.
    def getindexinfo(index_name: nil)
      send_request("getindexinfo", [index_name].compact)
    end

    # Returns an object that contains information regarding memory usage.
    #
    # @return [Hash] JSON response containing information regarding memory usage.
    def getmemoryinfo
      send_request("getmemoryinfo")
    end

    # Returns all in-mempool ancestors for a transaction in the mempool.
    #
    # @param txid [String] The transaction id, and it must be in mempool.
    # @param verbose [Boolean] Set to true for a JSON object and false for an array of transaction ids. Default is false.
    # @return [Hash] JSON response containing all in-mempool ancestors for the specified transaction.
    def getmempoolancestors(txid:, verbose: false)
      send_request("getmempoolancestors", [txid, verbose])
    end

    # Returns all in-mempool descendants for a transaction in the mempool.
    #
    # @param txid [String] The transaction id, and it must be in mempool.
    # @param verbose [Boolean] Set to true for a JSON object and false for an array of transaction ids. Default is false.
    # @return [Hash] JSON response containing all in-mempool descendants for the specified transaction.
    def getmempooldescendants(txid:, verbose: false)
      send_request("getmempooldescendants", [txid, verbose])
    end

    # Returns information about the active state of the TX memory pool.
    #
    # @return [Hash] JSON response containing information about the active state of the TX memory pool.
    def getmempoolinfo
      send_request("getmempoolinfo")
    end

    # Returns all transaction ids in memory pool.
    #
    # @param verbose [Boolean] Set to true for a JSON object and false for an array of transaction ids. Default is false.
    # @param mempool_sequence [Boolean] Set to true to return a JSON object with transaction list and mempool sequence number attached. Default is false.
    # @return [Hash] JSON response containing all transaction ids in memory pool.
    def getrawmempool(verbose: false, mempool_sequence: false)
      send_request("getrawmempool", [verbose, mempool_sequence])
    end

    # Returns the raw transaction data.
    #
    # @param txid [String] The transaction id.
    # @param verbose [Integer] A numeric parameter that can take one of the following values: '0' for hex-encoded data, '1' for JSON object, and '2' for JSON object with fee and prevout. Default is 0.
    # @param blockhash [String, nil] The block in which to look for the transaction.
    # @return [Hash] JSON response containing the raw transaction data.
    def getrawtransaction(txid:, verbose: 0, blockhash: nil)
      send_request("getrawtransaction", [txid, verbose, blockhash].compact)
    end

    # Ensures that the transactions are within block and returns proof of transaction inclusion.
    #
    # @param txids [Array] An array of transaction hashes.
    # @param blockhash [String, nil] If specified, looks for txid in the block with this hash.
    # @return [Hash] JSON response containing proof of transaction inclusion.
    def gettxoutproof(txids:, blockhash: nil)
      send_request("gettxoutproof", [txids, blockhash].compact)
    end

    # Returns information about the unspent transaction output set.
    #
    # @param hash_type [String] It tells about which UTXO set hash should be calculated. Possible values: "hash_serialized_3", "none", "muhash". Default is "hash_serialized_3".
    # @param hash_or_height [String, Integer, nil] The block hash or height of the target height.
    # @param use_index [Boolean] Use coinstatsindex if available. Default is true.
    # @return [Hash] JSON response containing information about the unspent transaction output set.
    def gettxoutsetinfo(hash_type: "hash_serialized_3", hash_or_height: nil, use_index: true)
      send_request("gettxoutsetinfo", [hash_type, hash_or_height, use_index].compact)
    end

    # Returns details about an unspent transaction output.
    #
    # @param txid [String] The transaction id.
    # @param n [Numeric] Vout number.
    # @param include_mempool [Boolean] Whether to include the mempool. Default is true.
    # @return [Hash] JSON response containing details about the unspent transaction output.
    def gettxout(txid:, n:, include_mempool: true)
      send_request("gettxout", [txid, n, include_mempool])
    end

    # Submits a raw transaction (serialized, hex-encoded) to a node.
    #
    # @param hexstring [String] The transaction hex string.
    # @param maxfeerate [Numeric/String] It rejects transactions with a fee rate higher than the specified value. Default is 0.10. It can be set to 0 to accept any fee rate.
    # @return [Hash] JSON response containing the transaction id if the transaction was accepted.
    def sendrawtransaction(hexstring:, maxfeerate: 0.10)
      send_request("sendrawtransaction", [hexstring, maxfeerate])
    end

    # Submits a package of raw transactions (serialized, hex-encoded) to the local node.
    #
    # @param package [Array] An array of raw transactions.
    # @return [Hash] JSON response containing the result of submitting the package.
    def submitpackage(package:)
      send_request("submitpackage", [package])
    end

    # Returns the output of mempool acceptance tests, indicating if the mempool will accept serialized, hex-encoded raw transactions.
    #
    # @param rawtxs [Array] An array of raw transactions in the form of a hex string.
    # @param maxfeerate [String/Numeric] It rejects transactions with a fee rate higher than the specified value. Default is 0.10. It can be set to 0 to accept any fee rate.
    # @return [Hash] JSON response containing the output of mempool acceptance tests.
    def testmempoolaccept(rawtxs:, maxfeerate: 0.10)
      send_request("testmempoolaccept", [rawtxs, maxfeerate])
    end

    # Returns information about the given bitcoin address.
    #
    # @param address [String] The bitcoin address to validate.
    # @return [Hash] JSON response containing information about the given bitcoin address.
    def validateaddress(address:)
      send_request("validateaddress", [address])
    end

    # Verifies a signed message.
    #
    # @param address [String] The bitcoin address to use for the signature.
    # @param signature [String] The signature provided by the signer in base64 encoding.
    # @param message [String] The message that was signed.
    # @return [Hash] JSON response indicating whether the signature is valid for the given message and address.
    def verifymessage(address:, signature:, message:)
      send_request("verifymessage", [address, signature, message])
    end

    private

    # Sends an HTTP request to the API endpoint.
    #
    # @param method [String] The method name to be called.
    # @param params [Array] The parameters to be passed to the method.
    # @return [Hash] JSON response from the API.
    def send_request(method, params = [])
      payload = {id: 1, jsonrpc: "2.0", method: method, params: params}
      request = Net::HTTP::Post.new(URI(@base_url))
      request["Content-Type"] = "application/json"
      request.body = JSON.dump(payload)

      response = @https.request(request)

      case response
      when Net::HTTPSuccess
        response.read_body
      else
        raise "HTTP request failed: #{response.code} - #{response.message}"
      end
    end
  end
end
