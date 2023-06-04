module Langchain::Tool
  class Database < Base
    description <<~DESC
      Useful for getting the result of a database query.

      The input to this tool should be valid SQL.
    DESC

    # Establish a database connection
    # example: 'postgres://user:password@localhost:5432/db_name'
    def initialize(db_connection_string)
      depends_on "sequel"
      require "sequel"
      require "sequel/extensions/schema_dumper"

      raise StandardError, "db_connection_string parameter cannot be blank" if db_connection_string.empty?

      @db = Sequel.connect(db_connection_string)
      @db.extension :schema_dumper
    end

    def schema
      Langchain.logger.info("[#{self.class.name}]".light_blue + ": Dumping schema")
      @db.dump_schema_migration(same_db: true, indexes: false) unless @db.adapter_scheme == :mock
    end

    # Evaluates a sql expression
    # @param input [String] sql expression
    # @return [Array] results
    def execute(input:)
      Langchain.logger.info("[#{self.class.name}]".light_blue + ": Executing \"#{input}\"")
      begin
        @db[input].to_a
      rescue Sequel::DatabaseError => e
        Langchain.logger.error("[#{self.class.name}]".light_red + ": #{e.message}")
      end
    end
  end
end
