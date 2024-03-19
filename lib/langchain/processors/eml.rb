require "uri"

module Langchain
  module Processors
    class Eml < Base
      EXTENSIONS = [".eml"]
      CONTENT_TYPES = ["message/rfc822"]

      def initialize(*)
        depends_on "mail"
      end

      # Parse the document and return the cleaned text
      # @param [File] data
      # @return [String]
      def parse(data)
        mail = Mail.read(data.path)
        text_content = extract_text_content(mail)
        clean_content(text_content)
      end

      private

      # Extract text content from the email, preferring plaintext over HTML
      def extract_text_content(mail)
        text_content = ""
        text_content += "From: #{mail.from}\n" \
                        "To: #{mail.to}\n" \
                        "Cc: #{mail.cc}\n" \
                        "Bcc: #{mail.bcc}\n" \
                        "Subject: #{mail.subject}\n" \
                        "Date: #{mail.date}\n\n"
        if mail.multipart?
          mail.parts.each do |part|
            if part.content_type.start_with?("text/plain")
              text_content += part.body.decoded.force_encoding("UTF-8").strip + "\n"
            elsif part.content_type.start_with?("multipart/alternative", "multipart/mixed")
              text_content += extract_text_content(part) + "\n" # Recursively extract from multipart
            elsif part.content_type.start_with?("message/rfc822")
              # Handle embedded .eml parts as separate emails
              embedded_mail = Mail.read_from_string(part.body.decoded)
              text_content += "--- Begin Embedded Email ---\n"
              text_content += extract_text_content(embedded_mail) + "\n"
              text_content += "--- End Embedded Email ---\n"
            end
          end
        elsif mail.content_type.start_with?("text/plain")
          text_content = mail.body.decoded.force_encoding("UTF-8").strip
        end
        text_content
      end

      # Clean and format the extracted content
      def clean_content(content)
        content
          .gsub(/\[cid:[^\]]+\]/, "") # Remove embedded image references
          .gsub(URI::DEFAULT_PARSER.make_regexp(%w[http https])) { |match| "<#{match}>" } # Format URLs
          .gsub(/\r\n?/, "\n") # Normalize line endings to Unix style
          .gsub(/[\u200B-\u200D\uFEFF]/, "") # Remove zero width spaces and similar characters
          .gsub(/<\/?[^>]+>/, "") # Remove any HTML tags that might have sneaked in
      end
    end
  end
end
