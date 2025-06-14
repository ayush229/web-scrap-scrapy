import scrapy
from urllib.parse import urljoin, urlparse
import json
import os
import logging

# Configure logging for the spider
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the Scrapy Item to hold scraped data
class PageContentItem(scrapy.Item):
    url = scrapy.Field()
    status = scrapy.Field()
    type = scrapy.Field()
    data = scrapy.Field() # For raw HTML or structured content
    error = scrapy.Field()

class GeneralPurposeSpider(scrapy.Spider):
    name = 'general_purpose_spider'

    # --- Added/Modified for 403 Error Handling ---
    # Set a common User-Agent to mimic a browser
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'ROBOTSTXT_OBEY': False, # Disable robots.txt obedience (use with caution!)
        'DOWNLOAD_DELAY': 1, # Add a small delay to avoid being too aggressive
        'RANDOMIZE_DOWNLOAD_DELAY': True, # Randomize the delay
        # 'HTTPERROR_ALLOWED_CODES': [403], # Uncomment this if you want parse_page to receive 403 responses
    }
    # If you want to process 403 responses within parse_page, add this line:
    handle_httpstatus_list = [403]
    # -----------------------------------------------

    def __init__(self, *args, **kwargs):
        super(GeneralPurposeSpider, self).__init__(*args, **kwargs)
        self.start_urls = kwargs.get('start_urls', '').split(',')
        self.start_urls = [u.strip() for u in self.start_urls if u.strip()]

        if not self.start_urls:
            raise ValueError("No start_urls provided to the spider.")

        self.scrape_mode = kwargs.get('scrape_mode', 'beautify') # 'raw' or 'beautify'
        self.crawl_enabled = kwargs.get('crawl_enabled', 'false').lower() == 'true' # 'true' or 'false'
        self.max_pages = int(kwargs.get('max_pages', 50))
        self.output_file = kwargs.get('output_file')

        if not self.output_file:
            raise ValueError("output_file parameter is required for the spider.")

        # Determine allowed_domains from the first URL for crawling
        self.allowed_domains_list = []
        if self.start_urls:
            parsed_start_url = urlparse(self.start_urls[0])
            self.allowed_domains_list.append(parsed_start_url.netloc)
            self.domain_to_crawl = parsed_start_url.netloc

        self.pages_crawled_count = 0
        self.results = [] # To accumulate all results before writing to file

        logger.info(f"Spider initialized with: Start URLs: {self.start_urls}, Scrape Mode: {self.scrape_mode}, Crawl Enabled: {self.crawl_enabled}, Max Pages: {self.max_pages}, Output File: {self.output_file}, Allowed Domain: {self.allowed_domains_list}")

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse_page, meta={'depth': 0})

    def parse_page(self, response):
        # Check if the response status is not 200 (OK)
        if response.status != 200:
            logger.warning(f"Failed to retrieve {response.url} with status {response.status}")
            item = PageContentItem()
            item['url'] = response.url
            item['status'] = 'failed'
            item['type'] = self.scrape_mode # Still record the scrape mode attempt
            item['data'] = None
            item['error'] = f"HTTP status {response.status}"
            self.results.append(dict(item))
            return # Stop processing this response if it's an error

        self.pages_crawled_count += 1
        logger.info(f"Processing page {self.pages_crawled_count}/{self.max_pages}: {response.url}")

        item = PageContentItem()
        item['url'] = response.url
        item['status'] = 'success'
        item['type'] = self.scrape_mode

        if self.scrape_mode == 'raw':
            item['data'] = response.text
            item['error'] = None
        elif self.scrape_mode == 'beautify':
            content_data = []
            extracted_links = []

            # Find common content containers (similar to BeautifulSoup's find_all(['section', 'div', 'article']))
            # Using XPath for robustness, but CSS selectors can also be used.
            # This selection is a general attempt; real-world sites might need more specific selectors.
            content_sections = response.xpath('//body//*[self::section or self::div or self::article or self::main]')

            if not content_sections:
                # Fallback to general body content if specific sections are not found
                content_sections = response.xpath('//body')

            for sec in content_sections:
                section_data = {
                    "heading": None,
                    "content": [],
                    "images": [],
                    "links": []
                }

                # Extract headings
                heading_tags = sec.xpath('.//h1|.//h2|.//h3|.//h4|.//h5|.//h6')
                if heading_tags:
                    # Take the first non-empty heading
                    for h in heading_tags:
                        h_text = h.xpath('string()').get(default='').strip()
                        if h_text:
                            section_data["heading"] = {"tag": h.xpath('name()').get(), "text": h_text}
                            break # Found a heading for this section

                # Extract paragraphs and list items
                paragraphs_and_lists = sec.xpath('.//p|.//li')
                for p_or_li in paragraphs_and_lists:
                    text = p_or_li.xpath('string()').get(default='').strip()
                    if text:
                        section_data["content"].append(text)

                # Extract images
                for img in sec.xpath('.//img/@src').getall():
                    abs_url = urljoin(response.url, img)
                    section_data["images"].append(abs_url)

                # Extract links for crawling if enabled
                for a_href in sec.xpath('.//a/@href').getall():
                    # Construct absolute URL and remove fragment (hash anchor)
                    abs_url = urljoin(response.url, a_href).split('#')[0].rstrip('/')
                    section_data["links"].append(abs_url)
                    extracted_links.append(abs_url) # Collect all links for potential follow-up

                # Only add sections that have actual content
                if section_data["heading"] or section_data["content"] or section_data["images"] or section_data["links"]:
                    content_data.append(section_data)

            item['data'] = {"sections": content_data}
            item['error'] = None

            # --- Crawling Logic ---
            if self.crawl_enabled and self.pages_crawled_count < self.max_pages:
                current_depth = response.meta.get('depth', 0)
                next_depth = current_depth + 1

                for link in extracted_links:
                    parsed_link = urlparse(link)
                    # Check if the link is within the allowed domain and is HTTP/HTTPS
                    if parsed_link.scheme in ['http', 'https'] and parsed_link.netloc == self.domain_to_crawl:
                        # Use a set in meta to track URLs seen in the current crawl to avoid redundant requests
                        # This 'crawled_urls' set should ideally be managed by Scrapy's DuplicatesFilter,
                        # but adding it here for explicit control within the spider's logic if needed.
                        # Scrapy's scheduler handles duplicates effectively by default.
                        if link not in response.request.meta.get('crawled_urls', set()): # Avoid re-requesting in the same crawl
                            # We need to pass the scrape_mode and other args to the next request
                            yield scrapy.Request(
                                url=link,
                                callback=self.parse_page,
                                meta={'depth': next_depth, 'crawled_urls': response.request.meta.get('crawled_urls', set()) | {link}}
                            )

        self.results.append(dict(item)) # Convert Item to dict and store

    def closed(self, reason):
        # This method is called when the spider finishes or is closed
        logger.info(f"Spider finished. Reason: {reason}. Total pages processed: {self.pages_crawled_count}")
        # Ensure the directory exists before writing
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Write all results to the output file
        final_output = {
            "spider_name": self.name,
            "start_urls": self.start_urls,
            "scrape_mode": self.scrape_mode,
            "crawl_enabled": self.crawl_enabled,
            "max_pages": self.max_pages,
            "total_pages_crawled": self.pages_crawled_count,
            "results": self.results,
            "closure_reason": reason
        }
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
            logger.info(f"Spider results successfully written to {self.output_file}")
        except Exception as e:
            logger.error(f"Error writing spider results to file {self.output_file}: {e}")

# This part is for running the spider from an external script
# (e.g., your Flask app). It will not be directly executed when `scrapy_spider.py`
# is imported by Scrapy itself.
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings
    import sys
    import argparse

    # Basic setup for running directly, mainly for testing
    settings = get_project_settings()
    # You might want to override some settings for direct run, e.g.:
    # settings.set('LOG_LEVEL', 'INFO')
    # settings.set('ROBOTSTXT_OBEY', False) # Be cautious with this in production

    # For direct execution and testing, let's set up argument parsing
    parser = argparse.ArgumentParser(description='Run GeneralPurposeSpider.')
    parser.add_argument('--start_urls', required=True, help='Comma-separated URLs to start scraping from.')
    parser.add_argument('--scrape_mode', default='beautify', help='Scraping mode: "raw" or "beautify".')
    parser.add_argument('--crawl_enabled', default='false', help='Enable crawling: "true" or "false".')
    parser.add_argument('--max_pages', type=int, default=50, help='Maximum pages to crawl if crawling is enabled.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSON file.')

    args = parser.parse_args()

    # Pass all arguments to the spider
    process = CrawlerProcess(settings)
    process.crawl(GeneralPurposeSpider,
                    start_urls=args.start_urls,
                    scrape_mode=args.scrape_mode,
                    crawl_enabled=args.crawl_enabled,
                    max_pages=args.max_pages,
                    output_file=args.output_file)

    process.start() # The script will block here until the crawl finishes
