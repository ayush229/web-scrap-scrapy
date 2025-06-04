import scrapy
from urllib.parse import urljoin, urlparse
import json
import os
import logging
import re

# IMPORTANT: You must install scrapy-splash and have a Splash server running.
# If scrapy-splash is not installed or Splash is not running, requests will fail.
try:
    from scrapy_splash import SplashRequest
except ImportError:
    # This will allow the spider to still run without Splash, but Splash-related
    # functionality will be skipped. Log a warning to indicate this.
    SplashRequest = None
    logging.getLogger(__name__).warning("scrapy-splash is not installed. Splash functionality will be unavailable.")

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

    # --- ALL Scrapy Settings Go Here Directly for standalone script ---
    # These settings will be applied to the spider and the crawler process.
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'ROBOTSTXT_OBEY': False, # Disable robots.txt obedience (use with caution!)
        'DOWNLOAD_DELAY': 1, # Add a small delay to avoid being too aggressive
        'RANDOMIZE_DOWNLOAD_DELAY': True, # Randomize the delay
        # 'HTTPERROR_ALLOWED_CODES': [403], # Keep this commented unless you specifically want to parse a 403 response body

        # --- Splash Settings: IMPORTANT ---
        # Replace 'http://YOUR_SPLASH_SERVICE_URL' with the actual URL from your Railway Splash service.
        # Example for internal Railway service: 'http://my-splash-service.railway.internal:8050'
        # Example for local testing: 'http://localhost:8050'
        'SPLASH_URL': 'http://YOUR_SPLASH_SERVICE_URL', # <<<--- CONFIGURE THIS URL

        'DOWNLOADER_MIDDLEWARES': {
            'scrapy_splash.SplashCookiesMiddleware': 723,
            'scrapy_splash.SplashMiddleware': 725,
            'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
        },
        'SPIDER_MIDDLEWARES': {
            'scrapy_splash.SplashDeduplicateArgsMiddleware': 100,
        },
        'DUPEFILTER_CLASS': 'scrapy_splash.SplashAwareDupeFilter',
        'HTTPCACHE_STORAGE': 'scrapy_splash.SplashAwareFSCacheStorage',
        # -----------------------------------------------------------------
    }
    handle_httpstatus_list = [403] # Still handle 403 to at least log it

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
        self.use_splash = kwargs.get('use_splash', 'false').lower() == 'true' # New parameter for Splash

        if not self.output_file:
            raise ValueError("output_file parameter is required for the spider.")

        self.allowed_domains_list = []
        if self.start_urls:
            parsed_start_url = urlparse(self.start_urls[0])
            self.allowed_domains_list.append(parsed_start_url.netloc)
            self.domain_to_crawl = parsed_start_url.netloc

        self.pages_crawled_count = 0
        self.results = []

        logger.info(f"Spider initialized with: Start URLs: {self.start_urls}, Scrape Mode: {self.scrape_mode}, Crawl Enabled: {self.crawl_enabled}, Max Pages: {self.max_pages}, Output File: {self.output_file}, Allowed Domain: {self.allowed_domains_list}, Use Splash: {self.use_splash}")

    def start_requests(self):
        for url in self.start_urls:
            if self.use_splash and SplashRequest: # Check if SplashRequest is available
                logger.info(f"Using Splash for URL: {url}")
                yield SplashRequest(url, self.parse_js_rendered_page, args={'wait': 2, 'render_all': 1})
            else:
                if self.use_splash and not SplashRequest:
                    logger.warning(f"Splash was requested for {url}, but scrapy-splash is not installed. Proceeding with direct HTTP request.")
                yield scrapy.Request(url=url, callback=self.parse_page, meta={'depth': 0})

    def parse_page(self, response):
        # This method handles responses NOT rendered by JavaScript (initial HTML)
        self._process_response(response, is_js_rendered=False)

    def parse_js_rendered_page(self, response):
        # This method handles responses RENDERED by JavaScript via Splash
        self._process_response(response, is_js_rendered=True)

    def _process_response(self, response, is_js_rendered):
        """
        Internal method to process the response, shared by parse_page and parse_js_rendered_page.
        """
        # Record failed responses
        if response.status != 200:
            logger.warning(f"Failed to retrieve {response.url} with status {response.status}. JS Rendered: {is_js_rendered}")
            item = PageContentItem()
            item['url'] = response.url
            item['status'] = 'failed'
            item['type'] = self.scrape_mode # Still record the scrape mode attempt
            item['data'] = None
            item['error'] = f"HTTP status {response.status}"
            self.results.append(dict(item))
            return # Stop processing this response if it's an error

        self.pages_crawled_count += 1
        logger.info(f"Processing page {self.pages_crawled_count}/{self.max_pages}: {response.url} (JS Rendered: {is_js_rendered})")

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

            # --- Advanced Content Extraction (More specific and robust) ---

            # 1. Extract JSON-like data from script tags
            for script in response.css('script::text').getall():
                json_matches = re.findall(r'{.*}', script, re.DOTALL)
                for match in json_matches:
                    try:
                        json_data = json.loads(match)
                        if isinstance(json_data, dict) and not any(key in json_data for key in ['dataLayer', 'gtm', 'analytics']):
                             content_data.append({"type": "script_json", "data": json_data})
                    except json.JSONDecodeError:
                        pass

            # 2. Extract structured content from various common semantic tags
            content_containers = response.xpath('//main|//article|//section|//div[contains(@class, "content")]|//div[contains(@id, "main")]|//div[contains(@class, "post")]|//body')

            for container in content_containers:
                section_data = {
                    "tag": container.xpath('name()').get(),
                    "id": container.xpath('@id').get(),
                    "class": container.xpath('@class').get(),
                    "heading": None,
                    "paragraphs": [],
                    "lists": [],
                    "images": [],
                    "links": [],
                    "tables": [],
                    "forms": []
                }

                heading_tags = container.xpath('.//h1|.//h2|.//h3|.//h4|.//h5|.//h6')
                if heading_tags:
                    for h in heading_tags:
                        h_text = h.xpath('string()').get(default='').strip()
                        if h_text:
                            section_data["heading"] = {"tag": h.xpath('name()').get(), "text": h_text}
                            break

                for p in container.xpath('.//p'):
                    p_text = p.xpath('string()').get(default='').strip()
                    if p_text:
                        section_data["paragraphs"].append(p_text)

                for list_tag in container.xpath('.//ul|.//ol'):
                    list_items = [li.xpath('string()').get(default='').strip() for li in list_tag.xpath('.//li')]
                    if list_items:
                        section_data["lists"].append({"type": list_tag.xpath('name()').get(), "items": list_items})

                for img in container.xpath('.//img/@src').getall():
                    abs_url = urljoin(response.url, img)
                    alt_text = container.xpath(f'.//img[@src="{img}"]/@alt').get(default='')
                    section_data["images"].append({"src": abs_url, "alt": alt_text})

                for a_href in container.xpath('.//a/@href').getall():
                    abs_url = urljoin(response.url, a_href).split('#')[0].rstrip('/')
                    link_text = container.xpath(f'.//a[@href="{a_href}"]/text()').get(default='').strip()
                    if abs_url and link_text:
                        section_data["links"].append({"href": abs_url, "text": link_text})
                    extracted_links.append(abs_url)

                for table in container.xpath('.//table'):
                    table_data = []
                    headers = [th.xpath('string()').get(default='').strip() for th in table.xpath('.//th')]
                    if headers:
                        table_data.append(headers)
                    for row in table.xpath('.//tr'):
                        cells = [td.xpath('string()').get(default='').strip() for td in row.xpath('.//td')]
                        if cells:
                            table_data.append(cells)
                    if table_data:
                        section_data["tables"].append(table_data)

                for form in container.xpath('.//form'):
                    form_details = {
                        "action": urljoin(response.url, form.xpath('@action').get(default='')),
                        "method": form.xpath('@method').get(default='GET').upper(),
                        "inputs": []
                    }
                    for input_field in form.xpath('.//input|.//textarea|.//select'):
                        input_type = input_field.xpath('@type').get(default='text')
                        input_name = input_field.xpath('@name').get(default='')
                        input_id = input_field.xpath('@id').get(default='')
                        input_value = input_field.xpath('@value').get(default='')
                        form_details["inputs"].append({
                            "tag": input_field.xpath('name()').get(),
                            "type": input_type,
                            "name": input_name,
                            "id": input_id,
                            "value": input_value
                        })
                    if form_details["inputs"]:
                        section_data["forms"].append(form_details)

                if any(section_data[key] for key in ["heading", "paragraphs", "lists", "images", "links", "tables", "forms"]):
                    content_data.append(section_data)

            item['data'] = {"sections": content_data}
            item['error'] = None

            # --- Crawling Logic ---
            if self.crawl_enabled and self.pages_crawled_count < self.max_pages:
                current_depth = response.meta.get('depth', 0)
                next_depth = current_depth + 1

                for link in extracted_links:
                    parsed_link = urlparse(link)
                    if parsed_link.scheme in ['http', 'https'] and parsed_link.netloc == self.domain_to_crawl:
                        if link not in response.request.meta.get('crawled_urls', set()):
                            if self.use_splash and SplashRequest: # Check if SplashRequest is available
                                yield SplashRequest(
                                    url=link,
                                    callback=self.parse_js_rendered_page,
                                    args={'wait': 2, 'render_all': 1},
                                    meta={'depth': next_depth, 'crawled_urls': response.request.meta.get('crawled_urls', set()) | {link}}
                                )
                            else:
                                if self.use_splash and not SplashRequest:
                                    logger.warning(f"Splash was requested for {link}, but scrapy-splash is not installed. Proceeding with direct HTTP request.")
                                yield scrapy.Request(
                                    url=link,
                                    callback=self.parse_page,
                                    meta={'depth': next_depth, 'crawled_urls': response.request.meta.get('crawled_urls', set()) | {link}}
                                )

        self.results.append(dict(item))

    def closed(self, reason):
        logger.info(f"Spider finished. Reason: {reason}. Total pages processed: {self.pages_crawled_count}")
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        final_output = {
            "spider_name": self.name,
            "start_urls": self.start_urls,
            "scrape_mode": self.scrape_mode,
            "crawl_enabled": self.crawl_enabled,
            "max_pages": self.max_pages,
            "use_splash": self.use_splash,
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
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    # When not using a full Scrapy project, get_project_settings() might return empty or default settings.
    # We will rely on the custom_settings defined within the spider itself.
    from scrapy.utils.project import get_project_settings # Still imported, but not used to load external file
    import sys
    import argparse

    # When running standalone, you don't load external settings,
    # but CrawlerProcess still expects an object. We'll pass an empty dict
    # or rely on the spider's custom_settings.
    # The spider's custom_settings will be picked up automatically by CrawlerProcess.
    # No need to manually set settings here from a file if you don't have one.
    settings = {} # An empty settings object is fine here

    parser = argparse.ArgumentParser(description='Run GeneralPurposeSpider.')
    parser.add_argument('--start_urls', required=True, help='Comma-separated URLs to start scraping from.')
    parser.add_argument('--scrape_mode', default='beautify', help='Scraping mode: "raw" or "beautify".')
    parser.add_argument('--crawl_enabled', default='false', help='Enable crawling: "true" or "false".')
    parser.add_argument('--max_pages', type=int, default=50, help='Maximum pages to crawl if crawling is enabled.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSON file.')
    parser.add_argument('--use_splash', default='false', help='Use Scrapy-Splash for JavaScript rendering: "true" or "false".')

    args = parser.parse_args()

    process = CrawlerProcess(settings) # Pass the empty settings here
    process.crawl(GeneralPurposeSpider,
                    start_urls=args.start_urls,
                    scrape_mode=args.scrape_mode,
                    crawl_enabled=args.crawl_enabled,
                    max_pages=args.max_pages,
                    output_file=args.output_file,
                    use_splash=args.use_splash)

    process.start()
