import os
import json
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from typing import List, Dict, Optional, Tuple
from slugify import slugify
from urllib.parse import urlparse, urljoin
import trafilatura
from concurrent.futures import ThreadPoolExecutor
from anthropic import Anthropic
import math
from minio import Minio
from datetime import datetime
import pymysql
import uuid
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from PIL import Image
import io
import requests
import re


# Import our GoogleSheetsManager
from sheets_setup import GoogleSheetsManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('article_processor.log'),
        logging.StreamHandler()
    ]
)

class DatabaseHandler:
    def __init__(self, database_url: str):
        parsed = urlparse(database_url)
        self.connection_params = {
            'host': parsed.hostname,
            'user': parsed.username,
            'password': parsed.password,
            'db': parsed.path[1:],  # Remove leading slash
            'port': parsed.port
        }

    def connect(self):
        return pymysql.connect(**self.connection_params)

    def create_or_update_article(self, article_data: Dict) -> bool:
        """Create or update article in the database"""
        try:
            with self.connect() as connection:
                with connection.cursor() as cursor:
                    # Generate a CUID-like id for new articles
                    import time
                    import random
                    import string
                    
                    # Generate CUID-style ID
                    timestamp = int(time.time() * 1000)
                    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
                    new_id = f"clr{timestamp:x}{random_str}"

                    # Check if article exists
                    cursor.execute("SELECT id FROM Article WHERE slug = %s", (article_data["slug"],))
                    existing_article = cursor.fetchone()

                    if existing_article:
                        # Update existing article
                        sql = """
                        UPDATE Article 
                        SET 
                            title = %s,
                            content = %s,
                            markdownUrl = %s,
                            excerpt = %s,
                            coverImage = %s,
                            readingTime = %s,
                            wordCount = %s,
                            status = %s,
                            metaTitle = %s,
                            metaDescription = %s,
                            featured = %s,
                            spotlight = %s,
                            evergreen = %s,
                            sponsored = %s,
                            sponsorName = %s,
                            partnerContent = %s,
                            affiliate = %s,
                            crowdsourced = %s,
                            premium = %s,
                            hasVideo = %s,
                            hasAudio = %s,
                            hasGallery = %s,
                            updatedAt = NOW()
                        WHERE slug = %s
                        """
                        cursor.execute(sql, (
                            article_data["title"],
                            article_data["content"],
                            article_data["markdownUrl"],
                            article_data["excerpt"],
                            article_data["coverImage"],
                            article_data["readingTime"],
                            article_data["wordCount"],
                            article_data["status"],
                            article_data["metaTitle"],
                            article_data["metaDescription"],
                            article_data["featured"],
                            article_data["spotlight"],
                            article_data["evergreen"],
                            article_data["sponsored"],
                            article_data.get("sponsorName", ""),
                            article_data["partnerContent"],
                            article_data["affiliate"],
                            article_data["crowdsourced"],
                            article_data["premium"],
                            article_data["hasVideo"],
                            article_data["hasAudio"],
                            article_data["hasGallery"],
                            article_data["slug"]
                        ))
                    else:
                        # Create new article
                        sql = """
                        INSERT INTO Article (
                            id, title, slug, content, markdownUrl, excerpt, coverImage,
                            readingTime, wordCount, publishedAt, updatedAt, status,
                            viewCount, metaTitle, metaDescription, 
                            featured, spotlight, evergreen, sponsored, sponsorName,
                            partnerContent, affiliate, crowdsourced, premium,
                            hasVideo, hasAudio, hasGallery, authorId
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s,
                            0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        """
                        cursor.execute(sql, (
                            new_id,
                            article_data["title"],
                            article_data["slug"],
                            article_data["content"],
                            article_data["markdownUrl"],
                            article_data["excerpt"],
                            article_data["coverImage"],
                            article_data["readingTime"],
                            article_data["wordCount"],
                            "PUBLISHED",
                            article_data["metaTitle"],
                            article_data["metaDescription"],
                            article_data["featured"],
                            article_data["spotlight"],
                            article_data["evergreen"],
                            article_data["sponsored"],
                            article_data.get("sponsorName", ""),
                            article_data["partnerContent"],
                            article_data["affiliate"],
                            article_data["crowdsourced"],
                            article_data["premium"],
                            article_data["hasVideo"],
                            article_data["hasAudio"],
                            article_data["hasGallery"],
                            article_data["authorId"]
                        ))

                    # After creating/updating the article, handle categories
                    if article_data.get("categories"):
                        # First, ensure categories exist
                        for category_slug in article_data["categories"]:
                            # Check if category exists
                            cursor.execute(
                                "SELECT id FROM Category WHERE slug = %s",
                                (category_slug,)
                            )
                            category = cursor.fetchone()
                            
                            if not category:
                                # Create category if it doesn't exist
                                category_id = str(uuid.uuid4())  # Generate a UUID for the category
                                cursor.execute(
                                    "INSERT INTO Category (id, name, slug) VALUES (%s, %s, %s)",
                                    (category_id, category_slug.title(), category_slug)
                                )
                                category = (category_id,)

                            # Create category-article relationship
                            cursor.execute("""
                                INSERT INTO _ArticleToCategory (A, B)
                                VALUES (%s, %s)
                                ON DUPLICATE KEY UPDATE A = VALUES(A)
                            """, (
                                existing_article[0] if existing_article else new_id,
                                category[0]
                            ))

                    connection.commit()
                    print(f"Successfully created/updated article with ID: {new_id if not existing_article else existing_article[0]}")
                    return True
                    
        except Exception as e:
            print(f"Database error creating/updating article: {e}")
            print("Article data:", article_data)
            raise

        return False

class ArticleScraper:
    def __init__(self):
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def scrape_url(self, url: str) -> Dict:
        """Scrape content from a URL using trafilatura for better content extraction"""
        try:
            await self.init_session()
            logging.info(f"\nAttempting to scrape: {url}")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logging.error(f"ERROR: HTTP {response.status} error fetching URL")
                    return {"error": f"HTTP {response.status}", "content": None}
                
                html = await response.text()
                logging.info(f"Retrieved HTML content length: {len(html)} characters")
                
                # Use trafilatura for main content extraction
                content = trafilatura.extract(html, include_comments=False, 
                                         include_tables=True,
                                         include_images=False,
                                         include_links=False)
                
                if not content:
                    logging.warning("WARNING: Trafilatura failed to extract content, falling back to BeautifulSoup")
                    # Fallback to BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Debug print of HTML structure
                    logging.info("HTML Structure Overview:")
                    logging.info(f"- Found {len(soup.find_all('article'))} article tags")
                    logging.info(f"- Found {len(soup.find_all('p'))} paragraph tags")
                    logging.info(f"- Found {len(soup.find_all('img'))} image tags")
                    
                    # Remove unwanted elements
                    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        element.decompose()
                    
                    # Try to find main content
                    article = soup.find('article') or soup.find(class_=['content', 'article', 'post'])
                    if article:
                        content = article.get_text(separator='\n', strip=True)
                        logging.info("SUCCESS: Found main content in article tag")
                    else:
                        # Fallback to all paragraphs
                        paragraphs = soup.find_all('p')
                        content = '\n'.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
                        logging.info(f"INFO: Extracted content from {len(paragraphs)} paragraphs")

                if content:
                    logging.info(f"SUCCESS: Successfully extracted content ({len(content)} characters)")
                else:
                    logging.error("ERROR: Failed to extract any content")

                return {
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "content": content,
                    "error": None
                }

        except Exception as e:
            logging.error(f"ERROR: Error scraping URL: {str(e)}")
            return {"error": str(e), "content": None, "url": url}

class CategoryDetector:
    def __init__(self, categories_file: str = "categories.json"):
        with open(categories_file, 'r', encoding='utf-8') as f:
            self.categories = json.load(f)["categories"]
        self.create_keyword_mappings()

    def create_keyword_mappings(self):
        """Create keyword mappings using keywords from config"""
        self.category_keywords = {}
        self.subcategory_keywords = {}
        
        for category in self.categories:
            keywords = set()
            
            # Add keywords from category itself
            if "keywords" in category:
                keywords.update(category["keywords"])
            
            # Add words from name and description
            keywords.update(category["name"].lower().split())
            if category.get("description"):
                keywords.update(category["description"].lower().split())
            
            self.category_keywords[category["slug"]] = keywords
            
            # Create subcategory keywords mapping
            self.subcategory_keywords[category["slug"]] = {}
            for sub in category["subCategories"]:
                sub_keywords = set()
                if "keywords" in sub:
                    sub_keywords.update(sub["keywords"])
                sub_keywords.update(sub["name"].lower().split())
                if sub.get("description"):
                    sub_keywords.update(sub["description"].lower().split())
                self.subcategory_keywords[category["slug"]][sub["slug"]] = sub_keywords

    def detect_categories(self, content: str, threshold: float = 0.3) -> Dict:
        """
        Detect primary and additional categories for content
        """
        content_lower = content.lower()
        content_words = set(content_lower.split())
        
        # Score each category
        category_scores = []
        max_score = 0
        
        for category in self.categories:
            # Calculate category score
            cat_keywords = self.category_keywords[category["slug"]]
            base_score = len(content_words.intersection(cat_keywords))
            
            # Add bonus for exact phrase matches
            if category["name"].lower() in content_lower:
                base_score += 10
                
            # Find best matching subcategory
            sub_scores = []
            for sub in category["subCategories"]:
                sub_score = 0
                sub_keywords = self.subcategory_keywords[category["slug"]][sub["slug"]]
                
                # Calculate subcategory score
                sub_score += len(content_words.intersection(sub_keywords))
                if sub["name"].lower() in content_lower:
                    sub_score += 5
                    
                sub_scores.append((sub["slug"], sub_score))
            
            best_subcategory = max(sub_scores, key=lambda x: x[1])
            total_score = base_score + best_subcategory[1]
            max_score = max(max_score, total_score)
            
            category_scores.append({
                "category": category["slug"],
                "subcategory": best_subcategory[0],
                "score": total_score,
                "normalized_score": total_score
            })
        
        # Normalize scores
        if max_score > 0:
            for score in category_scores:
                score["normalized_score"] = score["score"] / max_score
        
        # Sort by score descending
        category_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "primary": category_scores[0],
            "additional": [
                cat for cat in category_scores[1:]
                if cat["normalized_score"] >= threshold
            ]
        }

    def get_category_name(self, slug: str) -> str:
        """Get category name from slug"""
        category = next((c for c in self.categories if c["slug"] == slug), None)
        return category["name"] if category else slug

    def get_subcategory_name(self, category_slug: str, subcategory_slug: str) -> str:
        """Get subcategory name from slugs"""
        category = next((c for c in self.categories if c["slug"] == category_slug), None)
        if not category:
            return subcategory_slug
        subcategory = next((s for s in category["subCategories"] if s["slug"] == subcategory_slug), None)
        return subcategory["name"] if subcategory else subcategory_slug

class ContentGenerator:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
        self.minio_client = Minio(
            os.getenv("MINIO_ENDPOINT").replace("https://", ""),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=True
        )
        
        try:
            with open("article_template.md", "r", encoding="utf-8") as f:
                self.template = f.read()
        except Exception as e:
            print(f"Error loading template: {e}")
            self.template = None

    def validate_content(self, content_data: Dict) -> bool:
        """Validate generated content meets requirements"""
        word_count = len(content_data["content"].split())
        if not (800 <= word_count <= 1500):
            logging.warning(f"Content length ({word_count} words) outside target range")
            return False
            
        if len(content_data["metaTitle"]) > 70:
            logging.warning("Meta title too long")
            return False
            
        if len(content_data["metaDescription"]) > 150:
            logging.warning("Meta description too long")
            return False
            
        return True

    def generate_content(self, sources: List[Dict], title: str) -> Dict:
        source_texts = [f"Source {i+1} ({s['url']}):\n{s['content']}\n\n" 
                       for i, s in enumerate(sources)]
        
        combined_sources = "\n".join(source_texts)
        
        prompt = f"""
        Write a comprehensive, SEO-optimized article that synthesizes information from the provided sources.

        Requirements:
        1. Structure:
            - Start with an engaging introduction (2-3 paragraphs)
            - Include 4-5 H2 subheadings for main sections
            - Each section must have 2-3 paragraphs minimum
            - Use proper HTML tags: <h1>, <h2>, <p> for all content
            - Total length MUST be between 800-1500 words
        
        2. Writing Style:
            - Write for a 9th-grade reading level
            - Use short, clear sentences
            - Each paragraph should be 2-4 sentences
            - Include relevant statistics and data from sources
        
        3. SEO Optimization:
            - Include LSI keywords naturally
            - Use descriptive subheadings
            - Ensure proper heading hierarchy
            - Include transition sentences between sections

        Title: {title}

        Sources:
        {combined_sources}

        Return ONLY a JSON object in this exact format:
        {{
            "content": "Your full HTML-formatted article content with proper tags",
            "excerpt": "Compelling 2-3 sentence summary (max 550 chars)",
            "metaTitle": "SEO-optimized title (max 70 chars)",
            "metaDescription": "SEO-optimized description (max 150 chars)",
            "tags": ["tag1", "tag2", "tag3"],
            "suggestedFeatures": {{
                "hasVideo": false,
                "hasAudio": false,
                "hasGallery": false
            }}
        }}

        The content MUST use proper HTML tags and be between 800-1500 words. Do not include any markdown formatting - use HTML tags instead.
        """

        response = self.anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].strip()
        
        response_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', response_text)
        
        return json.loads(response_text)

    def _generate_fallback_content(self, title: str) -> Dict:
        """Generate fallback content when all attempts fail"""
        return {
            "content": f"<h1>{title}</h1>\n<p>Content generation failed after multiple attempts. Please try again.</p>",
            "excerpt": f"Article about {title}",
            "metaTitle": title[:70],
            "metaDescription": f"Learn about {title}",
            "tags": [],
            "suggestedFeatures": {
                "hasVideo": False,
                "hasAudio": False,
                "hasGallery": False
            }
        }

    def create_article_markdown(self, content_data: Dict, article_data: Dict) -> str:
        """Create markdown file content using template"""
        if not self.template:
            raise ValueError("Article template not loaded")
        
        try:
            word_count = len(content_data["content"].split())
            reading_time = math.ceil(word_count / 200)
            
            template_data = {
                "title": article_data["title"],
                "slug": article_data["slug"],
                "excerpt": content_data["excerpt"],
                "coverImage": "/images/default.webp",
                "publishedAt": datetime.now().isoformat(),
                "author": "cm3pw0m9u00026hqfvtiqmtcw",
                "status": "PUBLISHED",
                "readingTime": str(reading_time),
                "metaTitle": content_data["metaTitle"],
                "metaDescription": content_data["metaDescription"],
                "category": article_data["primary_category"]["slug"],
                "subCategory": article_data["primary_category"]["subcategory"]["slug"],
                "tags": str(content_data["tags"]),
                
                "featured": "false",
                "spotlight": "false",
                "evergreen": "false",
                "sponsored": "false",
                "sponsorName": "",
                "partnerContent": "false",
                "affiliate": "false",
                "crowdsourced": "false",
                "premium": "false",
                
                "hasVideo": str(content_data["suggestedFeatures"]["hasVideo"]).lower(),
                "hasAudio": str(content_data["suggestedFeatures"]["hasAudio"]).lower(),
                "hasGallery": str(content_data["suggestedFeatures"]["hasGallery"]).lower(),
                
                "series": "",
                "seriesOrder": "0",
                
                "content": content_data["content"]
            }
            
            return self.template.format(**template_data)
            
        except Exception as e:
            print(f"Error creating markdown: {str(e)}")
            print("Template data:", json.dumps(template_data, indent=2))
            raise

    async def save_to_minio(self, markdown_content: str, slug: str) -> str:
        """Save markdown file to MinIO"""
        bucket_name = os.getenv("MINIO_BUCKET")
        
        try:
            if not self.minio_client.bucket_exists(bucket_name):
                self.minio_client.make_bucket(bucket_name)
            
            file_path = f"articles/{slug}.md"
            content_bytes = markdown_content.encode('utf-8')
            
            # Use BytesIO to create a file-like object
            from io import BytesIO
            content_stream = BytesIO(content_bytes)
            
            self.minio_client.put_object(
                bucket_name,
                file_path,
                content_stream,
                length=len(content_bytes),
                content_type="text/markdown"
            )
            
            return f"{os.getenv('MINIO_ENDPOINT')}/{bucket_name}/{file_path}"
        except Exception as e:
            print(f"Error saving to MinIO: {e}")
            return None

class ImageProcessor:
    def __init__(self, minio_client):
        self.minio_client = minio_client
        self.min_width = 800
        self.min_height = 600
        self.aspect_ratio = 16/9
        self.default_image_path = "C:/Users/Usuario/Desktop/Cursor/NewsMag/public/images/default.webp"
        
    def process_image(self, image: Image.Image) -> Image.Image:
        """Convert to black and white and crop to 16:9 aspect ratio"""
        # Convert to black and white
        bw_image = image.convert('L')
        
        # Calculate target dimensions for 16:9
        current_ratio = image.width / image.height
        
        if current_ratio > self.aspect_ratio:
            # Image is too wide
            new_width = int(image.height * self.aspect_ratio)
            offset = (image.width - new_width) // 2
            bw_image = bw_image.crop((offset, 0, offset + new_width, image.height))
        else:
            # Image is too tall
            new_height = int(image.width / self.aspect_ratio)
            offset = (image.height - new_height) // 2
            bw_image = bw_image.crop((0, offset, image.width, offset + new_height))
            
        return bw_image
        
    async def save_image(self, image: Image.Image, article_slug: str, article_title: str) -> str:
        """Save processed image to MinIO"""
        try:
            # Generate filename from article title
            safe_title = slugify(article_title)[:50]  # Limit length
            filename = f"{safe_title}-{uuid.uuid4().hex[:8]}.webp"
            bucket_name = os.getenv("MINIO_BUCKET")
            file_path = f"images/articles/{article_slug}/{filename}"
            
            # Convert to WebP format
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=85)
            buffer.seek(0)
            
            self.minio_client.put_object(
                bucket_name,
                file_path,
                buffer,
                length=buffer.getbuffer().nbytes,
                content_type="image/webp"
            )
            
            return f"{os.getenv('MINIO_ENDPOINT')}/{bucket_name}/{file_path}"
            
        except Exception as e:
            print(f"Error saving image: {e}")
            return f"/images/default.webp"
            
    async def get_largest_relevant_image(self, html: str, article_title: str, url: str) -> Dict:
        """Find the largest relevant image from the article HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        images = []
        
        logging.info(f"Searching for images in article: {article_title}")
        logging.info(f"URL: {url}")
        
        # Debug: Print total number of img tags found
        all_images = soup.find_all('img')
        logging.info(f"Total <img> tags found: {len(all_images)}")
        
        for img in all_images:
            src = img.get('src')
            if not src:
                continue
            
            logging.info(f"Processing image: {src}")
            logging.info(f"Alt text: {img.get('alt', 'None')}")
            logging.info(f"Title: {img.get('title', 'None')}")
            
            # Clean and normalize the URL
            if not src.startswith(('http://', 'https://')):
                original_src = src
                src = urljoin(url, src)
                logging.info(f"Normalized URL from {original_src} to {src}")
            
            # Skip small icons and common non-article images
            if any(skip in src.lower() for skip in ['icon', 'logo', 'avatar', 'ads', 'banner']):
                logging.info("Skipping: appears to be icon/logo/banner")
                continue
            
            try:
                logging.info("Downloading image...")
                response = requests.get(src, timeout=10)
                
                if response.status_code != 200:
                    logging.info(f"Failed to download: HTTP {response.status_code}")
                    continue
                
                logging.info("Checking dimensions...")
                img_file = Image.open(io.BytesIO(response.content))
                width, height = img_file.size
                logging.info(f"Dimensions: {width}x{height}")
                
                if width >= self.min_width and height >= self.min_height:
                    images.append({
                        'src': src,
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'width': width,
                        'height': height,
                        'area': width * height
                    })
                    logging.info("Image added to candidates")
                else:
                    logging.info(f"Skipping: too small (minimum: {self.min_width}x{self.min_height})")
            
            except Exception as e:
                logging.error(f"Error processing image: {str(e)}")
        
        if not images:
            logging.info("No suitable images found")
            return None
        
        # Sort by area and return largest
        images.sort(key=lambda x: x['area'], reverse=True)
        selected_image = images[0]
        logging.info("Selected best image:")
        logging.info(f"Source: {selected_image['src']}")
        logging.info(f"Dimensions: {selected_image['width']}x{selected_image['height']}")
        logging.info(f"Alt text: {selected_image['alt']}")
        
        return selected_image

    async def get_and_process_image(self, html: str, article_title: str, article_slug: str, url: str) -> str:
        """Main method to get, process and save article image"""
        logging.info(f"Starting image processing for: {article_title}")
        
        try:
            logging.info("Searching for suitable image...")
            image_data = await self.get_largest_relevant_image(html, article_title, url)
            
            if image_data:
                logging.info(f"Downloading selected image: {image_data['src']}")
                response = requests.get(image_data['src'])
                
                logging.info("Processing image...")
                img = Image.open(io.BytesIO(response.content))
                processed_img = self.process_image(img)
                
                logging.info("Saving to MinIO...")
                image_url = await self.save_image(processed_img, article_slug, article_title)
                logging.info(f"Image saved: {image_url}")
                return image_url
                
            else:
                logging.warning("No suitable image found, using default")
                
        except Exception as e:
            logging.error(f"Error in image processing: {str(e)}")
            logging.warning("Falling back to default image")
        
        logging.info(f"Using default image: {self.default_image_path}")
        return "/images/default.webp"

class ArticleProcessor(GoogleSheetsManager):
    def __init__(self):
        super().__init__(os.getenv('GOOGLE_APPLICATION_CREDENTIALS', './google-credentials.json'))
        self.scraper = ArticleScraper()
        self.category_detector = CategoryDetector()
        self.content_generator = ContentGenerator()
        self.db_handler = DatabaseHandler(os.getenv('DATABASE_URL'))
        self.image_processor = ImageProcessor(self.content_generator.minio_client)

    async def get_articles(self):
        """Get articles from sheet"""
        sheet = self.service.spreadsheets()
        result = sheet.values().get(
            spreadsheetId=os.getenv('SHEET_ID'),
            range='A2:Z'
        ).execute()
        
        values = result.get('values', [])
        articles = []
        
        for idx, row in enumerate(values, start=2):
            if not row:
                continue
            
            title = row[0].strip()
            slug = slugify(title)
            final_url = f"https://riskbagel.com/articles/{slug}"
            
            source_urls = []
            for col_idx, cell in enumerate(row[1:], start=1):
                if cell and cell.strip():
                    source_urls.append({
                        'column': chr(65 + col_idx),
                        'url': cell.strip()
                    })
            
            articles.append({
                "row": idx,
                "title": title,
                "slug": slug,
                "final_url": final_url,
                "source_urls": source_urls
            })
            
        return articles

    async def process_article(self, article_data: Dict) -> Dict:
        """Process a single article with all its sources"""
        logging.info(f"\nProcessing article: {article_data['title']}")
        source_contents = []
        
        # Scrape all source URLs
        for source in article_data["source_urls"]:
            logging.info(f"\nProcessing source: {source['url']}")
            content = await self.scraper.scrape_url(source["url"])
            if not content["error"] and content["content"]:
                source_contents.append(content)
                logging.info(f"SUCCESS: Successfully scraped content ({len(content['content'])} chars)")
            else:
                logging.warning(f"WARNING: Failed to scrape content: {content.get('error', 'Unknown error')}")
        
        if not source_contents:
            logging.error("ERROR: No content could be scraped from any source")
            return {"error": "No content could be scraped from any source"}
        
        # Log content stats
        logging.info("\nContent Statistics:")
        for idx, source in enumerate(source_contents, 1):
            logging.info(f"Source {idx}: {len(source['content'])} chars from {source['domain']}")

        try:
            # Generate content
            logging.info("\nGenerating content using Claude...")
            content_data = self.content_generator.generate_content(source_contents, article_data["title"])
            
            # Create markdown
            logging.info("Creating markdown content...")
            markdown_content = self.content_generator.create_article_markdown(content_data, article_data)
            
            # Save to MinIO
            logging.info("Saving to MinIO...")
            storage_url = await self.content_generator.save_to_minio(markdown_content, article_data["slug"])
            
            # Add results to return dictionary
            article_data.update({
                "content": content_data["content"],
                "storage_url": storage_url,
                "excerpt": content_data.get("excerpt", ""),
                "metaTitle": content_data.get("metaTitle", article_data["title"][:70]),
                "metaDescription": content_data.get("metaDescription", ""),
                "tags": content_data.get("tags", [])
            })
            
            logging.info(f"SUCCESS: Successfully saved to: {storage_url}")
            return article_data
            
        except Exception as e:
            logging.error(f"ERROR: Error in content generation: {str(e)}")
            logging.error(f"Full error: {e.__class__.__name__}: {str(e)}")
            article_data["error"] = f"Content generation failed: {str(e)}"
            return article_data

async def main():
    processor = None
    try:
        processor = ArticleProcessor()
        
        print("\nFetching articles from sheet...")
        articles = await processor.get_articles()
        
        print("\nProcessing articles...")
        results = []
        for article in articles:
            print(f"\nProcessing: {article['title']}")
            result = await processor.process_article(article)
            results.append(result)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                continue
            
            print(f"Primary Category: {result['primary_category']['name']} / {result['primary_category']['subcategory']['name']}")
            if result.get('additional_categories'):
                print("Additional Categories:")
                for cat in result['additional_categories']:
                    print(f"  - {cat['name']} / {cat['subcategory']['name']} (Score: {cat['score']:.2f})")
            
            if 'storage_url' in result:
                print(f"Saved to: {result['storage_url']}")
            print("---")
        
        print("\nProcessing complete!")
        print(f"Successfully processed {len([r for r in results if 'error' not in r])} articles")
        print(f"Failed to process {len([r for r in results if 'error' in r])} articles")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        if processor and processor.scraper:
            await processor.scraper.close_session()

if __name__ == "__main__":
    asyncio.run(main())

