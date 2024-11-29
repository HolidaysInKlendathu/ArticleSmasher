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
from urllib.parse import urlparse
import trafilatura
from concurrent.futures import ThreadPoolExecutor
from anthropic import Anthropic
import math
from minio import Minio
from datetime import datetime
import pymysql


# Import our GoogleSheetsManager
from sheets_setup import GoogleSheetsManager

load_dotenv()

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
                            "DRAFT",  # status
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

    async def scrape_url(self, url: str) -> Dict:
        """Scrape content from a URL using trafilatura for better content extraction"""
        try:
            await self.init_session()
            async with self.session.get(url) as response:
                if response.status != 200:
                    return {"error": f"HTTP {response.status}", "content": None}
                
                html = await response.text()
                
                # Use trafilatura for main content extraction
                content = trafilatura.extract(html, include_comments=False, 
                                           include_tables=True,
                                           include_images=False,
                                           include_links=False)
                
                if not content:
                    # Fallback to BeautifulSoup if trafilatura fails
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        element.decompose()
                    
                    # Try to find main content
                    article = soup.find('article') or soup.find(class_=['content', 'article', 'post'])
                    if article:
                        content = article.get_text(separator='\n', strip=True)
                    else:
                        # Fallback to all paragraphs
                        paragraphs = soup.find_all('p')
                        content = '\n'.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())

                return {
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "content": content,
                    "error": None
                }

        except Exception as e:
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

    def generate_content(self, sources: List[Dict], title: str) -> Dict:
        """Generate article content using Claude"""
        source_texts = [f"Source {i+1} ({s['url']}):\n{s['content']}\n\n" 
                       for i, s in enumerate(sources)]
        
        combined_sources = "\n".join(source_texts)
        
        prompt = f"""
        Write an informative article based on the provided sources.
        Return ONLY a JSON object with no additional text.

        Requirements:
        - Reading level: 9-10 years old
        - Use simple sentences and short paragraphs
        - Length: 800-1500 words
        
        Important: In your JSON response, ensure the content field has no line breaks in it - use \\n for newlines.
        
        Title: {title}

        Sources:
        {combined_sources}

        Return your response in this exact JSON format, making sure all text fields are properly escaped:
        {{
            "content": "Single line of text with \\n for newlines",
            "excerpt": "2-3 sentence summary under 550 characters",
            "metaTitle": "SEO title under 70 characters",
            "metaDescription": "SEO description under 150 characters",
            "tags": ["tag1", "tag2", "tag3"],
            "suggestedFeatures": {{
                "hasVideo": false,
                "hasAudio": false,
                "hasGallery": false
            }}
        }}
        """

        try:
            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Clean up response text
            response_text = response.content[0].text.strip()
            
            # Debug print
            print("\nRaw response beginning:")
            print(response_text[:500])
            
            # Try to extract JSON if wrapped in code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            # Clean the response text - replace literal newlines with \n in content
            cleaned_text = response_text.replace('"\n', '"\\n')
            
            try:
                result = json.loads(cleaned_text)
            except json.JSONDecodeError:
                # If still failing, try more aggressive cleaning
                import re
                cleaned_text = re.sub(r'(?<!\\)\n', '\\n', response_text)
                result = json.loads(cleaned_text)
            
            # Ensure all required fields exist
            required_fields = {
                "content": "",
                "excerpt": "",
                "metaTitle": "",
                "metaDescription": "",
                "tags": [],
                "suggestedFeatures": {
                    "hasVideo": False,
                    "hasAudio": False,
                    "hasGallery": False
                }
            }
            
            for key, default_value in required_fields.items():
                if key not in result:
                    result[key] = default_value
            
            return result
            
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            return {
                "content": f"Error generating content: {str(e)}",
                "excerpt": f"Error processing article: {title}",
                "metaTitle": title[:70],
                "metaDescription": f"Article about {title[:140]}",
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
                "coverImage": "/images/default-cover.jpg",
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

class ArticleProcessor(GoogleSheetsManager):
    def __init__(self):
        super().__init__(os.getenv('GOOGLE_APPLICATION_CREDENTIALS', './google-credentials.json'))
        self.scraper = ArticleScraper()
        self.category_detector = CategoryDetector()
        self.content_generator = ContentGenerator()
        self.db_handler = DatabaseHandler(os.getenv('DATABASE_URL'))

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
        source_contents = []
        
        # Scrape all source URLs
        for source in article_data["source_urls"]:
            content = await self.scraper.scrape_url(source["url"])
            if not content["error"] and content["content"]:
                source_contents.append(content)
        
        if not source_contents:
            return {"error": "No content could be scraped from any source"}
        
        # Combine contents for category detection
        combined_content = "\n".join(s["content"] for s in source_contents)
        
        # Detect categories
        categories = self.category_detector.detect_categories(combined_content)
        
        result = {
            "title": article_data["title"],
            "slug": article_data["slug"],
            "final_url": article_data["final_url"],
            "sources": source_contents,
            "primary_category": {
                "slug": categories["primary"]["category"],
                "name": self.category_detector.get_category_name(categories["primary"]["category"]),
                "subcategory": {
                    "slug": categories["primary"]["subcategory"],
                    "name": self.category_detector.get_subcategory_name(
                        categories["primary"]["category"],
                        categories["primary"]["subcategory"]
                    )
                }
            },
            "additional_categories": [
                {
                    "slug": cat["category"],
                    "name": self.category_detector.get_category_name(cat["category"]),
                    "subcategory": {
                        "slug": cat["subcategory"],
                        "name": self.category_detector.get_subcategory_name(
                            cat["category"],
                            cat["subcategory"]
                        )
                    },
                    "score": cat["score"]
                }
                for cat in categories["additional"]
            ]
        }

        # Generate content using Claude
        try:
            content_data = self.content_generator.generate_content(source_contents, article_data["title"])
            
            # Create markdown
            markdown_content = self.content_generator.create_article_markdown(content_data, result)
            
            # Save to MinIO
            storage_url = await self.content_generator.save_to_minio(markdown_content, result["slug"])

            # Prepare article data for database
            db_article_data = {
                "title": result["title"],
                "slug": result["slug"],
                "content": content_data["content"],
                "markdownUrl": storage_url,
                "excerpt": content_data["excerpt"],
                "coverImage": "/images/default-cover.jpg",
                "readingTime": math.ceil(len(content_data["content"].split()) / 200),
                "wordCount": len(content_data["content"].split()),
                "status": "DRAFT",
                "metaTitle": content_data["metaTitle"],
                "metaDescription": content_data["metaDescription"],
                "featured": False,
                "spotlight": False,
                "evergreen": False,
                "sponsored": False,
                "sponsorName": "",
                "partnerContent": False,
                "affiliate": False,
                "crowdsourced": False,
                "premium": False,
                "hasVideo": content_data["suggestedFeatures"]["hasVideo"],
                "hasAudio": content_data["suggestedFeatures"]["hasAudio"],
                "hasGallery": content_data["suggestedFeatures"]["hasGallery"],
                "authorId": "cm3pw0m9u00026hqfvtiqmtcw",
                "categories": [result["primary_category"]["slug"]],
                "tags": content_data["tags"]
            }
            
            # Update database
            if storage_url and 'error' not in result:
                db_update_success = self.db_handler.create_or_update_article(db_article_data)
                if db_update_success:
                    result["database_updated"] = True
                    print(f"Database updated for article: {result['title']}")
                else:
                    result["database_updated"] = False
                    print(f"Failed to update database for article: {result['title']}")
            
            result.update({
                "storage_url": storage_url,
                "markdown_content": markdown_content,
                "generated_content": content_data
            })
            
        except Exception as e:
            print(f"Error in content generation: {e}")
            result["error"] = f"Content generation failed: {str(e)}"
        
        return result

async def main():
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
            if result['additional_categories']:
                print("Additional Categories:")
                for cat in result['additional_categories']:
                    print(f"  - {cat['name']} / {cat['subcategory']['name']} (Score: {cat['score']:.2f})")
            
            if 'storage_url' in result:
                print(f"Saved to: {result['storage_url']}")
            print("---")
        
        # Close sessions
        await processor.scraper.close_session()
        
        print("\nProcessing complete!")
        print(f"Successfully processed {len([r for r in results if 'error' not in r])} articles")
        print(f"Failed to process {len([r for r in results if 'error' in r])} articles")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())