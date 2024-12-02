import os
from anthropic import Anthropic
from typing import List, Dict
import json
from datetime import datetime
import yaml
from minio import Minio
import math
import logging

class ContentGenerator:
    def __init__(self):
        self.anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
        self.minio_client = Minio(
            os.getenv("MINIO_ENDPOINT").replace("https://", ""),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=True
        )
        
        # Load article template
        with open("article_template.md", "r", encoding="utf-8") as f:
            self.template = f.read()

    def generate_content(self, sources: List[Dict], title: str) -> Dict:
        """Generate article content using Claude"""
        try:
            source_texts = [f"Source {i+1} ({s['url']}):\n{s['content']}\n\n" 
                        for i, s in enumerate(sources)]
            
            combined_sources = "\n".join(source_texts)
            
            prompt = f"""
            You are a professional content writer. Write a comprehensive article that synthesizes information from the provided sources.
            
            Return ONLY a JSON object in this exact format, with no additional text or formatting:
            {{
                "content": "# {title}\\n\\nIntroduction...\\n\\n## Section 1\\n\\nContent...\\n\\n## Section 2\\n\\nContent...",
                "excerpt": "Brief summary",
                "metaTitle": "{title}",
                "metaDescription": "Learn about {title}",
                "tags": ["tag1", "tag2"],
                "suggestedFeatures": {{
                    "hasVideo": false,
                    "hasAudio": false,
                    "hasGallery": false
                }}
            }}

            Sources to synthesize:
            {combined_sources}

            Requirements:
            1. Content should be 800-1200 words
            2. Use proper markdown formatting
            3. Write in a simple, engaging style
            4. Include 3-4 main sections with H2 headings
            """

            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Clean up response text
            response_text = response.content[0].text.strip()
            logging.info("Raw response from Claude:")
            logging.info(response_text[:500] + "...")  # Log first 500 chars
            
            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}")
                logging.error(f"Attempted to parse: {response_text[:500]}...")
                # Return fallback content
                return {
                    "content": f"# {title}\n\nContent generation failed. Please try again.",
                    "excerpt": f"Article about {title}",
                    "metaTitle": title[:70],
                    "metaDescription": f"Learn about {title}",
                    "tags": [],
                    "suggestedFeatures": {"hasVideo": False, "hasAudio": False, "hasGallery": False}
                }
                
        except Exception as e:
            logging.error(f"Error in content generation: {str(e)}")
            return {
                "content": f"# {title}\n\nError: {str(e)}",
                "excerpt": f"Article about {title}",
                "metaTitle": title[:70],
                "metaDescription": f"Learn about {title}",
                "tags": [],
                "suggestedFeatures": {"hasVideo": False, "hasAudio": False, "hasGallery": False}
            }

    def create_article_markdown(self, content_data: Dict, article_data: Dict) -> str:
        """Create markdown file content using template"""
        try:
            # Calculate reading time
            word_count = len(content_data["content"].split())
            reading_time = math.ceil(word_count / 200)
            
            # Ensure primary_category exists
            if "primary_category" not in article_data:
                article_data["primary_category"] = {
                    "slug": "uncategorized",
                    "subcategory": {"slug": "general"}
                }
            
            # Prepare template variables
            template_vars = {
                "title": article_data["title"],
                "slug": article_data["slug"],
                "excerpt": content_data.get("excerpt", ""),
                "coverImage": "/images/default.webp",
                "publishedAt": datetime.now().isoformat(),
                "author": "cm3pw0m9u00026hqfvtiqmtcw",
                "status": "PUBLISHED",
                "readingTime": str(reading_time),
                "metaTitle": content_data.get("metaTitle", article_data["title"][:70]),
                "metaDescription": content_data.get("metaDescription", ""),
                "category": article_data["primary_category"]["slug"],
                "subCategory": article_data["primary_category"]["subcategory"]["slug"],
                "tags": json.dumps(content_data.get("tags", [])),
                "featured": "false",
                "spotlight": "false",
                "evergreen": "false",
                "sponsored": "false",
                "sponsorName": "",
                "partnerContent": "false",
                "affiliate": "false",
                "crowdsourced": "false",
                "premium": "false",
                "hasVideo": str(content_data.get("suggestedFeatures", {}).get("hasVideo", False)).lower(),
                "hasAudio": str(content_data.get("suggestedFeatures", {}).get("hasAudio", False)).lower(),
                "hasGallery": str(content_data.get("suggestedFeatures", {}).get("hasGallery", False)).lower(),
                "series": "",
                "seriesOrder": "0",
                "content": content_data["content"]
            }
            
            return self.template.format(**template_vars)
            
        except Exception as e:
            logging.error(f"Error creating markdown: {str(e)}")
            raise

    async def save_to_minio(self, markdown_content: str, slug: str) -> str:
        """Save markdown file to MinIO"""
        bucket_name = os.getenv("MINIO_BUCKET")
        
        # Create bucket if it doesn't exist
        if not self.minio_client.bucket_exists(bucket_name):
            self.minio_client.make_bucket(bucket_name)
        
        # Prepare file path
        file_path = f"articles/{slug}.md"
        
        # Save to MinIO
        self.minio_client.put_object(
            bucket_name,
            file_path,
            data=markdown_content.encode(),
            length=len(markdown_content.encode()),
            content_type="text/markdown"
        )
        
        return f"{os.getenv('MINIO_ENDPOINT')}/{bucket_name}/{file_path}"