import os
from anthropic import Anthropic
from typing import List, Dict
import json
from datetime import datetime
import yaml
from minio import Minio
import math
import logging
import re

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
            
            prompt = f"""You are a professional content writer. Write a comprehensive article that synthesizes information from the provided sources.

Return ONLY a valid JSON object with this exact structure (no additional text or formatting):
{{
    "content": "<h1>{title}</h1>\\n<p>Introduction paragraph...</p>\\n<h2>First Section</h2>\\n<p>Content...</p>",
    "excerpt": "Brief 1-2 sentence summary",
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
1. Content must be 800-1500 words
2. Use proper HTML tags (<h1>, <h2>, <p>)
3. Write in a clear, engaging style
4. Include 3-4 main sections with H2 headings
5. Return ONLY the JSON object - no other text or formatting"""

            response = self.anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            logging.debug(f"Raw response from Claude: {response_text[:500]}...")
            
            # Clean up the response text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()
            
            # Remove any potential leading/trailing characters
            response_text = re.sub(r'^[^{]*', '', response_text)
            response_text = re.sub(r'[^}]*$', '', response_text)
            
            try:
                # Log the cleaned response for debugging
                logging.debug(f"Cleaned response text: {response_text[:500]}...")
                
                result = json.loads(response_text)
                
                # Validate the required fields
                required_fields = ["content", "excerpt", "metaTitle", "metaDescription", "tags", "suggestedFeatures"]
                missing_fields = [field for field in required_fields if field not in result]
                
                if missing_fields:
                    logging.error(f"Missing required fields in response: {missing_fields}")
                    return self._generate_fallback_content(title)
                    
                return result
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error: {str(e)}")
                logging.error(f"Problematic response text: {response_text[:500]}...")
                return self._generate_fallback_content(title)
                
        except Exception as e:
            logging.error(f"Error in content generation: {str(e)}")
            return self._generate_fallback_content(title)

    def _generate_fallback_content(self, title: str) -> Dict:
        """Generate fallback content when main generation fails"""
        return {
            "content": f"<h1>{title}</h1>\n<p>Content generation failed. Please try again.</p>",
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