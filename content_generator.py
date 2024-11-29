import os
from anthropic import Anthropic
from typing import List, Dict
import json
from datetime import datetime
import yaml
from minio import Minio
import math

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
            source_texts = [f"Source {i+1} ({s['url']}):\n{s['content']}\n\n" 
                        for i, s in enumerate(sources)]
            
            combined_sources = "\n".join(source_texts)
            
            prompt = f"""
            Write a comprehensive article that synthesizes information from all provided sources.

            Article Requirements:
            - Structure:
            * Start with a clear H1 title
            * Include 3-4 H2 subheadings for main sections
            * Each section should be 2-3 paragraphs
            - Writing Style:
            * Casual and bold tone
            * Short paragraphs (2-4 sentences each)
            * Simple sentences for 9-10 year old comprehension
            * Include technical details where relevant but explain them simply
            - Length: 800-1200 words
            - SEO:
            * Include relevant keywords naturally
            * Structure content for readability
            * Use descriptive subheadings

            Title: {title}

            Sources to synthesize:
            {combined_sources}

            Return your response in this exact JSON format, with proper Markdown formatting and escaped newlines:
            {{
                "content": "# Main Title\\n\\nIntroduction paragraphs...\\n\\n## First Section\\n\\nContent...\\n\\n## Second Section\\n\\nContent...",
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
            
            Important: Ensure you:
            1. Synthesize information from ALL provided sources
            2. Use proper markdown formatting
            3. Keep paragraphs short and engaging
            4. Explain technical concepts simply
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
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].strip()
                
                print("Response text:", response_text[:200])  # Debug print
                
                result = json.loads(response_text)
                return result
                
            except Exception as e:
                print(f"Error processing Claude response: {str(e)}")
                return {
                    "content": response.content[0].text if hasattr(response, 'content') else "Error generating content",
                    "excerpt": "Article about " + title,
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
            
            # Create markdown content
            return self.template.format(**template_vars)
            
        except Exception as e:
            print(f"Error creating markdown: {str(e)}")
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