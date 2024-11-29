import pymysql
from urllib.parse import urlparse

class DatabaseHandler:
    def __init__(self, database_url: str):
        parsed = urlparse(database_url)
        self.connection_params = {
            'host': parsed.hostname,
            'user': parsed.username,
            'password': parsed.password,
            'db': parsed.path[1:],
            'port': parsed.port
        }

    def connect(self):
        return pymysql.connect(**self.connection_params)

    def create_or_update_article(self, article_data: Dict) -> bool:
        """Create or update article in the database"""
        try:
            with self.connect() as connection:
                with connection.cursor() as cursor:
                    # Check if article exists
                    cursor.execute("SELECT id FROM Articles WHERE slug = %s", (article_data["slug"],))
                    existing_article = cursor.fetchone()

                    if existing_article:
                        # Update existing article
                        sql = """
                        UPDATE Articles 
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
                        INSERT INTO Articles (
                            id, title, slug, content, markdownUrl, excerpt, coverImage,
                            readingTime, wordCount, publishedAt, status, metaTitle,
                            metaDescription, featured, spotlight, evergreen, sponsored,
                            sponsorName, partnerContent, affiliate, crowdsourced,
                            premium, hasVideo, hasAudio, hasGallery, authorId, updatedAt
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                        )
                        """
                        cursor.execute(sql, (
                            article_data.get("id", self.generate_cuid()),
                            article_data["title"],
                            article_data["slug"],
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
                            article_data.get("authorId", "cm3pw0m9u00026hqfvtiqmtcw")  # Default author ID
                        ))

                    # Handle categories
                    if "categories" in article_data:
                        # First, remove existing category relationships
                        cursor.execute("DELETE FROM _ArticleToCategory WHERE A = %s", (existing_article[0] if existing_article else article_data["id"],))
                        
                        # Add new category relationships
                        for category in article_data["categories"]:
                            cursor.execute("""
                                INSERT INTO _ArticleToCategory (A, B)
                                VALUES (%s, (SELECT id FROM Category WHERE slug = %s))
                            """, (existing_article[0] if existing_article else article_data["id"], category))

                    # Handle tags
                    if "tags" in article_data:
                        # First, remove existing tag relationships
                        cursor.execute("DELETE FROM _ArticleToTag WHERE A = %s", (existing_article[0] if existing_article else article_data["id"],))
                        
                        # Add new tag relationships
                        for tag in article_data["tags"]:
                            cursor.execute("""
                                INSERT INTO _ArticleToTag (A, B)
                                VALUES (%s, (SELECT id FROM Tag WHERE name = %s))
                            """, (existing_article[0] if existing_article else article_data["id"], tag))

                connection.commit()
                return True
        except Exception as e:
            print(f"Database error creating/updating article: {e}")
            return False

    def generate_cuid(self) -> str:
        """Generate a CUID for new articles"""
        import time
        import random
        import string
        
        timestamp = int(time.time() * 1000)
        counter = random.randint(0, 999999)
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        return f"cl{timestamp:x}{counter:x}{rand}"