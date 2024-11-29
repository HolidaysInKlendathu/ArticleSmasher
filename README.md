# ArticleSmasher

ArticleSmasher is an automated content generation and management system that processes multiple news sources to create well-structured, SEO-optimized articles. The system integrates with Google Sheets for source management, uses Claude AI for content generation, and handles storage through MinIO and a MySQL database.

![ArticleSmasher Project Flow](project-diagram.svg)

## Features

- **Source Management**: Fetches article sources from Google Sheets
- **Content Scraping**: Extracts content from multiple source URLs
- **AI Content Generation**: Uses Claude AI to generate new articles from multiple sources
- **Category Detection**: Automatically categorizes content using keyword analysis
- **Storage Solutions**:
  - Saves markdown files to MinIO storage
  - Creates database records with article metadata
  - Manages relationships between articles, categories, and tags

## System Requirements

- Python 3.9+
- MySQL/MariaDB database
- MinIO object storage
- Google Sheets API access
- Anthropic Claude API access

## Installation

1. Clone the repository:

```bash
git clone https://github.com/HolidaysInKlendathu/articlesmasher.git
cd articlesmasher
```

2. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with the following configuration:

```env
# API Keys
ANTHROPIC_KEY=your_anthropic_key_here

# Database Configuration
DATABASE_URL=mysql://user:password@host:port/database_name

# Google Sheets Configuration
SHEET_ID=your_sheet_id_here
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json

# MinIO Configuration
MINIO_ENDPOINT=your_minio_endpoint
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET=your_bucket_name
MINIO_USE_SSL=true
```

## Usage

1. Set up your Google Sheet with article sources:

   - Column A: Article title
   - Columns B onwards: Source URLs

2. Run the article processor:

```bash
python article_processor.py
```

## Project Structure

```
articlesmasher/
├── article_processor.py     # Main processing script
├── categories.json         # Category configuration
├── article_template.md     # Markdown template
├── .env                    # Environment variables
└── google-credentials.json # Google API credentials
```

## Features in Detail

### Content Generation

- Processes multiple source articles
- Generates new content using Claude AI
- Ensures readability for 9-10 year old level
- Creates SEO-optimized metadata

### Category Management

- Automated category detection
- Support for multiple categories per article
- Hierarchical category structure with subcategories

### Storage

- MinIO for markdown file storage
- MySQL database for article metadata
- Handles category and tag relationships

## Database Schema

The system uses the following main tables:

- Article: Stores article content and metadata
- Category: Manages content categories
- Tag: Handles article tags
- Relationship tables for categories and tags

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

[MIT License](LICENSE)
