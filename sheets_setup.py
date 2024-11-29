import os
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from typing import List, Dict, Optional
from slugify import slugify

load_dotenv()

class GoogleSheetsManager:
    def __init__(self, credentials_path: str):
        try:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")
                
            self.creds = Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets.readonly']
            )
            self.service = build('sheets', 'v4', credentials=self.creds)
        except Exception as e:
            raise Exception(f"Failed to initialize Google Sheets manager: {str(e)}")
        
    def get_articles_from_sheet(self, spreadsheet_id: str, range_name: str = 'A2:Z') -> List[dict]:
        """
        Read articles from sheet, handling multiple source URLs per title
        Returns list of articles with their source URLs
        """
        try:
            sheet = self.service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                print("No data found in sheet.")
                return []
            
            print("\nReading articles from sheet:")
            articles = []
            for idx, row in enumerate(values, start=2):
                if not row:
                    print(f"Row {idx}: Empty row, skipping")
                    continue
                
                # Get title from column A
                title = row[0].strip()
                slug = slugify(title)
                final_url = f"https://riskbagel.com/articles/{slug}"
                
                # Get source URLs from remaining columns
                source_urls = []
                for col_idx, cell in enumerate(row[1:], start=1):
                    if cell and cell.strip():
                        source_urls.append({
                            'column': chr(65 + col_idx),  # Convert to column letter (B, C, D, etc.)
                            'url': cell.strip()
                        })
                
                article_data = {
                    "row": idx,
                    "title": title,
                    "slug": slug,
                    "final_url": final_url,
                    "source_urls": source_urls
                }
                
                articles.append(article_data)
                
                # Print article details
                print(f"\nRow {idx}:")
                print(f"Title: '{title}'")
                print(f"Slug: {slug}")
                print(f"Final URL: {final_url}")
                print("Source URLs:")
                for source in source_urls:
                    print(f"  Column {source['column']}: {source['url']}")
                print("---")
            
            return articles
            
        except Exception as e:
            print(f"Error reading from sheet: {str(e)}")
            return []

def test_setup():
    try:
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', './google-credentials.json')
        sheet_id = os.getenv('SHEET_ID')
        
        print("\n=== Google Sheets Connection Test ===")
        print(f"Using credentials from: {credentials_path}")
        print(f"Sheet ID: {sheet_id}")
        
        if not sheet_id:
            raise ValueError("SHEET_ID environment variable not set")
        
        # Initialize sheets manager
        sheets_manager = GoogleSheetsManager(credentials_path)
        
        # Test fetching articles
        print("\nAttempting to read from sheet...")
        articles = sheets_manager.get_articles_from_sheet(sheet_id)
        
        print("\n=== Test Results ===")
        if articles:
            print(f"Success! Found {len(articles)} articles")
            print("\nSummary:")
            for article in articles:
                print(f"\nTitle: {article['title']}")
                print(f"Number of sources: {len(article['source_urls'])}")
                for source in article['source_urls']:
                    print(f"  - Column {source['column']}: {source['url']}")
        else:
            print("No articles found in the sheet.")
            print("\nPlease check that:")
            print("1. Your titles are in Column A")
            print("2. Source URLs are in columns B onwards")
            print("3. Data starts from row 2")
            print("4. The sheet has been shared with the service account email")
            
    except Exception as e:
        print(f"\nSetup test failed: ❌")
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    test_setup()