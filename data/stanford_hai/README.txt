Stanford HAI AI Index - Downloaded Raw Data
============================================
Downloaded: March 2026

SUMMARY
-------
Raw CSV data from the Stanford HAI AI Index Report has been downloaded for 
two report years: 2024 and 2023. Data comes from the publicly shared Google 
Drive folders linked on the report pages.

SOURCE URLS
-----------
- Main page:    https://hai.stanford.edu/ai-index
- 2024 report:  https://hai.stanford.edu/ai-index/2024-ai-index-report
- 2023 report:  https://hai.stanford.edu/ai-index/2023-ai-index-report
- 2025 report:  https://hai.stanford.edu/ai-index/2025-ai-index-report
  (2025 data folder not found publicly; no download link found on page)

GOOGLE DRIVE SOURCE FOLDERS
-----------------------------
- 2024: https://drive.google.com/drive/folders/1_9oLjgrgZlRdAWOY1fhGNlPv9nSfDDwN
- 2023: https://drive.google.com/drive/folders/1ma9WZJzKreS8f2It1rMy_KkkbX6XwDOK

FILES DOWNLOADED
----------------
2024/all_csvs/    - 300 CSV files (284 successfully downloaded, ~16 failed due to rate limits)
                    Total size: ~1.5 MB
                    Source: 2024 AI Index Report data (Chapters 1-9)

2023/all_csvs/    - 235 CSV files (214 successfully downloaded, ~21 failed due to rate limits)
                    Total size: ~1.1 MB
                    Source: 2023 AI Index Report data (Chapters 1-8)

Also in 2024/ folder (from initial gdown run):
2024/1. Research and Development/Data/   - 12 CSV files (Ch.1 R&D data)
2024/1. Research and Development/Charts/ - 42 PDF chart files (Ch.1 only)

CHAPTER COVERAGE (2024)
------------------------
Chapter 1: Research and Development (publications, patents, citations)
Chapter 2: Technical Performance
Chapter 3: Responsible AI
Chapter 4: Economy (investment, job postings, workforce)
Chapter 5: Science and Medicine
Chapter 6: Education
Chapter 7: Policy and Governance
Chapter 8: Diversity
Chapter 9: Public Opinion

CHAPTER COVERAGE (2023)
------------------------
Chapter 1: Research and Development
Chapter 2: Technical Performance
Chapter 3: Technical AI Ethics
Chapter 4: The Economy
Chapter 5: Education
Chapter 6: Policy and Governance
Chapter 7: Diversity
Chapter 8: Public Opinion

FILE NAMING CONVENTION
-----------------------
Files are named fig_X.Y.Z.csv where:
  X = chapter number
  Y = section number
  Z = figure number

COUNTRY-LEVEL DATA AVAILABLE
------------------------------
Key files with geographic/country-level data include:
- fig_1.1.5.csv   - AI publications by country
- fig_1.2.3.csv   - AI patents by country/region
- fig_1.2.5.csv   - AI patent filings by geography
- fig_1.2.6.csv   - AI patents by country
- fig_1.2.7.csv   - AI patents trend by country
- fig_4.2.1.csv   - AI job postings by geographic area
- fig_4.2.X.csv   - Various labor market data
- Various fig_1.X.X.csv - Research output by country

Also available from Global AI Vibrancy Tool:
  https://hai.stanford.edu/ai-index/global-vibrancy-tool
  (36 countries ranked, 2017-2024, on 42 indicators across 7 pillars)

NOTES ON 2025 DATA
-------------------
The 2025 AI Index Report is available as PDF:
  https://hai.stanford.edu/assets/files/hai_ai_index_report_2025.pdf
No public Google Drive raw data folder link was found for 2025 as of March 2026.
The chapter PDFs are available individually at:
  https://hai.stanford.edu/assets/files/hai_ai-index-report-2025_chapter1_final.pdf
  (and similar URLs for chapters 2-8)

FAILED DOWNLOADS
-----------------
Some files failed due to Google Drive rate limiting (returned empty responses).
These are primarily from Technical Performance (Ch.2) and some Economy/Policy sections.
The failures appear to be rate-limit related, not permission issues, as other files
in the same folders downloaded successfully.

To retry failed downloads, run:
  python3 /tmp/download_csvs.py 2024
  python3 /tmp/download_csvs.py 2023
(Script uses --remaining-ok logic, skips already-downloaded files)
