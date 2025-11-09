# import sys
# import pandas as pd
# from playwright.sync_api import sync_playwright, TimeoutError
# import time

# def scrape_merolagani(symbol: str):
#     url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
#     all_data = []

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=False)  # 👀 Browser is visible
#         page = browser.new_page()
#         page.goto(url, timeout=60000)

#         print("🕒 Please manually click the 'Price History' tab in the opened browser window.")
#         print("💡 After clicking, the script will automatically detect the table and start scraping.")
        
#         # Wait for the history table to appear after manual click
#         try:
#             # Wait for the actual table with data to load
#             page.wait_for_selector("#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr", timeout=60000)
#             print("✅ Price History table detected!")
#         except TimeoutError:
#             print("❌ Price History table did not load in time.")
#             browser.close()
#             return pd.DataFrame()

#         current_page = 1
#         while True:
#             print(f"🔍 Scraping page {current_page}...")

#             # Wait a bit for the page to fully load
#             time.sleep(2)

#             # Get all table rows (skip header row)
#             rows = page.query_selector_all("#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr")
            
#             for i, row in enumerate(rows):
#                 if i == 0:  # Skip header row
#                     continue
                    
#                 cols = row.query_selector_all("td")
#                 if len(cols) >= 9:  # Make sure we have all columns
#                     row_data = []
#                     for j, col in enumerate(cols):
#                         if j == 0:  # Skip the index column
#                             continue
#                         row_data.append(col.inner_text().strip())
                    
#                     if len(row_data) >= 8:  # We should have 8 columns after removing index
#                         all_data.append(row_data)

#             # Check if there's a "Next" button and if it's clickable
#             try:
#                 next_btn = page.query_selector("a[title='Next Page']")
#                 if next_btn:
#                     # Check if the next button is actually clickable (not disabled)
#                     onclick_attr = next_btn.get_attribute("onclick")
#                     if onclick_attr and "changePageIndex" in onclick_attr:
#                         print(f"📄 Moving to page {current_page + 1}...")
#                         next_btn.click()
                        
#                         # Wait for the new page to load
#                         time.sleep(3)
                        
#                         # Wait for the table to update
#                         page.wait_for_selector("#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr", timeout=10000)
#                         current_page += 1
#                     else:
#                         print("🏁 No more pages available.")
#                         break
#                 else:
#                     print("🏁 Next button not found. Reached last page.")
#                     break
                    
#             except TimeoutError:
#                 print("⚠️ Timed out waiting for next page to load.")
#                 break
#             except Exception as e:
#                 print(f"⚠️ Error navigating to next page: {e}")
#                 break

#         browser.close()

#     # Create DataFrame with proper column names
#     if all_data:
#         df = pd.DataFrame(all_data, columns=["Date", "LTP", "% Change", "High", "Low", "Open", "Quantity", "Turnover"])
#         return df
#     else:
#         return pd.DataFrame()

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python scraper.py SYMBOL")
#         print("Example: python scraper.py NABIL")
#         sys.exit(1)

#     symbol = sys.argv[1].upper()
#     print(f"🚀 Starting scraper for symbol: {symbol}")
    
#     df = scrape_merolagani(symbol)

#     if not df.empty:
#         filename = f"{symbol}_price_history.csv"
#         df.to_csv(filename, index=False)
#         print(f"\n✅ Successfully scraped {len(df)} rows!")
#         print(f"📁 Data saved to: {filename}")
#         print(f"📊 Date range: {df['Date'].iloc[-1]} to {df['Date'].iloc[0]}")
#     else:
#         print("⚠️ No data was scraped. Please check the symbol and try again.")


import sys
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError
import time
import os

DATA_DIR = "data2"

def scrape_merolagani(symbol: str, last_date: str = None):
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    all_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        print("🕒 Please manually click the 'Price History' tab in the opened browser window.")
        print("💡 After clicking, the script will detect the table and start scraping.")

        try:
            page.wait_for_selector(
                "#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr",
                timeout=60000
            )
            print("✅ Price History table detected!")
        except TimeoutError:
            print("❌ Price History table did not load in time.")
            browser.close()
            return pd.DataFrame()

        current_page = 1
        stop_scraping = False

        while not stop_scraping:
            print(f"🔍 Scraping page {current_page}...")
            time.sleep(2)

            rows = page.query_selector_all(
                "#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr"
            )

            for i, row in enumerate(rows):
                if i == 0:  # skip header
                    continue
                cols = row.query_selector_all("td")
                if len(cols) >= 9:
                    row_data = [col.inner_text().strip() for col in cols[1:]]  # skip index
                    if len(row_data) >= 8:
                        try:
                            # Convert to YYYY-MM-DD string
                            web_date = pd.to_datetime(row_data[0], format="%Y/%m/%d").strftime("%Y-%m-%d")
                        except:
                            continue

                        # Stop scraping if we reach an existing date
                        if last_date and web_date <= last_date:
                            print(f"⛔ Reached existing date {web_date}, stopping scrape.")
                            stop_scraping = True
                            break

                        all_data.append([web_date] + row_data[1:])  # prepend formatted date

            if stop_scraping:
                break

            # Check for next page
            try:
                next_btn = page.query_selector("a[title='Next Page']")
                if next_btn:
                    onclick_attr = next_btn.get_attribute("onclick")
                    if onclick_attr and "changePageIndex" in onclick_attr:
                        print(f"📄 Moving to page {current_page + 1}...")
                        next_btn.click()
                        time.sleep(3)
                        page.wait_for_selector(
                            "#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr",
                            timeout=10000
                        )
                        current_page += 1
                    else:
                        break
                else:
                    break
            except Exception as e:
                print(f"⚠️ Error navigating to next page: {e}")
                break

        browser.close()

    if not all_data:
        return pd.DataFrame()

    # Convert scraped data → DataFrame
    df = pd.DataFrame(
        all_data,
        columns=["Date", "Close", "% Change", "High", "Low", "Open", "Volume", "Turnover"]
    )

    df["Symbol"] = symbol

    # keep only required columns in exact order
    df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]

    # Convert numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].str.replace(",", "").astype(float)
    df["Volume"] = df["Volume"].str.replace(",", "").astype(int)

    # Ensure order oldest → newest
    df = df.sort_values("Date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scrapper.py SYMBOL")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    print(f"🚀 Starting scraper for symbol: {symbol}")

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    last_date = None
    df_old = pd.DataFrame()

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df_old = df_old.loc[:, ~df_old.columns.str.contains('^Unnamed')]
        df_old.columns = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
        # Convert to YYYY-MM-DD string
        df_old["Date"] = pd.to_datetime(df_old["Date"], errors='coerce').dt.strftime("%Y-%m-%d")
        df_old = df_old.dropna(subset=["Date"])
        if not df_old.empty:
            last_date = df_old["Date"].max()
            print(f"📅 Last saved date in {symbol}.csv: {last_date}")

    # Scrape new data
    df_new = scrape_merolagani(symbol, last_date)

    if df_new.empty:
        print("⚠️ No new data scraped.")
        print(f"📊 CSV already up-to-date: {file_path}")
        sys.exit(0)

    # Concatenate safely
    if not df_old.empty:
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new.copy()

    df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)

    df.to_csv(file_path, index=False)
    print(f"\n✅ Saved {len(df)} rows to {file_path}")
    print(f"📊 Date range: {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")
