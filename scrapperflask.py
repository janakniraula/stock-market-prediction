# import pandas as pd
# from playwright.sync_api import sync_playwright, TimeoutError
# import time
# import os

# DATA_DIR = "data"


# def scrape_merolagani(symbol: str, last_date: str = None):
#     url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
#     all_data = []

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)  # ✅ headless mode for API
#         page = browser.new_page()
#         page.goto(url, timeout=60000)

#         try:
#             page.click("text=Price History", timeout=10000)
#         except Exception:
#             pass  # tab might be active already

#         try:
#             page.wait_for_selector(
#                 "#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr",
#                 timeout=60000
#             )
#         except TimeoutError:
#             browser.close()
#             return pd.DataFrame()

#         current_page = 1
#         stop_scraping = False

#         while not stop_scraping:
#             time.sleep(1.5)

#             rows = page.query_selector_all(
#                 "#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr"
#             )

#             for i, row in enumerate(rows):
#                 if i == 0:  # skip header
#                     continue
#                 cols = row.query_selector_all("td")
#                 if len(cols) >= 9:
#                     row_data = [col.inner_text().strip() for col in cols[1:]]  # skip index
#                     if len(row_data) >= 8:
#                         try:
#                             web_date = pd.to_datetime(row_data[0], format="%Y/%m/%d").strftime("%Y-%m-%d")
#                         except:
#                             continue

#                         if last_date and web_date <= last_date:
#                             stop_scraping = True
#                             break

#                         all_data.append([web_date] + row_data[1:])

#             if stop_scraping:
#                 break

#             # Next page
#             next_btn = page.query_selector("a[title='Next Page']")
#             if not next_btn:
#                 break
#             onclick_attr = next_btn.get_attribute("onclick")
#             if onclick_attr and "changePageIndex" in onclick_attr:
#                 next_btn.click()
#                 time.sleep(2)
#                 current_page += 1
#             else:
#                 break

#         browser.close()

#     if not all_data:
#         return pd.DataFrame()

#     df = pd.DataFrame(
#         all_data,
#         columns=["Date", "Close", "% Change", "High", "Low", "Open", "Volume", "Turnover"]
#     )
#     df["Symbol"] = symbol
#     df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]

#     for col in ["Open", "High", "Low", "Close"]:
#         df[col] = df[col].str.replace(",", "").astype(float)
#     df["Volume"] = df["Volume"].str.replace(",", "").astype(int)

#     df = df.sort_values("Date").reset_index(drop=True)
#     return df


# def update_csv(symbol: str):
#     os.makedirs(DATA_DIR, exist_ok=True)
#     file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

#     last_date = None
#     df_old = pd.DataFrame()

#     if os.path.exists(file_path):
#         df_old = pd.read_csv(file_path)
#         df_old = df_old.loc[:, ~df_old.columns.str.contains('^Unnamed')]
#         df_old.columns = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
#         df_old["Date"] = pd.to_datetime(df_old["Date"], errors='coerce').dt.strftime("%Y-%m-%d")
#         df_old = df_old.dropna(subset=["Date"])
#         if not df_old.empty:
#             last_date = df_old["Date"].max()

#     df_new = scrape_merolagani(symbol, last_date)

#     if df_new.empty:
#         return {"message": "No new data scraped", "file": file_path, "rows": len(df_old)}

#     df = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new
#     df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
#     df.to_csv(file_path, index=False)

#     return {
#         "message": "Data updated successfully",
#         "file": file_path,
#         "rows": len(df),
#         "date_range": [df["Date"].iloc[0], df["Date"].iloc[-1]]
#     }
# import pandas as pd
# from playwright.sync_api import sync_playwright, TimeoutError
# import time
# import os
# import random

# DATA_DIR = "data"

# # -----------------------------
# # Helper: Wait until data rows appear
# # -----------------------------   
# def wait_for_data(page, retries=6, delay=2):
#     """Wait until at least one data row is visible in the price table."""
#     for _ in range(retries):
#         rows = page.query_selector_all(
#             "#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr"
#         )
#         if len(rows) > 1:  # header + data
#             return True
#         time.sleep(delay)
#     return False


# # -----------------------------
# # Core Scraper Function
# # -----------------------------
# def scrape_merolagani(symbol: str, last_date: str = None):
#     url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
#     all_data = []

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page()
#         page.goto(url, timeout=60000)

#         # --- Ensure "Price History" tab is active ---
#         try:
#             page.click("text=Price History", timeout=10000)
#             page.wait_for_load_state("networkidle")
#         except Exception:
#             pass

#         # --- Wait until the table data is actually present ---
#         if not wait_for_data(page):
#             print(f"⚠️ Data not loaded for {symbol}, retrying page reload...")
#             page.reload()
#             try:
#                 page.click("text=Price History", timeout=10000)
#                 page.wait_for_load_state("networkidle")
#             except Exception:
#                 pass
#             if not wait_for_data(page):
#                 print(f"❌ Still no data for {symbol}, skipping.")
#                 browser.close()
#                 return pd.DataFrame()

#         current_page = 1
#         stop_scraping = False

#         while not stop_scraping:
#             # Random small delay to prevent throttling
#             time.sleep(random.uniform(1.5, 2.5))

#             rows = page.query_selector_all(
#                 "#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr"
#             )

#             for i, row in enumerate(rows):
#                 if i == 0:  # skip header
#                     continue
#                 cols = row.query_selector_all("td")
#                 if len(cols) >= 9:
#                     row_data = [col.inner_text().strip() for col in cols[1:]]  # skip index
#                     if len(row_data) >= 8:
#                         try:
#                             web_date = pd.to_datetime(row_data[0], format="%Y/%m/%d").strftime("%Y-%m-%d")
#                         except:
#                             continue

#                         if last_date and web_date <= last_date:
#                             stop_scraping = True
#                             break

#                         all_data.append([web_date] + row_data[1:])

#             if stop_scraping:
#                 break

#             # --- Go to Next Page if exists ---
#             next_btn = page.query_selector("a[title='Next Page']")
#             if not next_btn:
#                 break

#             onclick_attr = next_btn.get_attribute("onclick")
#             if onclick_attr and "changePageIndex" in onclick_attr:
#                 try:
#                     next_btn.click()
#                     page.wait_for_load_state("networkidle")
#                     if not wait_for_data(page):
#                         break
#                     current_page += 1
#                 except Exception:
#                     break
#             else:
#                 break

#         browser.close()

#     if not all_data:
#         return pd.DataFrame()

#     # --- Clean DataFrame ---
#     df = pd.DataFrame(
#         all_data,
#         columns=["Date", "Close", "% Change", "High", "Low", "Open", "Volume", "Turnover"]
#     )
#     df["Symbol"] = symbol
#     df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]

#     # Convert numeric columns
#     for col in ["Open", "High", "Low", "Close"]:
#         df[col] = df[col].str.replace(",", "", regex=False).astype(float)
#     df["Volume"] = df["Volume"].str.replace(",", "", regex=False).astype(int)

#     df = df.sort_values("Date").reset_index(drop=True)
#     return df


# # -----------------------------
# # Retry Wrapper for Reliability
# # -----------------------------
# def scrape_with_retry(symbol, last_date=None, max_retries=2):
#     for attempt in range(1, max_retries + 1):
#         df = scrape_merolagani(symbol, last_date)
#         if not df.empty:
#             if attempt > 1:
#                 print(f"✅ Success for {symbol} on retry #{attempt}")
#             return df
#         print(f"⚠️ Attempt {attempt} failed for {symbol}, retrying...")
#         time.sleep(random.uniform(2, 4))
#     print(f"❌ Failed to scrape {symbol} after {max_retries} attempts.")
#     return pd.DataFrame()


# # -----------------------------
# # Update CSV
# # -----------------------------
# def update_csv(symbol: str):
#     os.makedirs(DATA_DIR, exist_ok=True)
#     file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

#     last_date = None
#     df_old = pd.DataFrame()

#     if os.path.exists(file_path):
#         df_old = pd.read_csv(file_path)
#         df_old = df_old.loc[:, ~df_old.columns.str.contains('^Unnamed')]
#         df_old.columns = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
#         df_old["Date"] = pd.to_datetime(df_old["Date"], errors='coerce').dt.strftime("%Y-%m-%d")
#         df_old = df_old.dropna(subset=["Date"])
#         if not df_old.empty:
#             last_date = df_old["Date"].max()

#     df_new = scrape_with_retry(symbol, last_date)

#     if df_new.empty:
#         return {"message": "No new data scraped", "file": file_path, "rows": len(df_old)}

#     df = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new
#     df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
#     df.to_csv(file_path, index=False)

#     return {
#         "message": "Data updated successfully",
#         "file": file_path,
#         "rows": len(df),
#         "date_range": [df["Date"].iloc[0], df["Date"].iloc[-1]]
#     }


import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError
import time
import os
import random
from sqlalchemy import create_engine, Column, Integer, String, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATA_DIR = "data"

# -----------------------------
# SQLAlchemy Setup
# -----------------------------
DATABASE_URL = "postgresql://root:1234@localhost:5432/stock-market-prediction"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# -----------------------------
# Stocks Model
# -----------------------------
class Stock(Base):
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_name = Column(String, nullable=False)
    company_code = Column(String, unique=True, nullable=False)


# Create table if not exists
Base.metadata.create_all(bind=engine)


# -----------------------------
# Save Company Info
# -----------------------------
def save_stock_info(company_name, company_code):
    """Save company info into PostgreSQL using SQLAlchemy ORM."""
    session = SessionLocal()
    try:
        existing = session.query(Stock).filter_by(company_code=company_code).first()
        if not existing:
            stock = Stock(company_name=company_name, company_code=company_code)
            session.add(stock)
            session.commit()
        else:
            # Optional: update name if changed
            if existing.company_name != company_name:
                existing.company_name = company_name
                session.commit()
    except Exception as e:
        session.rollback()
        print(f"⚠️ Error saving to DB: {e}")
    finally:
        session.close()


# -----------------------------
# Helper: Wait until data rows appear
# -----------------------------   
def wait_for_data(page, retries=6, delay=2):
    """Wait until at least one data row is visible in the price table."""
    for _ in range(retries):
        rows = page.query_selector_all(
            "#ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice table.table-bordered tr"
        )
        if len(rows) > 1:  # header + data
            return True
        time.sleep(delay)
    return False


# -----------------------------
# Core Scraper Function
# -----------------------------
def scrape_merolagani(symbol: str, last_date: str = None):
    url = f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
    all_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # --- Extract company name and code ---
        company_name = None
        company_code = None
        try:
            element = page.query_selector("#ctl00_ContentPlaceHolder1_CompanyDetail1_companyName")
            if element:
                text_val = element.inner_text().strip()
                if "(" in text_val and ")" in text_val:
                    company_name = text_val.split("(")[0].strip()
                    company_code = text_val.split("(")[1].split(")")[0].strip()
                else:
                    company_name = text_val
                    company_code = symbol
        except Exception as e:
            print(f"⚠️ Could not extract company name for {symbol}: {e}")

        # --- Save to DB ---
        if company_name and company_code:
            print(f"{company_name} {company_code}")
            save_stock_info(company_name, company_code)

        # --- Ensure "Price History" tab is active ---
        try:
            page.click("text=Price History", timeout=10000)
            page.wait_for_load_state("networkidle")
        except Exception:
            pass

        # --- Wait until the table data is actually present ---
        if not wait_for_data(page):
            print(f"⚠️ Data not loaded for {symbol}, retrying page reload...")
            page.reload()
            try:
                page.click("text=Price History", timeout=10000)
                page.wait_for_load_state("networkidle")
            except Exception:
                pass
            if not wait_for_data(page):
                print(f"❌ Still no data for {symbol}, skipping.")
                browser.close()
                return pd.DataFrame()

        stop_scraping = False

        while not stop_scraping:
            time.sleep(random.uniform(1.5, 2.5))

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
                            web_date = pd.to_datetime(row_data[0], format="%Y/%m/%d").strftime("%Y-%m-%d")
                        except:
                            continue

                        if last_date and web_date <= last_date:
                            stop_scraping = True
                            break

                        all_data.append([web_date] + row_data[1:])

            if stop_scraping:
                break

            # --- Go to Next Page if exists ---
            next_btn = page.query_selector("a[title='Next Page']")
            if not next_btn:
                break

            onclick_attr = next_btn.get_attribute("onclick")
            if onclick_attr and "changePageIndex" in onclick_attr:
                try:
                    next_btn.click()
                    page.wait_for_load_state("networkidle")
                    if not wait_for_data(page):
                        break
                except Exception:
                    break
            else:
                break

        browser.close()

    if not all_data:
        return pd.DataFrame()

# --- Clean DataFrame (without company name/code) ---
    df = pd.DataFrame(
        all_data,
        columns=["Date", "Close", "% Change", "High", "Low", "Open", "Volume", "Turnover"]
    )

    # Keep only relevant columns for CSV
    df["Symbol"] = symbol
    df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]

    # Convert numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].str.replace(",", "", regex=False).astype(float)
    df["Volume"] = df["Volume"].str.replace(",", "", regex=False).astype(int)

    df = df.sort_values("Date").reset_index(drop=True)
    return df



# -----------------------------
# Retry Wrapper for Reliability
# -----------------------------
def scrape_with_retry(symbol, last_date=None, max_retries=2):
    for attempt in range(1, max_retries + 1):
        df = scrape_merolagani(symbol, last_date)
        if not df.empty:
            if attempt > 1:
                print(f"✅ Success for {symbol} on retry #{attempt}")
            return df
        print(f"⚠️ Attempt {attempt} failed for {symbol}, retrying...")
        time.sleep(random.uniform(2, 4))
    print(f"❌ Failed to scrape {symbol} after {max_retries} attempts.")
    return pd.DataFrame()


# -----------------------------
# Update CSV
# -----------------------------
def update_csv(symbol: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

    last_date = None
    df_old = pd.DataFrame()

    if os.path.exists(file_path):
        df_old = pd.read_csv(file_path)
        df_old = df_old.loc[:, ~df_old.columns.str.contains('^Unnamed')]
        df_old.columns = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
        df_old["Date"] = pd.to_datetime(df_old["Date"], errors='coerce').dt.strftime("%Y-%m-%d")
        df_old = df_old.dropna(subset=["Date"])
        if not df_old.empty:
            last_date = df_old["Date"].max()

    df_new = scrape_with_retry(symbol, last_date)

    if df_new.empty:
        return {"message": "No new data scraped", "file": file_path, "rows": len(df_old)}

    df = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new
    df = df.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
    df.to_csv(file_path, index=False)

    return {
        "message": "Data updated successfully",
        "file": file_path,
        "rows": len(df),
        "date_range": [df["Date"].iloc[0], df["Date"].iloc[-1]]
    }


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":
    symbol = "NRN"
    result = update_csv(symbol)
    print(result)
