import requests
import os
from pathlib import Path
import json
import time
import logging
from datetime import datetime
import re
import sys

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# MOSDAC API ENDPOINTS
# ============================================================

token_url = "https://mosdac.gov.in/download_api/gettoken"
search_url = "https://mosdac.gov.in/apios/datasets.json"
download_url = "https://mosdac.gov.in/download_api/download"
refresh_url = "https://mosdac.gov.in/download_api/refresh-token"
logout_url = "https://mosdac.gov.in/download_api/logout"


# ============================================================
# GLOBALS
# ============================================================

bearer_token = None
refresh_token = None
config = None

logger = logging.getLogger("MOSDAC")
logger.setLevel(logging.INFO)


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():

    global config

    config_path = Path("config.json")

    if not config_path.exists():
        print("[ERROR] config.json not found.")
        sys.exit(1)

    try:

        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        print("[INFO] Configuration loaded successfully.")

    except Exception as e:

        print(f"[ERROR] Could not load config.json: {e}")
        sys.exit(1)


# ============================================================
# LOGGING
# ============================================================

def setup_logging():

    global logger

    download_settings = config.get(
        "download_settings",
        {}
    )

    generate_logs = download_settings.get(
        "generate_error_logs",
        True
    )

    if not generate_logs:
        return

    log_directory = download_settings.get(
        "error_logs_dir",
        ""
    )

    if not log_directory:
        log_directory = "logs"

    Path(log_directory).mkdir(
        parents=True,
        exist_ok=True
    )

    log_file = (
        Path(log_directory)
        / f"mosdac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    handler = logging.FileHandler(
        log_file,
        encoding="utf-8"
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    print(f"[INFO] Log file: {log_file}")


# ============================================================
# AUTHENTICATION
# ============================================================

def authenticate():

    global bearer_token
    global refresh_token

    credentials = config.get(
        "user_credentials",
        {}
    )

    username = credentials.get(
        "username/email",
        ""
    )

    password = credentials.get(
        "password",
        ""
    )

    if not username or not password:

        print(
            "[ERROR] Username/email or password missing "
            "in config.json."
        )

        return False

    payload = {
        "username": username,
        "password": password
    }

    try:

        print("[INFO] Authenticating with MOSDAC...")

        response = requests.post(
            token_url,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        bearer_token = (
            result.get("token")
            or result.get("access_token")
            or result.get("bearer_token")
        )

        refresh_token = result.get(
            "refresh_token"
        )

        if not bearer_token:

            print(
                "[ERROR] Authentication succeeded but "
                "no bearer token was returned."
            )

            logger.error(
                f"Authentication response: {result}"
            )

            return False

        print(
            "[INFO] Authentication successful."
        )

        return True

    except Exception as e:

        print(
            f"[ERROR] Authentication failed: {e}"
        )

        logger.exception(
            "Authentication failure"
        )

        return False


# ============================================================
# TOKEN REFRESH
# ============================================================

def refresh_authentication():

    global bearer_token
    global refresh_token

    if not refresh_token:
        return False

    try:

        response = requests.post(
            refresh_url,
            json={
                "refresh_token": refresh_token
            },
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        new_token = (
            result.get("token")
            or result.get("access_token")
            or result.get("bearer_token")
        )

        if new_token:

            bearer_token = new_token

            print(
                "[INFO] Authentication token refreshed."
            )

            return True

        return False

    except Exception as e:

        logger.exception(
            f"Token refresh failed: {e}"
        )

        return False


# ============================================================
# DATASET SEARCH
# ============================================================

def search_results():

    search_parameters = config.get(
        "search_parameters",
        {}
    )

    dataset_id = search_parameters.get(
        "datasetId",
        ""
    )

    start_time = search_parameters.get(
        "startTime",
        ""
    )

    end_time = search_parameters.get(
        "endTime",
        ""
    )

    count = search_parameters.get(
        "count",
        ""
    )

    bounding_box = search_parameters.get(
        "boundingBox",
        ""
    )

    gid = search_parameters.get(
        "gId",
        ""
    )

    params = {
        "datasetId": dataset_id,
        "startTime": start_time,
        "endTime": end_time
    }

    if count:
        params["count"] = count

    if bounding_box:
        params["boundingBox"] = bounding_box

    if gid:
        params["gId"] = gid

    params = {
        key: value
        for key, value in params.items()
        if value not in ("", None)
    }

    print("\n" + "=" * 70)
    print("MOSDAC DATASET SEARCH")
    print("=" * 70)

    print(f"Dataset ID : {dataset_id}")
    print(f"Start      : {start_time}")
    print(f"End        : {end_time}")
    print(f"Count      : {count}")
    print(f"Bounding   : {bounding_box}")
    print(f"Granule ID : {gid}")

    try:

        headers = {
            "Authorization":
                f"Bearer {bearer_token}"
        }

        response = requests.get(
            search_url,
            headers=headers,
            params=params,
            timeout=60
        )

        if response.status_code == 401:

            print(
                "[WARNING] Token expired."
            )

            if refresh_authentication():

                headers = {
                    "Authorization":
                        f"Bearer {bearer_token}"
                }

                response = requests.get(
                    search_url,
                    headers=headers,
                    params=params,
                    timeout=60
                )

            else:

                print(
                    "[ERROR] Could not refresh token."
                )

                return None

        response.raise_for_status()

        result = response.json()

        total_results = result.get(
            "totalResults",
            result.get("total_records", 0)
        )

        total_size_mb = result.get(
            "totalSizeMB",
            result.get("total_size_mb", 0)
        )

        items_per_page = result.get(
            "itemsPerPage",
            result.get("items_per_page", 0)
        )

        print(
            f"\nFiles found       : {total_results}"
        )

        print(
            f"Total size        : {total_size_mb} MB"
        )

        print(
            f"Items in response : {items_per_page}"
        )

        print("=" * 70)

        return result

    except Exception as e:

        print(
            f"[ERROR] Dataset search failed: {e}"
        )

        logger.exception(
            "Dataset search failed"
        )

        return None


# ============================================================
# INTEGER HELPER
# ============================================================

def safe_int(value, default=0):

    try:
        return int(value)

    except Exception:
        return default


# ============================================================
# FORMAT BYTES
# ============================================================

def format_bytes(num_bytes):

    if num_bytes < 1024:
        return f"{num_bytes} B"

    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.2f} KB"

    if num_bytes < 1024 ** 3:
        return f"{num_bytes / 1024 ** 2:.2f} MB"

    return f"{num_bytes / 1024 ** 3:.2f} GB"


# ============================================================
# RESUMABLE DOWNLOAD
# ============================================================

def download_data(
    identifier,
    record_id,
    download_directory,
    filename,
    max_retries=20
):

    global bearer_token

    download_directory = Path(
        download_directory
    )

    download_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path = (
        download_directory
        / filename
    )

    tmp_file_path = Path(
        str(file_path) + ".part"
    )

    download_settings = config.get(
        "download_settings",
        {}
    )

    generate_logs = download_settings.get(
        "generate_error_logs",
        True
    )

    # ========================================================
    # RETRY / RESUME LOOP
    # ========================================================

    for attempt in range(
        1,
        max_retries + 1
    ):

        # ----------------------------------------------------
        # CHECK EXISTING PARTIAL FILE
        # ----------------------------------------------------

        if tmp_file_path.exists():

            existing_size = (
                tmp_file_path.stat().st_size
            )

        else:

            existing_size = 0

        print("\n" + "-" * 70)

        print(
            f"[DOWNLOAD] {filename}"
        )

        print(
            f"[INFO] Attempt {attempt}/{max_retries}"
        )

        if existing_size > 0:

            print(
                f"[INFO] Existing partial file: "
                f"{format_bytes(existing_size)}"
            )

            print(
                "[INFO] Attempting RESUME..."
            )

        else:

            print(
                "[INFO] Starting download from 0 bytes."
            )

        # ----------------------------------------------------
        # REQUEST HEADERS
        # ----------------------------------------------------

        headers = {
            "Authorization":
                f"Bearer {bearer_token}"
        }

        # ----------------------------------------------------
        # RANGE REQUEST
        # ----------------------------------------------------

        if existing_size > 0:

            headers["Range"] = (
                f"bytes={existing_size}-"
            )

        params = {
            "id": record_id
        }

        try:

            response = requests.get(
                download_url,
                headers=headers,
                params=params,
                stream=True,
                timeout=(30, 60)
            )

            # =================================================
            # TOKEN EXPIRED
            # =================================================

            if response.status_code == 401:

                print(
                    "[WARNING] Authentication token expired."
                )

                if refresh_authentication():

                    print(
                        "[INFO] Retrying with new token..."
                    )

                    continue

                print(
                    "[ERROR] Token refresh failed."
                )

                return None

            # =================================================
            # RANGE NOT SUPPORTED
            # =================================================

            if existing_size > 0 and response.status_code == 200:

                print(
                    "[WARNING] MOSDAC server did not honor "
                    "the Range request."
                )

                print(
                    "[INFO] Server returned HTTP 200 instead "
                    "of HTTP 206."
                )

                print(
                    "[INFO] Restarting this file from zero "
                    "to avoid corrupting it."
                )

                response.close()

                try:

                    tmp_file_path.unlink()

                except FileNotFoundError:

                    pass

                time.sleep(3)

                continue

            # =================================================
            # RANGE SUCCESS
            # =================================================

            if existing_size > 0:

                if response.status_code == 206:

                    print(
                        "[SUCCESS] Server accepted "
                        "resume request."
                    )

                else:

                    print(
                        f"[WARNING] Unexpected HTTP status "
                        f"{response.status_code} for resume."
                    )

            # =================================================
            # NORMAL DOWNLOAD
            # =================================================

            response.raise_for_status()

            # ------------------------------------------------
            # CONTENT LENGTH
            # ------------------------------------------------

            response_length = safe_int(
                response.headers.get(
                    "Content-Length",
                    0
                ),
                0
            )

            # ------------------------------------------------
            # DETERMINE TOTAL FILE SIZE
            # ------------------------------------------------

            content_range = response.headers.get(
                "Content-Range",
                ""
            )

            total_size = 0

            # Example:
            # bytes 140452571-425255359/425255360

            if content_range:

                match = re.search(
                    r"/(\d+)$",
                    content_range
                )

                if match:

                    total_size = int(
                        match.group(1)
                    )

            if total_size == 0:

                if existing_size > 0:

                    total_size = (
                        existing_size
                        + response_length
                    )

                else:

                    total_size = response_length

            # ------------------------------------------------
            # SHOW EXPECTED SIZE
            # ------------------------------------------------

            if total_size > 0:

                print(
                    f"[INFO] Expected total size: "
                    f"{format_bytes(total_size)} "
                    f"({total_size:,} bytes)"
                )

            # ------------------------------------------------
            # OPEN FILE IN CORRECT MODE
            # ------------------------------------------------

            if existing_size > 0 and response.status_code == 206:

                file_mode = "ab"

                downloaded_size = existing_size

            else:

                file_mode = "wb"

                downloaded_size = 0

                existing_size = 0

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            start_time = time.time()

            if tqdm:

                progress = tqdm(
                    total=total_size
                    if total_size > 0
                    else None,
                    initial=downloaded_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=filename,
                    leave=True
                )

            else:

                progress = None

            try:

                with open(
                    tmp_file_path,
                    file_mode
                ) as file:

                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):

                        if not chunk:
                            continue

                        file.write(chunk)

                        downloaded_size += len(
                            chunk
                        )

                        if progress:

                            progress.update(
                                len(chunk)
                            )

            finally:

                if progress:
                    progress.close()

                response.close()

            # ------------------------------------------------
            # ACTUAL FILE SIZE
            # ------------------------------------------------

            actual_size = (
                tmp_file_path.stat().st_size
                if tmp_file_path.exists()
                else 0
            )

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"\n[INFO] Current file size: "
                f"{format_bytes(actual_size)}"
            )

            if elapsed > 0:

                speed = (
                    actual_size
                    / elapsed
                    / 1024
                    / 1024
                )

                print(
                    f"[INFO] Approx. transfer speed: "
                    f"{speed:.2f} MB/s"
                )

            # =================================================
            # VERIFY COMPLETE DOWNLOAD
            # =================================================

            if total_size > 0:

                if actual_size == total_size:

                    print(
                        "\n[SUCCESS] COMPLETE FILE RECEIVED."
                    )

                else:

                    percentage = (
                        actual_size
                        / total_size
                        * 100
                    )

                    print(
                        "\n[WARNING] Download interrupted."
                    )

                    print(
                        f"[INFO] Received: "
                        f"{format_bytes(actual_size)}"
                    )

                    print(
                        f"[INFO] Expected: "
                        f"{format_bytes(total_size)}"
                    )

                    print(
                        f"[INFO] Progress: "
                        f"{percentage:.2f}%"
                    )

                    if generate_logs:

                        logger.warning(
                            f"Partial download: "
                            f"{filename} | "
                            f"{actual_size}/{total_size}"
                        )

                    if attempt < max_retries:

                        print(
                            f"[INFO] Keeping .part file "
                            f"for resume."
                        )

                        print(
                            f"[INFO] Retrying in 5 seconds..."
                        )

                        time.sleep(5)

                        continue

                    print(
                        "[ERROR] Maximum retry attempts reached."
                    )

                    return None

            # =================================================
            # FINALIZE
            # =================================================

            if not tmp_file_path.exists():

                print(
                    "[ERROR] Temporary file disappeared."
                )

                return None

            if file_path.exists():

                file_path.unlink()

            os.rename(
                tmp_file_path,
                file_path
            )

            print(
                "\n[SUCCESS] Download verified and finalized."
            )

            print(
                f"[SUCCESS] File: {file_path}"
            )

            print(
                f"[SUCCESS] Size: "
                f"{format_bytes(actual_size)}"
            )

            return str(file_path)

        # ====================================================
        # CONNECTION / TIMEOUT
        # ====================================================

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        ) as e:

            print(
                f"\n[ERROR] Connection interrupted:"
            )

            print(
                f"        {e}"
            )

            logger.warning(
                f"Connection interrupted for "
                f"{filename}: {e}"
            )

            # -----------------------------------------------
            # IMPORTANT:
            # DO NOT DELETE .part
            # -----------------------------------------------

            if tmp_file_path.exists():

                partial_size = (
                    tmp_file_path.stat().st_size
                )

                print(
                    f"[INFO] Partial file preserved: "
                    f"{format_bytes(partial_size)}"
                )

                print(
                    "[INFO] Next attempt will resume "
                    "from this byte."
                )

            if attempt < max_retries:

                time.sleep(5)

                continue

            print(
                "[ERROR] Maximum retry attempts reached."
            )

            return None

        # ====================================================
        # HTTP ERROR
        # ====================================================

        except requests.exceptions.HTTPError as e:

            print(
                f"\n[ERROR] HTTP error: {e}"
            )

            logger.exception(
                f"HTTP error downloading {filename}"
            )

            # -----------------------------------------------
            # If this was a bad Range request, keep the
            # partial file only if another attempt can
            # retry it.
            # -----------------------------------------------

            if response is not None:

                try:
                    response.close()
                except Exception:
                    pass

            if attempt < max_retries:

                print(
                    "[INFO] Retrying..."
                )

                time.sleep(5)

                continue

            return None

        # ====================================================
        # OTHER REQUEST ERROR
        # ====================================================

        except requests.exceptions.RequestException as e:

            print(
                f"\n[ERROR] Request error:"
            )

            print(
                f"        {e}"
            )

            logger.exception(
                f"Request error downloading {filename}"
            )

            if tmp_file_path.exists():

                partial_size = (
                    tmp_file_path.stat().st_size
                )

                print(
                    f"[INFO] Partial file preserved: "
                    f"{format_bytes(partial_size)}"
                )

            if attempt < max_retries:

                print(
                    "[INFO] Retrying with resume..."
                )

                time.sleep(5)

                continue

            return None

        # ====================================================
        # GENERAL ERROR
        # ====================================================

        except Exception as e:

            print(
                f"\n[ERROR] Unexpected error: {e}"
            )

            logger.exception(
                f"Unexpected error downloading {filename}"
            )

            if tmp_file_path.exists():

                partial_size = (
                    tmp_file_path.stat().st_size
                )

                print(
                    f"[INFO] Partial file preserved: "
                    f"{format_bytes(partial_size)}"
                )

            if attempt < max_retries:

                print(
                    "[INFO] Retrying..."
                )

                time.sleep(5)

                continue

            return None

    return None


# ============================================================
# FETCH AND DOWNLOAD DATA
# ============================================================

def fetch_and_download_data():

    search_parameters = config.get(
        "search_parameters",
        {}
    )

    download_settings = config.get(
        "download_settings",
        {}
    )

    dataset_id = search_parameters.get(
        "datasetId",
        ""
    )

    start_time = search_parameters.get(
        "startTime",
        ""
    )

    end_time = search_parameters.get(
        "endTime",
        ""
    )

    count = search_parameters.get(
        "count",
        ""
    )

    bounding_box = search_parameters.get(
        "boundingBox",
        ""
    )

    gid = search_parameters.get(
        "gId",
        ""
    )

    download_path = download_settings.get(
        "download_path",
        "./downloads/"
    )

    organize_by_date = download_settings.get(
        "organize_by_date",
        False
    )

    params = {
        "datasetId": dataset_id,
        "startTime": start_time,
        "endTime": end_time
    }

    if count:
        params["count"] = count

    if bounding_box:
        params["boundingBox"] = bounding_box

    if gid:
        params["gId"] = gid

    params = {
        key: value
        for key, value in params.items()
        if value not in ("", None)
    }

    headers = {
        "Authorization":
            f"Bearer {bearer_token}"
    }

    print("\n" + "=" * 70)
    print("FETCHING DATA RECORDS")
    print("=" * 70)

    try:

        response = requests.get(
            search_url,
            headers=headers,
            params=params,
            timeout=60
        )

        if response.status_code == 401:

            if refresh_authentication():

                headers = {
                    "Authorization":
                        f"Bearer {bearer_token}"
                }

                response = requests.get(
                    search_url,
                    headers=headers,
                    params=params,
                    timeout=60
                )

            else:

                print(
                    "[ERROR] Authentication refresh failed."
                )

                return

        response.raise_for_status()

        result = response.json()

    except Exception as e:

        print(
            f"[ERROR] Could not fetch records: {e}"
        )

        logger.exception(
            "Could not fetch records"
        )

        return

    total_results = safe_int(
        result.get(
            "totalResults",
            result.get("total_records", 0)
        ),
        0
    )

    requested_count = safe_int(
        count,
        0
    )

    if requested_count > 0:

        target_count = min(
            requested_count,
            total_results
            if total_results > 0
            else requested_count
        )

    else:

        target_count = total_results

    print(
        f"[INFO] Total available records: "
        f"{total_results}"
    )

    print(
        f"[INFO] Records requested: "
        f"{target_count}"
    )

    # ========================================================
    # PAGINATION
    # ========================================================

    start_index = 1
    processed_records = 0
    downloaded_files = 0

    batch_size = 100

    while processed_records < target_count:

        page_params = dict(params)

        page_params["startIndex"] = start_index

        page_params["itemsPerPage"] = min(
            batch_size,
            target_count - processed_records
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"[INFO] Requesting records "
            f"{start_index} - "
            f"{start_index + page_params['itemsPerPage'] - 1}"
        )

        try:

            response = requests.get(
                search_url,
                headers=headers,
                params=page_params,
                timeout=60
            )

            if response.status_code == 401:

                if refresh_authentication():

                    headers = {
                        "Authorization":
                            f"Bearer {bearer_token}"
                    }

                    response = requests.get(
                        search_url,
                        headers=headers,
                        params=page_params,
                        timeout=60
                    )

                else:

                    print(
                        "[ERROR] Authentication refresh failed."
                    )

                    return

            response.raise_for_status()

            page_result = response.json()

        except Exception as e:

            print(
                f"[ERROR] Pagination request failed: {e}"
            )

            logger.exception(
                "Pagination request failed"
            )

            return

        page_entries = page_result.get(
            "entries",
            page_result.get(
                "data",
                []
            )
        )

        if not page_entries:

            print(
                "[INFO] No more records returned."
            )

            break

        # ====================================================
        # DOWNLOAD EACH RECORD
        # ====================================================

        for item in page_entries:

            if processed_records >= target_count:
                break

            processed_records += 1

            identifier = (
                item.get("identifier")
                or item.get("fileName")
                or item.get("filename")
                or item.get("name")
            )

            record_id = (
                item.get("recordId")
                or item.get("record_id")
                or item.get("id")
                or item.get("gId")
                or item.get("gid")
            )

            updated = (
                item.get("updated")
                or item.get("date")
                or ""
            )

            if not record_id:

                print(
                    "[WARNING] Record has no record ID. "
                    "Skipping."
                )

                logger.warning(
                    f"Record missing ID: {item}"
                )

                continue

            if not identifier:

                identifier = str(
                    record_id
                )

            filename = os.path.basename(
                str(identifier)
            )

            if not filename.lower().endswith(
                (
                    ".h5",
                    ".nc",
                    ".zip",
                    ".tar",
                    ".gz"
                )
            ):

                filename += ".h5"

            # =================================================
            # ORGANIZE BY DATE
            # =================================================

            target_directory = Path(
                download_path
            )

            if organize_by_date and updated:

                date_match = re.search(
                    r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})",
                    str(updated)
                )

                if date_match:

                    year = date_match.group(1)

                    month_day = (
                        date_match.group(2)
                        + date_match.group(3)
                    )

                    target_directory = (
                        target_directory
                        / year
                        / month_day
                    )

            target_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            # =================================================
            # DOWNLOAD
            # =================================================

            result_file = download_data(
                identifier=identifier,
                record_id=record_id,
                download_directory=target_directory,
                filename=filename
            )

            if result_file:

                downloaded_files += 1

        start_index += len(
            page_entries
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(
        f"Records processed : "
        f"{processed_records}"
    )

    print(
        f"Files downloaded  : "
        f"{downloaded_files}"
    )

    print("=" * 70)


# ============================================================
# LOGOUT
# ============================================================

def logout():

    global bearer_token

    if not bearer_token:
        return

    try:

        requests.post(
            logout_url,
            headers={
                "Authorization":
                    f"Bearer {bearer_token}"
            },
            timeout=15
        )

        print(
            "[INFO] Logged out from MOSDAC."
        )

    except Exception as e:

        logger.warning(
            f"Logout failed: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("MOSDAC DATA DOWNLOAD TOOL")
    print("=" * 70)

    load_config()

    setup_logging()

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    if not authenticate():

        print(
            "[ERROR] Authentication failed."
        )

        return

    try:

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        search_results()

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        print(
            "\n[INFO] Starting resumable download..."
        )

        fetch_and_download_data()

    finally:

        logout()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()