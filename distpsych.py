import argparse
import logging
import os
import pickle
import threading
import tkinter.filedialog
from dataclasses import dataclass
from typing import Dict, List

import chardet
import customtkinter
import pandas as pd
import requests


@dataclass
class ProviderData:
    district: str | None = None
    can_serve: List[str] | None = None
    cannot_serve: List[str] | None = None


class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.insert("end", msg + "\n")
        self.text_widget.see("end")


def parse_args():
    """Parse command-line arguments for processing client and provider data."""
    parser = argparse.ArgumentParser(description="Process client and provider data")
    parser.add_argument(
        "--dem", "-d", help="Path to demographics CSV file", metavar="FILE"
    )
    parser.add_argument(
        "--provider", "-p", help="Path to provider CSV file", metavar="FILE"
    )
    parser.add_argument(
        "--insurance", "-i", help="Path to insurance CSV file", metavar="FILE"
    )
    return parser.parse_args()


def setup_logger(gui_mode=False, text_widget=None):
    """
    Set up the logger for either GUI or console mode.

    Args:
        gui_mode (bool): True if running in GUI mode, False for console mode.
        text_widget: The text widget to log to in GUI mode (only used if gui_mode is True).

    Returns:
        None
    """
    # Get the root logger and set its level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a formatter for log messages
    log_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
    )

    if gui_mode and text_widget:
        # Set up logging for GUI mode
        handler = TextHandler(text_widget)
    else:
        # Set up logging for console mode
        handler = logging.StreamHandler()

    # Apply the formatter and add the handler to the logger
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    logging.info("Logger setup complete.")


global_district_count = 0


def update_district_count_callback():
    global global_district_count
    if App.instance:
        App.instance.after(
            0,
            App.instance.update_district_count,
            global_district_count,
            global_active_client_count,
        )


def get_district(street: str, city: str, state: str, zip: str) -> str:
    """Get the school district from a street address using the U.S. Census Bureau's API."""
    url = "https://geocoding.geo.census.gov/geocoder/geographies/address"
    params = {
        "street": street.strip(),
        "city": city.strip(),
        "state": state.strip(),
        "zip": zip.strip(),
        "benchmark": "Public_AR_Current",
        "format": "json",
        "vintage": "Current_Current",
        "layers": 14,
    }

    if any(x == "nan" for x in [street, city, state]):
        logging.error("Street, city, or state cannot be empty.")
        return "Not found"

    try:
        logging.info(
            f"Searching for {params['street']} {params['city']}, {params['state']} {params['zip']}"
        )
        return search_district(url, params)
    except requests.RequestException as e:
        logging.error(f"Error fetching school district data: {e}")
        return "Not found"


def search_district(url: str, params: dict) -> str:
    """Search for district with given parameters, retry without ZIP if failed."""
    district = get_district_from_response(url, params)
    if district != "Not found":
        return district

    logging.warning("Search failed, attempting again without a ZIP code...")
    params.pop("zip")
    return get_district_from_response(url, params)


def get_district_from_response(url: str, params: dict) -> str:
    """Make API request and extract district from response."""
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if data["result"]["addressMatches"]:
        district = data["result"]["addressMatches"][0]["geographies"][
            "Unified School Districts"
        ][0]["NAME"]
        logging.info(f"District found: {district}")
        return district
    else:
        logging.error("No district found.")
        return "Not found"


def pick_file():
    file = tkinter.filedialog.askopenfilename()
    if file:
        logging.info(f"File selected: {file}")
    else:
        logging.warning("No file selected.")
    return file


def create_provider_location_dict(df: pd.DataFrame) -> Dict[str, ProviderData]:
    """
    Create a dictionary of provider locations from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing provider information.

    Returns:
        Dict[str, ProviderData]: Dictionary of provider names mapped to their location data.
    """
    logging.info("Creating provider location dictionary...")
    provider_names = df.columns[2:]
    location_index = df[df["Unnamed: 0"] == "Excluded Areas"].index.item()

    provider_locations = {}
    for i, provider_name in enumerate(provider_names):
        location = df.iloc[
            location_index, i + 2
        ]  # i + 2 accounts for the extra beggining columns
        provider_data = ProviderData()

        if pd.notna(location):
            provider_data.district = normalize_district_name(location)

        for row_index in range(df.shape[0]):
            cell_value = df.iloc[row_index, i + 2]
            if isinstance(cell_value, str):
                if "CANNOT" in cell_value:
                    provider_data.cannot_serve.extend(
                        extract_zip_codes(cell_value, "CANNOT")
                    )
                elif "CAN" in cell_value:
                    provider_data.can_serve.extend(extract_zip_codes(cell_value, "CAN"))

        provider_locations[provider_name] = provider_data

    logging.info(
        f"Created provider location dictionary with {len(provider_locations)} providers."
    )
    return {k: v for k, v in provider_locations.items()}


def normalize_district_name(location: str) -> str:
    """
    Normalize district names by replacing abbreviations with full names.

    Args:
        location (str): The original district name.

    Returns:
        str: The normalized district name.
    """
    replacements = {
        "DD4": "Dorchester District 4",
        "Berkeley": "Berkeley County School District",
        "Georgetown": "Georgetown County School District",
        "Horry": "Horry County School District",
        "Charleston": "Charleston County School District",
    }
    for old, new in replacements.items():
        location = location.replace(old, new)
    logging.debug(f"Normalized district name: {location}")
    return location


def extract_zip_codes(cell_value: str, prefix: str) -> List[str]:
    """
    Extract zip codes from a cell value based on a given prefix.

    Args:
        cell_value (str): The cell value containing zip codes.
        prefix (str): The prefix to split the cell value ("CAN" or "CANNOT").

    Returns:
        List[str]: A list of extracted zip codes.
    """
    zip_codes = cell_value.split(prefix)[-1].strip().split()
    logging.debug(f"Extracted {len(zip_codes)} zip codes for {prefix} prefix.")
    return zip_codes


def get_provider_insurance(df: pd.DataFrame) -> dict:
    """
    Extracts provider insurance information from a DataFrame.

    Args:
      df: A pandas DataFrame containing provider insurance data.

    Returns:
      A dictionary where keys are provider names and values are lists of accepted insurance.
    """
    logging.info("Extracting provider insurance information...")
    provider_names = df.columns[2:]
    provider_insurance = {}
    insurance_replacements = {
        "BABYNET": "BabyNet (Combined DA and Eval)",
        "SCM": "Medicaid South Carolina",
        "ATC": "Absolute Total Care - Medical",
        "SH": "Select Health of South Carolina",
        "Molina": "Molina Healthcare of South Carolina",
        "Humana": "Humana Behavioral Health (formerly LifeSynch)",
        "HB": "Healthy Blue Medicaid South Carolina",
        "AETNA": "Aetna Health, Inc.",
        "TriCare": "TREST",
        "UNITED/OPTUM": "United Healthcare/OptumHealth / OptumHealth Behavioral Solutions (formerly United Behavioral Health [UBH] and PacifiCare Behavioral Health )",
    }

    for i, row in df.iterrows():
        if row["Unnamed: 0"] == "Insurance":
            continue
        if row["Unnamed: 0"] == "Excluded Areas":
            break

        insurance_name = row["Unnamed: 0"]
        for replacement, full_name in insurance_replacements.items():
            insurance_name = insurance_name.replace(replacement, full_name)

        for j, provider_name in enumerate(provider_names):
            cell_value = row.iloc[j + 2]
            if pd.notna(cell_value) and (cell_value == "X" or cell_value.isdigit()):
                if provider_name not in provider_insurance:
                    provider_insurance[provider_name] = []
                provider_insurance[provider_name].append(insurance_name)

    logging.info(
        f"Extracted insurance information for {len(provider_insurance)} providers."
    )
    return provider_insurance


global_active_client_count = 0


def extract_client_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts relevant client data from a DataFrame.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing client information.

    Returns:
        pd.DataFrame: A new DataFrame containing only active clients and selected columns.
    """
    logging.info("Extracting client data...")
    # Filter out inactive clients
    active_clients = df[df["STATUS"] != "Inactive"]
    # Define columns to keep
    selected_columns = [
        "CLIENT_ID",
        "LASTNAME",
        "FIRSTNAME",
        "PREFERRED_NAME",
        "USER_ADDRESS_ADDRESS1",
        "USER_ADDRESS_CITY",
        "USER_ADDRESS_STATE",
        "USER_ADDRESS_ZIP",
    ]

    # Extract only the selected columns
    extracted_df = active_clients[selected_columns]
    global global_active_client_count
    global_active_client_count = len(extracted_df)

    logging.info(f"Extracted data for {len(extracted_df)} active clients.")
    return extracted_df


def add_districts_to_clients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'USER_DISTRICT' column to a DataFrame containing client data,
    using the get_district function to determine the school district based on address.

    Args:
        df: A pandas DataFrame containing client information, including address columns.

    Returns:
        A pandas DataFrame with a new 'USER_DISTRICT' column containing the school district for each client.
    """
    logging.info("Adding districts to clients...")
    df["USER_DISTRICT"] = None
    cache_file = "district_cache.pickle"
    district_cache = load_district_cache(cache_file)

    for index, row in df.iterrows():
        address_key = create_address_key(row)
        district = get_district_from_cache_or_api(address_key, district_cache)
        df.loc[index, "USER_DISTRICT"] = district

    save_district_cache(cache_file, district_cache)
    logging.info(f"Added districts to {len(df)} clients.")
    return df


def load_district_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
                logging.info(f"Loaded district cache with {len(cache)} entries.")
                return cache
        except (pickle.PickleError, EOFError):
            logging.warning("Error loading district cache, creating a new one")
    return {}


def create_address_key(row):
    return f"{row['USER_ADDRESS_ADDRESS1']}|{row['USER_ADDRESS_CITY']}|{row['USER_ADDRESS_STATE']}|{row['USER_ADDRESS_ZIP']}"


def get_district_from_cache_or_api(address_key, district_cache):
    global global_district_count
    global_district_count += 1
    update_district_count_callback()
    if address_key in district_cache:
        district = district_cache[address_key]
        logging.info(f"District found in cache: {district}")
    else:
        street, city, state, zip_code = address_key.split("|")
        district = get_district(street=street, city=city, state=state, zip=zip_code)
        if district != "Not found":
            district_cache[address_key] = district
    return district


def save_district_cache(cache_file, district_cache):
    with open(cache_file, "wb") as f:
        pickle.dump(district_cache, f)
    logging.info(f"Saved district cache with {len(district_cache)} entries.")


global_insurance_count = 0


def update_insurance_count_callback():
    global global_insurance_count
    if App.instance:
        App.instance.after(
            0,
            App.instance.update_insurance_count,
            global_insurance_count,
            global_active_client_count,
        )


def match_client_ids_and_add_insurance(
    client_df: pd.DataFrame, insurance_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Matches client IDs between two DataFrames and adds the insurance company name to the client DataFrame.

    Args:
        client_df (pd.DataFrame): DataFrame containing client information, including CLIENT_ID.
        insurance_df (pd.DataFrame): DataFrame containing insurance information, including CLIENT_ID and INSURANCE_COMPANYNAME.

    Returns:
        pd.DataFrame: The client DataFrame with a new column 'INSURANCE_COMPANYNAME'.
    """
    logging.info("Matching client IDs and adding insurance information...")
    cache_file = "insurance_cache.pickle"
    global global_insurance_count

    # Try to load the cache
    try:
        with open(cache_file, "rb") as f:
            insurance_mapping = pickle.load(f)
        logging.info(f"Loaded insurance cache with {len(insurance_mapping)} entries.")
    except (FileNotFoundError, pickle.UnpicklingError):
        # If cache doesn't exist or is corrupted, create a new mapping
        insurance_mapping = dict(
            zip(insurance_df["CLIENT_ID"], insurance_df["INSURANCE_COMPANYNAME"])
        )
        logging.info("Created new insurance cache.")

    # Add 'INSURANCE_COMPANYNAME' column to client DataFrame using the mapping
    for index, row in client_df.iterrows():
        global_insurance_count += 1
        update_insurance_count_callback()
        client_id = row["CLIENT_ID"]
        insurance = insurance_mapping.get(client_id, "None found")
        client_df.at[index, "INSURANCE_COMPANYNAME"] = insurance
        logging.info(f"Client {client_id} - {insurance}")

    # Save the updated mapping to cache
    with open(cache_file, "wb") as f:
        pickle.dump(insurance_mapping, f)
    logging.info(f"Saved insurance cache with {len(insurance_mapping)} entries.")

    logging.info(f"Added insurance information to {len(client_df)} clients.")
    return client_df


global_provider_count = 0


def update_provider_count_callback():
    global global_provider_count
    if App.instance:
        App.instance.after(
            0,
            App.instance.update_provider_count,
            global_provider_count,
            global_active_client_count,
        )


def assign_providers_to_clients(
    client_df: pd.DataFrame,
    provider_locations: Dict[str, ProviderData],
    provider_insurance: dict,
) -> pd.DataFrame:
    """
    Assigns providers to clients based on their location and providers' location.
    Places an 'X' in the client's DataFrame for each provider that can serve them.

    Args:
        client_df (pd.DataFrame): DataFrame containing client information, including USER_DISTRICT.
        provider_locations (dict): Dictionary mapping provider names to their districts.
        provider_insurance (dict): Dictionary mapping provider names to their accepted insurance.

    Returns:
        pd.DataFrame: The client DataFrame with 'X's in the provider columns for clients
        that can be served by that provider.
    """
    logging.info("Assigning providers to clients...")
    # Add a column for each provider with NaN values
    for provider_name in provider_locations.keys():
        client_df[provider_name] = None

    global global_provider_count

    for index, row in client_df.iterrows():
        client_district = row["USER_DISTRICT"]
        client_zip = row["USER_ADDRESS_ZIP"]
        client_insurance = row["INSURANCE_COMPANYNAME"]
        global_provider_count += 1
        update_provider_count_callback()

        for provider_name, provider_data in provider_locations.items():
            if provider_name not in provider_insurance:
                continue

            if client_insurance not in provider_insurance[provider_name]:
                continue

            match_reason = None

            # Check if provider serves different district
            if provider_data.district != client_district:
                match_reason = "Different district"

            # Check if client's ZIP is in provider's can_serve list
            elif client_zip in provider_data.can_serve:
                match_reason = "ZIP in can_serve list"

            # Check if client's ZIP is not in provider's cannot_serve list
            elif (
                provider_data.cannot_serve
                and client_zip not in provider_data.cannot_serve
            ):
                match_reason = "ZIP not in cannot_serve list"

            if match_reason:
                logging.info(
                    f"Provider match: {provider_name} - {client_district}/{client_zip} - {client_insurance} - Reason: {match_reason}"
                )
                client_df.loc[index, provider_name] = "X"

    logging.info(f"Assigned providers to {len(client_df)} clients.")
    return client_df


def load_csv(file_path):
    """
    Loads a CSV file with error handling and automatic encoding detection.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame or None: DataFrame containing CSV data if successful, None otherwise.
    """
    logging.info(f"Loading {file_path}...")
    try:
        # Read file in binary mode for encoding detection
        with open(file_path, "rb") as file:
            raw_data = file.read()

        # Detect file encoding
        detected_encoding = chardet.detect(raw_data)["encoding"]

        # Read CSV file with detected encoding
        df = pd.read_csv(file_path, encoding=detected_encoding)
        logging.info(f"Successfully loaded {file_path} with {len(df)} rows.")
        return df

    except UnicodeDecodeError as e:
        logging.error(f"Error decoding file: {e}")
    except pd.errors.EmptyDataError as e:
        logging.error(f"Error reading file, may be empty or corrupt: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return None


def process_data(dem_sheet, provider_sheet, insurance_sheet):
    logging.info("Starting data processing...")
    global global_district_count
    global_district_count = 0
    global global_insurance_count
    global_insurance_count = 0
    global global_provider_count
    global_provider_count = 0
    update_district_count_callback()
    update_insurance_count_callback()
    update_provider_count_callback()
    trimmed_clients = extract_client_data(dem_sheet)
    clients_with_districts = add_districts_to_clients(pd.DataFrame(trimmed_clients))
    clients_with_districts_and_insurance = match_client_ids_and_add_insurance(
        clients_with_districts, insurance_sheet
    )
    provider_locations = create_provider_location_dict(provider_sheet)
    provider_insurance = get_provider_insurance(provider_sheet)
    result_df = assign_providers_to_clients(
        clients_with_districts_and_insurance,
        provider_locations,
        provider_insurance,
    )
    save_results_to_csv(result_df)
    logging.info("Data processing completed successfully!")


def save_results_to_csv(result_df):
    logging.info("Saving results to RESULTS.csv...")
    # Rename all "Unnamed" columns to "" (blank)
    unnamed_columns = result_df.columns[result_df.columns.str.contains("Unnamed")]
    result_df = result_df.rename(columns={col: "" for col in unnamed_columns})
    # Remove all digit suffixes from duplicate column names
    result_df.columns = result_df.columns.str.replace(r"\.\d+$", "", regex=True)
    result_df.to_csv("RESULTS.csv", index=False)
    logging.info(f"Results saved to RESULTS.csv with {len(result_df)} rows.")


def truncate_text(text, max_length=20):
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        App.instance = self
        self.dem_sheet = None
        self.provider_sheet = None
        self.insurance_sheet = None

        self.title("DistPsych")
        self.geometry("600x600")
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure(3, weight=1)  # Make the log frame row expandable

        self.dem_sheet_button = customtkinter.CTkButton(
            self, text="Select Demographics Sheet", command=self.get_dem_sheet
        )
        self.dem_sheet_button.grid(row=0, column=0, padx=5, pady=10)

        self.provider_sheet_button = customtkinter.CTkButton(
            self, text="Select Provider Sheet", command=self.get_provider_sheet
        )
        self.provider_sheet_button.grid(row=0, column=1, padx=5, pady=10)

        self.insurance_sheet_button = customtkinter.CTkButton(
            self, text="Select Insurance Sheet", command=self.get_insurance_sheet
        )
        self.insurance_sheet_button.grid(row=0, column=2, padx=5, pady=10)

        self.process_button = customtkinter.CTkButton(
            self, text="Process!", command=self.process_sheet, state="disabled"
        )
        self.process_button.grid(
            row=1, column=0, columnspan=3, padx=20, pady=10, sticky="ew"
        )

        self.district_count_label = customtkinter.CTkLabel(
            self, text="Districts searched: 0"
        )
        self.district_count_label.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.insurance_count_label = customtkinter.CTkLabel(
            self, text="Insurance attempted: 0"
        )
        self.insurance_count_label.grid(row=2, column=1, padx=20, pady=10, sticky="ew")

        self.provider_count_label = customtkinter.CTkLabel(
            self, text="Clients attempted: 0"
        )
        self.provider_count_label.grid(row=2, column=2, padx=20, pady=10, sticky="ew")

        self.log_frame = customtkinter.CTkFrame(self)
        self.log_frame.grid(
            row=3, column=0, columnspan=3, padx=20, pady=10, sticky="nsew"
        )  # Changed to "nsew" to expand in all directions

        self.log_text = customtkinter.CTkTextbox(
            self.log_frame,
            wrap="word",
            state="normal",
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        # Set up custom logger
        setup_logger(gui_mode=True, text_widget=self.log_text)

    def get_dem_sheet(self):
        file = pick_file()
        if file:
            self.dem_sheet = load_csv(file)
            self.dem_sheet_button.configure(text=truncate_text(file.split("/")[-1]))
            self.check_process_button_state()

    def get_provider_sheet(self):
        file = pick_file()
        if file:
            self.provider_sheet = load_csv(file)
            self.provider_sheet_button.configure(
                text=truncate_text(file.split("/")[-1])
            )
            self.check_process_button_state()

    def get_insurance_sheet(self):
        file = pick_file()
        if file:
            self.insurance_sheet = load_csv(file)
            self.insurance_sheet_button.configure(
                text=truncate_text(file.split("/")[-1])
            )
            self.check_process_button_state()

    def process_sheet(self):
        # Create a new thread for the processing
        processing_thread = threading.Thread(target=self._process_data)
        processing_thread.start()

    def _process_data(self):
        if (
            self.dem_sheet is not None
            and self.provider_sheet is not None
            and self.insurance_sheet is not None
        ):
            process_data(self.dem_sheet, self.provider_sheet, self.insurance_sheet)

    def check_process_button_state(self):
        if (
            self.dem_sheet is not None
            and self.provider_sheet is not None
            and self.insurance_sheet is not None
        ):
            self.process_button.configure(state="normal")
            logging.info("All sheets loaded. Ready to process.")
        else:
            self.process_button.configure(state="disabled")

    def update_district_count(self, count, client_count):
        self.district_count_label.configure(
            text=f"Districts searched: {count}/{client_count}"
        )

    def update_insurance_count(self, count, client_count):
        self.insurance_count_label.configure(
            text=f"Insurance attempted: {count}/{client_count}"
        )

    def update_provider_count(self, count, client_count):
        self.provider_count_label.configure(
            text=f"Clients attempted: {count}/{client_count}"
        )


def main():
    args = parse_args()

    if args.dem and args.provider and args.insurance:
        # Command-line mode
        setup_logger(gui_mode=False)
        logging.info("Starting in command-line mode.")
        dem_sheet = load_csv(args.dem)
        provider_sheet = load_csv(args.provider)
        insurance_sheet = load_csv(args.insurance)

        if (
            dem_sheet is not None
            and provider_sheet is not None
            and insurance_sheet is not None
        ):
            process_data(dem_sheet, provider_sheet, insurance_sheet)
        else:
            logging.error(
                "Error loading one or more files. Please check the file paths."
            )
    else:
        # GUI mode
        logging.info("Starting in GUI mode.")
        app = App()
        app.mainloop()


if __name__ == "__main__":
    main()
