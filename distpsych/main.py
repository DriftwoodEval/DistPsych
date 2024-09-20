import threading
import time
import tkinter.filedialog

import chardet
import customtkinter
import pandas as pd
import requests
from logzero import logger


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
        logger.error("Street, city, or state cannot be empty.")
        return "Not found"

    try:
        logger.info(
            f"Searching for {params['street']} {params['city']}, {params['state']} {params['zip']}"
        )
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises HTTPError for bad responses

        # Parse the JSON response
        data = response.json()
        if data["result"]["addressMatches"]:
            match = data["result"]["addressMatches"][0]
            district = match["geographies"]["Unified School Districts"][0]["NAME"]
            logger.info(f"District found: {district}")
            return district
        else:
            logger.warning("Search failed, attempting again without a ZIP code...")
            params.pop("zip")
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data["result"]["addressMatches"]:
                match = data["result"]["addressMatches"][0]
                district = match["geographies"]["Unified School Districts"][0]["NAME"]
                logger.info(f"District found: {district}")
                return district
            else:
                logger.error("No district found.")
                return "Not found"
    except requests.RequestException as e:
        logger.error(f"Error fetching school district data: {e}")
        return "Not found"


def pick_file():
    return tkinter.filedialog.askopenfilename()


def create_provider_location_dict(df: pd.DataFrame):
    """
    Creates a dictionary mapping provider names to their corresponding locations.

    Args:
        df: A pandas DataFrame containing the employee data.

    Returns:
        A dictionary where keys are provider names and values are their locations.
    """
    provider_names = df.columns[2:]
    provider_locations = {}
    location_index = df[df["Unnamed: 0"] == "Location"].index.item()

    for i, provider_name in enumerate(provider_names):
        # The location information for each provider is stored in the row with "Location" label
        location = df.iloc[location_index, i + 2]  # i + 2 accounts for the offset
        provider_locations[provider_name] = {"district": None, "CAN": [], "CANNOT": []}
        if pd.notna(location):
            location = location.replace("DD4", "Dorchester District 4")
            location = location.replace("Berkeley", "Berkeley County School District")
            location = location.replace(
                "Georgetown", "Georgetown County School District"
            )
            location = location.replace("Horry", "Horry County School District")
            location = location.replace(
                "Charleston", "Charleston County School District"
            )
            if "District" in location:
                if "Only" in location:
                    provider_locations[provider_name]["district"] = location
                provider_locations[provider_name]["district"] = location
            elif "ALL" in location:
                provider_locations[provider_name]["district"] = "ALL"

        # Extract CAN zip codes from the provider's column, separating them
        for row_index in range(df.shape[0]):
            cell_value = df.iloc[row_index, i + 2]
            if isinstance(cell_value, str) and "CANNOT" in cell_value:
                cannot_zip_codes = cell_value.split("CANNOT")[-1].strip().split(" ")
                for zip_code in cannot_zip_codes:
                    provider_locations[provider_name]["CANNOT"].append(zip_code.strip())
            elif isinstance(cell_value, str) and "CAN" in cell_value:
                can_zip_codes = cell_value.split("CAN")[-1].strip().split(" ")
                for zip_code in can_zip_codes:
                    provider_locations[provider_name]["CAN"].append(zip_code.strip())
    provider_locations = {
        k: v for k, v in provider_locations.items() if not k.startswith("Unnamed")
    }
    for provider_name, provider_data in provider_locations.items():
        if not any(provider_data.values()):
            logger.warning(f"Provider {provider_name} is missing a location.")
    logger.info(provider_locations)
    return provider_locations


def get_provider_insurance(df: pd.DataFrame) -> dict:
    """
    Extracts provider insurance information from a DataFrame.

    Args:
      df: A pandas DataFrame containing provider insurance data.

    Returns:
      A dictionary where keys are provider names and values are lists of insurance
      they can take.
    """
    provider_names = df.columns[2:]
    provider_insurance = {}
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if row["Unnamed: 0"] == "Insurance":
            continue  # Skip the label
        if row["Unnamed: 0"] == "Location":
            break

        insurance_name = row["Unnamed: 0"]

        for j, name in enumerate(provider_names):
            # Check if provider is listed for this insurance
            cell_value = row.iloc[j + 2]
            if pd.notna(cell_value) and (
                cell_value == "X" or all(c.isdigit() for c in str(cell_value))
            ):
                # Create a list of insurance for each provider if it doesn't exist yet
                if name not in provider_insurance:
                    provider_insurance[name] = []
                insurance_name = insurance_name.replace(
                    "BABYNET", "BabyNet (Combined DA and Eval)"
                )
                insurance_name = insurance_name.replace(
                    "SCM", "Medicaid South Carolina"
                )
                insurance_name = insurance_name.replace(
                    "ATC", "Absolute Total Care - Medical"
                )
                insurance_name = insurance_name.replace(
                    "SH", "Select Health of South Carolina"
                )
                insurance_name = insurance_name.replace("SCM", "Size Count Medium")
                provider_insurance[name].append(insurance_name)

    return provider_insurance


def extract_client_data(df: pd.DataFrame):
    """
    Extracts relevant client data from a DataFrame.

    Args:
        df: A pandas DataFrame containing client information.

    Returns:
        A new DataFrame containing only active clients (status != "Inactive")
        and the following columns:
            - CLIENT_ID
            - LASTNAME
            - FIRSTNAME
            - PREFERRED_NAME
            - USER_ADDRESS_ADDRESS1
            - USER_ADDRESS_CITY
            - USER_ADDRESS_STATE
            - USER_ADDRESS_ZIP
    """
    active_clients = df[df["STATUS"] != "Inactive"]
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
    extracted_df = active_clients[selected_columns]

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
    df["USER_DISTRICT"] = None

    for index, row in df.iterrows():
        street_address = str(row["USER_ADDRESS_ADDRESS1"])
        city_name = str(row["USER_ADDRESS_CITY"])
        state_name = str(row["USER_ADDRESS_STATE"])
        zip = str(row["USER_ADDRESS_ZIP"])

        district = get_district(
            street=street_address, city=city_name, state=state_name, zip=zip
        )

        df.loc[index, "USER_DISTRICT"] = district

    return df


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
    # Create a dictionary to map CLIENT_ID to INSURANCE_COMPANYNAME
    insurance_mapping = dict(
        zip(insurance_df["CLIENT_ID"], insurance_df["INSURANCE_COMPANYNAME"])
    )

    # Define a function to retrieve insurance from the dictionary
    def get_insurance_company(client_id):
        return insurance_mapping.get(client_id, "None found")  # Use get for safe lookup

    # Add a new column 'INSURANCE_COMPANYNAME' to the client DataFrame
    client_df["INSURANCE_COMPANYNAME"] = client_df["CLIENT_ID"].map(
        get_insurance_company
    )

    return client_df


def assign_providers_to_clients(
    client_df: pd.DataFrame, provider_locations: dict, provider_insurance: dict
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

    for provider_name in provider_locations.keys():
        client_df[provider_name] = (
            None  # Add a column for each provider with NaN values
        )

    for index, row in client_df.iterrows():
        client_district = row["USER_DISTRICT"]
        client_zip = row["USER_ADDRESS_ZIP"]
        client_insurance = row["INSURANCE_COMPANYNAME"]
        for provider_name, provider_data in provider_locations.items():
            if provider_name in provider_insurance:
                if (
                    provider_data["district"] is not None
                    and "Only" in provider_data["district"]
                    and client_district in provider_data["district"]
                    and client_insurance in provider_insurance[provider_name]
                ):
                    logger.info(
                        f"Provider match: {provider_name} - {client_district} - {client_insurance}"
                    )
                    client_df.loc[index, provider_name] = "X"
                elif (
                    provider_data["district"] != client_district
                    and client_insurance in provider_insurance[provider_name]
                ):
                    logger.info(
                        f"Provider match: {provider_name} - {client_district} - {client_insurance}"
                    )
                    client_df.loc[index, provider_name] = "X"
                elif (
                    client_zip in provider_data["CAN"]
                    and client_insurance in provider_insurance[provider_name]
                ):
                    logger.info(
                        f"Provider match: {provider_name} - {client_zip} - {client_insurance}"
                    )
                    client_df.loc[index, provider_name] = "X"
                elif (
                    provider_data["CANNOT"]
                    and client_zip not in provider_data["CANNOT"]
                    and client_insurance in provider_insurance[provider_name]
                ):
                    logger.info(
                        f"Provider match: {provider_name} - {client_zip} - {client_insurance}"
                    )
                    client_df.loc[index, provider_name] = "X"

    return client_df


def load_csv(file_path):
    """Loads a CSV file with error handling and encoding detection."""
    try:
        with open(file_path, "rb") as f:  # Open in binary mode for encoding detection
            data = f.read()
            encoding = chardet.detect(data)["encoding"]  # Detect encoding
            df = pd.read_csv(
                file_path, encoding=encoding
            )  # Read with detected encoding
            return df
    except UnicodeDecodeError as e:
        print(f"Error decoding file: {e}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"Error reading file, may be empty or corrupt: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.dem_sheet = None
        self.provider_sheet = None
        self.insurance_sheet = None

        self.title("DistPsych")
        self.geometry("500x400")
        self.grid_columnconfigure(0, weight=1)

        self.dem_sheet_button = customtkinter.CTkButton(
            self, text="Select Demographics Sheet", command=self.get_dem_sheet
        )
        self.dem_sheet_button.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        self.provider_sheet_button = customtkinter.CTkButton(
            self, text="Select Provider Sheet", command=self.get_provider_sheet
        )
        self.provider_sheet_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.insurance_sheet_button = customtkinter.CTkButton(
            self, text="Select Insurance Sheet", command=self.get_insurance_sheet
        )
        self.insurance_sheet_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.process_button = customtkinter.CTkButton(
            self, text="Process!", command=self.process_sheet, state="disabled"
        )
        self.process_button.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.log_frame = customtkinter.CTkFrame(self)
        self.log_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        self.log_text = customtkinter.CTkTextbox(
            self.log_frame,
            width=350,
            height=100,
            wrap="word",
            state="normal",
        )
        self.log_text.pack(side="left", fill="both", expand=True)

    def get_dem_sheet(self):
        file = pick_file()
        if file:
            self.dem_sheet = load_csv(file)
            self.dem_sheet_button.configure(text=file.split("/")[-1])
            self.check_process_button_state()

    def get_provider_sheet(self):
        file = pick_file()
        if file:
            self.provider_sheet = load_csv(file)
            self.provider_sheet_button.configure(text=file.split("/")[-1])
            self.check_process_button_state()

    def get_insurance_sheet(self):
        file = pick_file()
        if file:
            self.insurance_sheet = load_csv(file)
            self.insurance_sheet_button.configure(text=file.split("/")[-1])
            self.check_process_button_state()

    def process_sheet(self):
        def add_log(message):
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            time.sleep(0.01)  # Allow GUI to update

        # Create a new thread for the processing
        processing_thread = threading.Thread(target=self._process_data, args=(add_log,))
        processing_thread.start()

    def _process_data(self, add_log):
        if (
            self.dem_sheet is not None
            and self.provider_sheet is not None
            and self.insurance_sheet is not None
        ):
            add_log("Extracting client data...")
            trimmed_clients = extract_client_data(self.dem_sheet)
            add_log("Getting districts from US Census... (this may take a while)")
            clients_with_districts = add_districts_to_clients(
                pd.DataFrame(trimmed_clients)
            )
            add_log("Matching client IDs and adding insurance...")
            clients_with_districts_and_insurance = match_client_ids_and_add_insurance(
                clients_with_districts, self.insurance_sheet
            )
            add_log("Creating provider location dictionary...")
            provider_locations = create_provider_location_dict(self.provider_sheet)
            add_log("Getting provider insurance information...")
            provider_insurance = get_provider_insurance(self.provider_sheet)
            add_log("Assigning providers to clients...")
            result_df = assign_providers_to_clients(
                clients_with_districts_and_insurance,
                provider_locations,
                provider_insurance,
            )
            add_log("Saving results to RESULTS.csv...")
            result_df.to_csv("RESULTS.csv", index=False)
            add_log("DONE!")

    def check_process_button_state(self):
        if (
            self.dem_sheet is not None
            and self.provider_sheet is not None
            and self.insurance_sheet is not None
        ):
            self.process_button.configure(state="normal")
        else:
            self.process_button.configure(state="disabled")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
