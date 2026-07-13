import pandas as pd
from config import SAP_DATA_FILE

def load_sap_documents():
    """Load SAP documents from CSV file."""
    df = pd.read_csv(SAP_DATA_FILE)
    documents = df["content"].tolist()
    return documents
