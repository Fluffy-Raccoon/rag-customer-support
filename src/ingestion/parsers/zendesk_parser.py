import pandas as pd


def parse_zendesk_csv(csv_path: str) -> list[str]:
    """Parse a Zendesk CSV export into Q&A pair documents.

    Expected columns: ticket_id, subject, description, agent_response
    Returns a list of text documents (one per ticket).
    """
    df = pd.read_csv(csv_path)

    required = {"subject", "description", "agent_response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    documents: list[str] = []
    for _, row in df.iterrows():
        subject = str(row.get("subject", "")).strip()
        description = str(row.get("description", "")).strip()
        response = str(row.get("agent_response", "")).strip()

        if not response or response == "nan":
            continue

        doc = (
            f"Customer Question: {subject}\n"
            f"{description}\n\n"
            f"Agent Response: {response}"
        )
        documents.append(doc)

    return documents
