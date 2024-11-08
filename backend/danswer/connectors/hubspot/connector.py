import time
import concurrent.futures
from datetime import datetime
from datetime import timezone
from typing import Any, Generator, Optional, List
from danswer.connectors.cross_connector_utils.rate_limit_wrapper import rate_limit_builder
from danswer.connectors.models import BasicExpertInfo
from danswer.utils.retry_wrapper import retry_builder
from functools import partial

import requests
from hubspot import HubSpot  # type: ignore
from hubspot.crm.objects import ApiException
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from danswer.configs.app_configs import INDEX_BATCH_SIZE
from danswer.configs.constants import DocumentSource
from danswer.connectors.interfaces import GenerateDocumentsOutput
from danswer.connectors.interfaces import LoadConnector
from danswer.connectors.interfaces import PollConnector
from danswer.connectors.interfaces import SecondsSinceUnixEpoch
from danswer.connectors.models import ConnectorMissingCredentialError
from danswer.connectors.models import Document
from danswer.connectors.models import Section
from danswer.utils.logger import setup_logger

HUBSPOT_BASE_URL = "https://app.hubspot.com/contacts/"
HUBSPOT_API_URL = "https://api.hubapi.com/integrations/v1/me"

logger = setup_logger()


class HubSpotConnector(LoadConnector, PollConnector):
    def __init__(
        self,
        batch_size: int = INDEX_BATCH_SIZE,
        access_token: Optional[str] = None,
        max_workers: int = 3,
        request_timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """Initialize HubSpot connector
        
        Args:
            batch_size: Number of documents to process at once
            access_token: HubSpot API access token
        """
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.batch_size = batch_size
        self.access_token = access_token
        self.portal_id: str | None = None
        self.ticket_base_url = HUBSPOT_BASE_URL
        self.max_workers = max_workers
        self.request_timeout = request_timeout
        
        # Setup connection pooling and retries
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
        self.session.mount("https://", adapter)
        self._api_client = None

    def get_portal_id(self) -> str:
        """Get HubSpot portal ID for constructing URLs
        
        Returns:
            Portal ID as string
            
        Raises:
            Exception: If portal ID cannot be fetched
        """
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(
                HUBSPOT_API_URL, 
                headers=headers,
                timeout=30
            )
            if response.status_code != 200:
                raise Exception("Error fetching portal ID")
            
            data = response.json()
            return data["portalId"]
        except Exception as e:
            logger.error(f"Failed to get portal ID: {e}")
            raise

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        self.access_token = credentials["hubspot_access_token"]

        if self.access_token:
            self.portal_id = self.get_portal_id()
            self.ticket_base_url = f"{HUBSPOT_BASE_URL}{self.portal_id}/ticket/"

        return None

    @retry_builder()
    @rate_limit_builder(max_calls=190, period=10)  # HubSpot API allows 190 calls/10 seconds
    def _get_api_client(self) -> HubSpot:
        """Get or create HubSpot API client"""
        if self._api_client is None:
            self._api_client = HubSpot(access_token=self.access_token)
        return self._api_client

    def _process_note_batch(
        self,
        note_batch: List[Any],
        start: datetime | None = None,
        end: datetime | None = None
    ) -> List[Document]:
        """Process a batch of notes in parallel"""
        api_client = self._get_api_client()
        note_docs: List[Document] = []
        
        for note in note_batch:
            if not note.properties.get("hs_body_preview"):
                continue
                
            updated_at = note.updated_at.replace(tzinfo=None)
            if start is not None and updated_at < start:
                continue
            if end is not None and updated_at > end:
                continue

            content_text = note.properties.get("hs_note_body", note.properties["hs_body_preview"])
            creator = note.properties.get("hs_created_by")
            creator_str = str(creator) if creator is not None else "Unknown"
            
            note_docs.append(
                Document(
                    id=f"note_{note.id}",
                    sections=[Section(link=f"{self.ticket_base_url}", text=content_text)],
                    source=DocumentSource.HUBSPOT,
                    semantic_identifier=f"Note by {creator_str}",
                    doc_updated_at=note.updated_at.replace(tzinfo=timezone.utc),
                    metadata={"type": "note"},
                    primary_owners=[
                        BasicExpertInfo(
                            display_name=creator_str,
                            email=None,  # Could fetch email if available in API
                        )
                    ],
                )
            )
        
        return note_docs

    def _process_all_notes(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> list[Document]:
        """Fetch all notes from HubSpot using parallel processing"""
        api_client = self._get_api_client()
        note_docs: List[Document] = []
        
        # Use pagination to get all notes
        after = None
        while True:
            try:
                response = api_client.crm.objects.notes.basic_api.get_page(
                    properties=["hs_body_preview", "hs_created_by", "hs_note_body"],
                    after=after,
                    limit=100
                )
            except ApiException as e:
                if e.status == 429:  # Rate limit exceeded
                    logger.warning("Rate limit hit, waiting before retry")
                    time.sleep(60)
                    continue
                raise
            
            results = response.results
            if not results:
                break

            # Get paging info before processing
            has_more = bool(response.paging)
            next_after = response.paging.next.after if response.paging else None

            # Process notes in parallel batches
            batch_size = min(len(results), 20)  # Process up to 20 notes at once
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Split results into smaller batches
                batches = [results[i:i + batch_size] for i in range(0, len(results), batch_size)]
                
                # Process each batch in parallel
                process_batch = partial(self._process_note_batch, start=start, end=end)
                results = executor.map(process_batch, batches)
                
                # Collect results
                for batch_docs in results:
                    note_docs.extend(batch_docs)

            # Clear response to free memory
            del response
            del results

            if not has_more:
                break
                
            after = next_after

        return note_docs

    def _process_tickets(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> GenerateDocumentsOutput:
        if self.access_token is None:
            raise ConnectorMissingCredentialError("HubSpot")

        try:
            api_client = HubSpot(access_token=self.access_token)
            # Get all standalone notes first
            doc_batch = self._process_all_notes(start=start, end=end)
        except Exception as e:
            logger.error(f"Failed to initialize HubSpot client: {e}")
            raise
        
        # Then process tickets with pagination
        after = None
        while True:
            try:
                response = api_client.crm.tickets.basic_api.get_page(
                    associations=["contacts", "notes"],
                    after=after,
                    limit=100
                )
                
                results = response.results
                if not results:
                    break

                for ticket in results:
                    updated_at = ticket.updated_at.replace(tzinfo=None)
                    if start is not None and updated_at < start:
                        continue
                    if end is not None and updated_at > end:
                        continue

                    title = ticket.properties.get("subject") or "Untitled Ticket"
                    link = self.ticket_base_url + ticket.id
                    content_text = ticket.properties.get("content", "")

                    associated_emails: list[str] = []
                    associated_notes: list[str] = []

                    if ticket.associations:
                        contacts = ticket.associations.get("contacts")
                        notes = ticket.associations.get("notes")

                        if contacts:
                            try:
                                for contact in contacts.results:
                                    contact_data = api_client.crm.contacts.basic_api.get_by_id(
                                        contact_id=contact.id
                                    )
                                    if "email" in contact_data.properties:
                                        associated_emails.append(contact_data.properties["email"])
                                    # Explicitly delete reference to free memory
                                    del contact_data
                            except Exception as e:
                                logger.error(f"Error fetching contact data: {e}")

                        if notes:
                            try:
                                for note in notes.results:
                                    note_data = api_client.crm.objects.notes.basic_api.get_by_id(
                                        note_id=note.id, properties=["content", "hs_body_preview"]
                                    )
                                    if note_data.properties.get("hs_body_preview"):
                                        associated_notes.append(note_data.properties["hs_body_preview"])
                                    # Explicitly delete reference to free memory
                                    del note_data
                            except Exception as e:
                                logger.error(f"Error fetching note data: {e}")

                    # Filter out None values and ensure strings
                    associated_emails_str = " ,".join(filter(None, (str(email) for email in associated_emails)))
                    associated_notes_str = " ".join(filter(None, (str(note) for note in associated_notes)))

                    content_text = (content_text or "").strip()
                    content_text = f"{content_text}\n emails: {associated_emails_str} \n notes: {associated_notes_str}"

                    doc_batch.append(
                        Document(
                            id=ticket.id,
                            sections=[Section(link=link, text=content_text)],
                            source=DocumentSource.HUBSPOT,
                            semantic_identifier=title,
                            # Is already in tzutc, just replacing the timezone format
                            doc_updated_at=ticket.updated_at.replace(tzinfo=timezone.utc),
                            metadata={},
                        )
                    )

                    if len(doc_batch) >= self.batch_size:
                        yield doc_batch
                        doc_batch = []

                # Handle pagination
                if not response.paging:
                    break
                    
                after = response.paging.next.after

            except ApiException as e:
                if e.status == 429:  # Rate limit exceeded
                    logger.warning("Rate limit hit, waiting before retry")
                    time.sleep(60)
                    continue
                raise
            except Exception as e:
                logger.error(f"Error processing tickets: {e}")
                raise

        # Yield any remaining documents
        if doc_batch:
            yield doc_batch

    def load_from_state(self) -> GenerateDocumentsOutput:
        return self._process_tickets()

    def poll_source(
        self, start: SecondsSinceUnixEpoch, end: SecondsSinceUnixEpoch
    ) -> GenerateDocumentsOutput:
        start_datetime = datetime.utcfromtimestamp(start)
        end_datetime = datetime.utcfromtimestamp(end)
        return self._process_tickets(start_datetime, end_datetime)
        
    def __del__(self) -> None:
        """Cleanup method to ensure proper resource disposal"""
        try:
            # Clear any cached data
            self.portal_id = None
            self.access_token = None
            
            # Close any active API client
            if hasattr(self, '_api_client'):
                self._api_client.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    import os

    connector = HubSpotConnector()
    connector.load_credentials(
        {"hubspot_access_token": os.environ["HUBSPOT_ACCESS_TOKEN"]}
    )

    document_batches = connector.load_from_state()
    print(next(document_batches))
