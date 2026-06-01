from pydantic import BaseModel, Field


class IncomingWhatsAppMessage(BaseModel):
    """Payload sent by the Node.js bridge when a WhatsApp message is received."""

    from_number: str = Field(..., alias="from")
    body: str = ""
    timestamp: int
    type: str = "chat"

    model_config = {"populate_by_name": True}


class WebhookResponse(BaseModel):
    status: str
    request_id: str | None = None
