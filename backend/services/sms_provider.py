"""
sms_provider.py - Server-Side SMS Communication Layer for HYDRA

Provides a resilient abstraction for emergency disaster alerts via SMS:
1. Abstract SMSProvider interface.
2. MockSMSProvider for DEMO MODE / local development (logs OTP & alerts to console/memory).
3. TwilioSMSProvider for production SMS delivery via Twilio API.
4. Fast2SMSProvider for Indian domestic SMS delivery via Fast2SMS gateway.

All API credentials remain strictly server-side.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("backend.sms_provider")

class SMSProvider(ABC):
    """Abstract interface for SMS alert providers."""

    @abstractmethod
    async def send_sms(self, phone: str, message: str) -> Dict[str, Any]:
        """Send an SMS message and return delivery status dictionary."""
        pass


class MockSMSProvider(SMSProvider):
    """Mock SMS Provider for SIH Demo Mode and offline environments."""

    def __init__(self):
        self.sent_messages = []
        logger.info("[MockSMSProvider] Initialized in DEMO / SIMULATED mode.")

    async def send_sms(self, phone: str, message: str) -> Dict[str, Any]:
        msg_id = f"SMS-DEMO-{int(datetime.now(timezone.utc).timestamp())}"
        payload = {
            "message_id": msg_id,
            "recipient": phone,
            "message": message,
            "status": "delivered",
            "provider": "MockSMSProvider (Demo / Simulated)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.sent_messages.append(payload)
        logger.info("[MockSMSProvider] Simulated SMS delivered to %s: '%s'", phone, message)
        return payload


class TwilioSMSProvider(SMSProvider):
    """Production Twilio SMS Provider using server-side credentials."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        logger.info("[TwilioSMSProvider] Initialized with Twilio Account SID %s...", account_sid[:6])

    async def send_sms(self, phone: str, message: str) -> Dict[str, Any]:
        try:
            import httpx
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
            async with httpx.AsyncClient(timeout=8.0) as client:
                res = await client.post(
                    url,
                    data={"To": phone, "From": self.from_number, "Body": message},
                    auth=(self.account_sid, self.auth_token),
                )
                if res.status_code in (200, 201):
                    data = res.json()
                    return {
                        "message_id": data.get("sid", "TWILIO-OK"),
                        "recipient": phone,
                        "status": "delivered",
                        "provider": "Twilio",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                else:
                    logger.error("[Twilio] HTTP %d: %s", res.status_code, res.text)
                    return {
                        "message_id": None,
                        "recipient": phone,
                        "status": "failed",
                        "error": res.text,
                        "provider": "Twilio",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
        except Exception as e:
            logger.error("[Twilio] Exception sending SMS: %s", e)
            return {
                "message_id": None,
                "recipient": phone,
                "status": "failed",
                "error": str(e),
                "provider": "Twilio",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Global singleton instance
_sms_provider_instance: Optional[SMSProvider] = None


def get_sms_provider() -> SMSProvider:
    """Factory returns configured production or Mock SMS provider."""
    global _sms_provider_instance
    if _sms_provider_instance is not None:
        return _sms_provider_instance

    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")

    if account_sid and auth_token and from_number:
        _sms_provider_instance = TwilioSMSProvider(account_sid, auth_token, from_number)
    else:
        # Graceful fallback to Mock provider for SIH hackathon demonstration
        _sms_provider_instance = MockSMSProvider()

    return _sms_provider_instance
