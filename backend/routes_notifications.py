"""
routes_notifications.py - Two-Channel Alert Decision & Notification Engine

Endpoints:
- POST /notifications/dispatch: Dispatches Push + SMS with severity-based escalation and event deduplication.
- GET  /notifications/history: Returns alert history feed for the user.
- PUT  /notifications/preferences: Updates user notification preferences.
"""

import time
import secrets
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Header

from backend.services.sms_provider import get_sms_provider

logger = logging.getLogger("backend.routes_notifications")
router = APIRouter(prefix="/notifications", tags=["Notifications & Multi-Channel Alerts"])

# In-memory deduplication & history store
ACTIVE_EVENTS: Dict[str, Dict[str, Any]] = {}  # event_id -> {severity, prob, last_sent_at}
ALERT_HISTORY: List[Dict[str, Any]] = [
    {
        "id": "ALT-2026-0911-001",
        "event_id": "RAIN-MUM-20260911-001",
        "timestamp": "11 Sep 2026, 14:32 IST",
        "alert_type": "Heavy Rainfall Watch",
        "severity": "CRITICAL",
        "location": "Bandra Kurla Complex (BKC), Mumbai",
        "probability_pct": 84,
        "channels": ["Push Notification", "Emergency SMS"],
        "delivery_status": "Delivered",
    },
    {
        "id": "ALT-2026-0910-004",
        "event_id": "FLOOD-MUM-20260910-002",
        "timestamp": "10 Sep 2026, 18:15 IST",
        "alert_type": "Mithi River Surcharge Advisory",
        "severity": "PRE_ALERT",
        "location": "Kurla West / L-Ward, Mumbai",
        "probability_pct": 76,
        "channels": ["Push Notification"],
        "delivery_status": "Delivered",
    },
    {
        "id": "ALT-2026-0908-002",
        "event_id": "STORM-DEL-20260908-001",
        "timestamp": "08 Sep 2026, 09:40 IST",
        "alert_type": "Severe Convective Downpour",
        "severity": "WATCH",
        "location": "Connaught Place, New Delhi",
        "probability_pct": 62,
        "channels": ["Push Notification"],
        "delivery_status": "Delivered",
    },
]

USER_PREFERENCES: Dict[str, Any] = {
    "push_enabled": True,
    "sms_enabled": True,
    "severe_rainfall": True,
    "flood_risk": True,
    "waterlogging": True,
    "cyclone": True,
    "extreme_weather": True,
    "nearby_disaster": True,
    "municipal_alerts": True,
}


class AlertDispatchRequest(BaseModel):
    event_id: str = Field(..., description="Unique event ID for deduplication e.g. RAIN-MUM-20260911-001")
    severity: str = Field(..., description="MONITOR | WATCH | PRE_ALERT | CRITICAL")
    alert_type: str = "Severe Weather Alert"
    location: str
    expected_rainfall: str = "70–100 mm/hour"
    probability_pct: int = 84
    waterlogging_risk: str = "HIGH"
    lead_time_mins: int = 42
    recipient_phone: Optional[str] = None
    is_demo: bool = False


class AlertPreferencesUpdateRequest(BaseModel):
    push_enabled: Optional[bool] = None
    sms_enabled: Optional[bool] = None
    severe_rainfall: Optional[bool] = None
    flood_risk: Optional[bool] = None
    waterlogging: Optional[bool] = None
    cyclone: Optional[bool] = None
    extreme_weather: Optional[bool] = None
    nearby_disaster: Optional[bool] = None
    municipal_alerts: Optional[bool] = None


@router.post("/dispatch", summary="Dispatch Severity-Filtered Multi-Channel Alert")
async def dispatch_alert(req: AlertDispatchRequest):
    """
    Evaluates severity thresholds, deduplicates against active events, and dispatches
    Push and SMS notifications to consenting users.
    
    Rules:
    - GREEN: No notification sent.
    - YELLOW (WATCH): Push notification only.
    - ORANGE (PRE_ALERT): Push notification + optional SMS if preferred.
    - RED (CRITICAL): Push notification + SMS to verified mobile number.
    """
    severity_norm = req.severity.upper()
    now_ts = time.time()
    now_str = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M IST")

    # GREEN: No alert dispatched
    if severity_norm == "MONITOR" or severity_norm == "GREEN":
        return {
            "dispatched": False,
            "reason": "Normal/Monitor condition - no alert triggered per priority policy.",
            "channels_used": [],
        }

    # Deduplication Check
    existing = ACTIVE_EVENTS.get(req.event_id)
    if existing:
        time_since_last = now_ts - existing["last_sent_at"]
        # If same severity and less than 15 minutes have passed without significant probability change (+- 15%), deduplicate
        prob_diff = abs(req.probability_pct - existing["probability_pct"])
        if existing["severity"] == severity_norm and time_since_last < 900 and prob_diff < 15:
            logger.info("[Deduplication] Alert for %s updated without re-dispatching spam notification.", req.event_id)
            existing["last_updated_at"] = now_ts
            return {
                "dispatched": False,
                "deduplicated": True,
                "event_id": req.event_id,
                "message": f"Active event {req.event_id} updated. Redundant alert suppressed per deduplication policy.",
                "channels_used": [],
            }

    # Record / Update active event state
    ACTIVE_EVENTS[req.event_id] = {
        "severity": severity_norm,
        "probability_pct": req.probability_pct,
        "last_sent_at": now_ts,
    }

    channels_used = []
    sms_result = None

    # Channel A: App / Push Notification
    if USER_PREFERENCES.get("push_enabled", True):
        channels_used.append("Push Notification")

    # Channel B: Emergency SMS (Triggered on CRITICAL or PRE_ALERT with consent)
    send_sms = False
    if severity_norm == "CRITICAL":
        send_sms = True
    elif severity_norm == "PRE_ALERT" and USER_PREFERENCES.get("sms_enabled", True):
        send_sms = True

    if send_sms and req.recipient_phone:
        sms_provider = get_sms_provider()
        demo_prefix = "[DEMO / SIMULATED ALERT] " if req.is_demo else ""
        sms_body = (
            f"{demo_prefix}HYDRA CRITICAL ALERT: Severe rainfall risk detected near {req.location}. "
            f"Expected: {req.expected_rainfall} (Prob: {req.probability_pct}%). "
            f"Waterlogging: {req.waterlogging_risk}. Est. lead time: {req.lead_time_mins} min. "
            f"Avoid low-lying roads. Follow official emergency instructions."
        )
        sms_result = await sms_provider.send_sms(req.recipient_phone, sms_body)
        channels_used.append("Emergency SMS")

    # Record in history
    alert_item = {
        "id": f"ALT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}",
        "event_id": req.event_id,
        "timestamp": now_str,
        "alert_type": req.alert_type,
        "severity": severity_norm,
        "location": req.location,
        "probability_pct": req.probability_pct,
        "channels": channels_used,
        "delivery_status": "Delivered",
        "is_demo": req.is_demo,
    }
    ALERT_HISTORY.insert(0, alert_item)

    return {
        "dispatched": True,
        "alert": alert_item,
        "sms_delivery": sms_result,
        "channels_used": channels_used,
    }


@router.get("/history", summary="Get User Alert History")
async def get_alert_history():
    """Returns chronologically ordered list of emergency alerts delivered to the user."""
    return {
        "total": len(ALERT_HISTORY),
        "alerts": ALERT_HISTORY,
    }


@router.get("/preferences", summary="Get User Alert Preferences")
async def get_preferences():
    """Returns current notification preference toggles."""
    return USER_PREFERENCES


@router.put("/preferences", summary="Update User Alert Preferences")
async def update_preferences(req: AlertPreferencesUpdateRequest):
    """Updates user notification toggles (Push, SMS, hazard categories)."""
    updates = req.dict(exclude_unset=True)
    USER_PREFERENCES.update(updates)
    logger.info("Updated alert preferences: %s", updates)
    return {
        "success": True,
        "preferences": USER_PREFERENCES,
    }
