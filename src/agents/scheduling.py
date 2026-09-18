import logging
from datetime import datetime, timedelta
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import and_

from src.db.session import get_db_session
from src.models.entities import Appointment, Doctor, Patient

logger = logging.getLogger(__name__)


class AvailableSlot(BaseModel):
    doctor_id: str
    doctor_name: str
    specialty: str
    location: str
    slot_time: str  # ISO 8601 datetime string


class BookingResult(BaseModel):
    success: bool
    appointment_id: Optional[str] = None
    doctor_name: Optional[str] = None
    specialty: Optional[str] = None
    location: Optional[str] = None
    slot_time: Optional[str] = None
    message: str


class SchedulingAgent:
    """
    Scheduling Agent that queries doctors by specialty for available slots
    and books confirmed appointments in Postgres.
    """

    def find_available_slots(
        self, specialty: str, limit: int = 5
    ) -> List[AvailableSlot]:
        """
        Queries Postgres for doctors matching specialty and returns their next N available slots.
        Filters out already-booked slots.
        """
        logger.info(f"[SchedulingAgent] Querying availability for specialty: '{specialty}'")
        slots: List[AvailableSlot] = []

        with get_db_session() as db:
            doctors = (
                db.query(Doctor)
                .filter(Doctor.specialty.ilike(f"%{specialty}%"))
                .all()
            )

            if not doctors:
                logger.warning(f"[SchedulingAgent] No doctors found for specialty: {specialty}")
                return []

            # Fetch already-booked start times per doctor to filter out conflicts
            booked_times: dict[str, set] = {}
            for doc in doctors:
                booked = db.query(Appointment.start_time).filter(
                    and_(
                        Appointment.doctor_id == doc.id,
                        Appointment.status == "scheduled",
                    )
                ).all()
                booked_times[doc.id] = {str(b.start_time) for b in booked}

            for doc in doctors:
                for slot_iso in doc.availability_slots:
                    try:
                        slot_dt = datetime.fromisoformat(slot_iso)
                        # Skip past slots
                        if slot_dt < datetime.utcnow():
                            continue
                        # Skip already booked
                        if slot_iso in booked_times.get(doc.id, set()):
                            continue
                        slots.append(AvailableSlot(
                            doctor_id=doc.id,
                            doctor_name=doc.name,
                            specialty=doc.specialty,
                            location=doc.location,
                            slot_time=slot_iso,
                        ))
                        if len(slots) >= limit:
                            break
                    except ValueError:
                        continue
                if len(slots) >= limit:
                    break

        logger.info(f"[SchedulingAgent] Found {len(slots)} available slots for '{specialty}'.")
        return slots

    def book_appointment(
        self,
        doctor_id: str,
        slot_time: str,
        patient_name: str,
        duration_minutes: int = 30,
        patient_id: Optional[str] = None,
    ) -> BookingResult:
        """
        Books an appointment slot for a patient with the specified doctor.
        Creates a Patient record automatically if patient_id is not supplied.
        """
        logger.info(f"[SchedulingAgent] Booking slot {slot_time} with doctor_id={doctor_id} for patient='{patient_name}'")
        try:
            start_time = datetime.fromisoformat(slot_time)
            end_time = start_time + timedelta(minutes=duration_minutes)
        except ValueError as e:
            return BookingResult(success=False, message=f"Invalid slot_time format: {e}")

        with get_db_session() as db:
            # Confirm doctor exists
            doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
            if not doctor:
                return BookingResult(success=False, message=f"Doctor with id '{doctor_id}' not found.")

            # Conflict check: ensure slot is not already booked
            existing = db.query(Appointment).filter(
                and_(
                    Appointment.doctor_id == doctor_id,
                    Appointment.start_time == start_time,
                    Appointment.status == "scheduled",
                )
            ).first()
            if existing:
                return BookingResult(success=False, message=f"Slot at {slot_time} is already booked.")

            # Capture doctor fields inside session scope to avoid DetachedInstanceError
            doctor_name = doctor.name
            doctor_specialty = doctor.specialty
            doctor_location = doctor.location

            # Create or fetch patient record
            if not patient_id:
                patient = Patient(name=patient_name)
                db.add(patient)
                db.flush()
                patient_id = patient.id

            # Book the appointment
            appt = Appointment(
                patient_id=patient_id,
                doctor_id=doctor_id,
                start_time=start_time,
                end_time=end_time,
                status="scheduled",
            )
            db.add(appt)
            db.flush()
            appt_id = appt.id

        logger.info(
            f"[SchedulingAgent] Successfully booked appointment {appt_id} "
            f"with {doctor_name} on {slot_time}."
        )
        return BookingResult(
            success=True,
            appointment_id=appt_id,
            doctor_name=doctor_name,
            specialty=doctor_specialty,
            location=doctor_location,
            slot_time=slot_time,
            message=(
                f"Your appointment with {doctor_name} ({doctor_specialty}) has been confirmed "
                f"on {start_time.strftime('%A, %B %d at %I:%M %p')} at {doctor_location}."
            ),
        )
