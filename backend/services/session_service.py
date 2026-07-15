from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


@dataclass
class VisionSession:

    session_id: str

    image_bytes: Optional[bytes] = None

    messages: list[dict] = field(
        default_factory=list
    )


class SessionService:

    def __init__(self):

        self.sessions: dict[
            str,
            VisionSession
        ] = {}


    def create_session(
        self
    ) -> VisionSession:

        session_id = str(
            uuid4()
        )

        session = VisionSession(
            session_id=session_id
        )

        self.sessions[
            session_id
        ] = session

        return session


    def get_session(
        self,
        session_id: str
    ) -> Optional[VisionSession]:

        return self.sessions.get(
            session_id
        )


    def get_or_create_session(
        self,
        session_id: Optional[str]
    ) -> VisionSession:

        if session_id:

            session = self.get_session(
                session_id
            )

            if session:

                return session


        return self.create_session()


    def set_image(
        self,
        session: VisionSession,
        image_bytes: bytes
    ) -> None:

        session.image_bytes = image_bytes


    def add_message(
        self,
        session: VisionSession,
        role: str,
        content: str
    ) -> None:

        session.messages.append({

            "role": role,

            "content": content

        })


    def clear_session(
        self,
        session_id: str
    ) -> bool:

        if (
            session_id
            in self.sessions
        ):

            del self.sessions[
                session_id
            ]

            return True


        return False


session_service = SessionService()