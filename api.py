import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from backend.database import init_db
from backend.recommender import get_first_song, get_next_song, init_conversation


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(lifespan=lifespan)


class VoteInput(BaseModel):
    artist_name: str
    track_name: str
    known: bool
    liked: bool
    feedback: str | None = None


class RecommendRequest(BaseModel):
    strategist_messages: list[dict[str, Any]] | None = None
    recommender_messages: list[dict[str, Any]] | None = None
    vote: VoteInput | None = None


@app.post("/recommend")
def recommend(body: RecommendRequest):
    session_id = str(uuid.uuid4())

    if body.vote is None:
        track = get_first_song(session_id)
        return {
            "artist_name": track["artist_name"],
            "track_name": track["track_name"],
            "embed_url": track.get("embed_url", ""),
            "strategy": None,
        }

    strat_msgs, rec_msgs = init_conversation()
    if body.strategist_messages:
        strat_msgs = body.strategist_messages
    if body.recommender_messages:
        rec_msgs = body.recommender_messages

    track, strategy, *_ = get_next_song(
        session_id=session_id,
        strategist_messages=strat_msgs,
        recommender_messages=rec_msgs,
        vote_result=body.vote.model_dump(),
    )

    return {
        "artist_name": track["artist_name"],
        "track_name": track["track_name"],
        "embed_url": track.get("embed_url", ""),
        "strategy": strategy,
    }
