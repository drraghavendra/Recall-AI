from recall_client.state import ViewState

ALICE = {"names": ["Alice"], "nickname": "", "relation": "colleague", "key_facts": ["works at Google"], "summary": "s"}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make(ttl=210.0):
    clock = FakeClock()
    return ViewState(ttl, clock), clock


def test_new_face_is_fetched_once_and_404_means_first_meeting():
    state, _ = make()
    assert state.face_update("a") == "a"
    assert state.face_update("a") is None  # fetch already in flight
    state.persona_fetched("a", None)
    snap = state.snapshot()
    assert snap.first_meeting and snap.persona is None


def test_same_face_keeps_cache_without_refetching():
    state, clock = make()
    state.face_update("a")
    state.persona_fetched("a", ALICE)
    clock.now = 100
    assert state.face_update("a") is None
    assert state.snapshot().persona == ALICE


def test_cache_ttl_triggers_safety_net_refetch():
    state, clock = make(ttl=210)
    state.face_update("a")
    state.persona_fetched("a", ALICE)
    clock.now = 211
    assert state.face_update("a") == "a"


def test_switching_back_to_a_cached_face_does_not_refetch():
    state, _ = make()
    state.face_update("a")
    state.persona_fetched("a", ALICE)
    assert state.face_update("b") == "b"
    assert state.face_update("a") is None
    assert state.snapshot().persona == ALICE


def test_persona_updated_replaces_first_meeting_message():
    state, _ = make()
    state.face_update("a")
    state.persona_fetched("a", None)
    state.persona_updated("a", ALICE)
    snap = state.snapshot()
    assert snap.persona == ALICE and not snap.first_meeting


def test_no_face_means_no_panel():
    state, _ = make()
    state.face_update("a")
    state.persona_fetched("a", ALICE)
    assert state.face_update(None) is None
    snap = state.snapshot()
    assert snap.face_id is None and snap.persona is None and not snap.first_meeting


def test_failed_fetch_is_retried_on_next_update():
    state, _ = make()
    state.face_update("a")
    state.fetch_failed("a")
    assert state.face_update("a") == "a"


def test_unfetched_face_is_neither_first_meeting_nor_known():
    state, _ = make()
    state.face_update("a")
    snap = state.snapshot()
    assert snap.persona is None and not snap.first_meeting


def test_conversation_tracks_partner_persona():
    state, _ = make()
    state.face_update("a")
    state.persona_fetched("a", ALICE)
    state.conversation_started("a")
    snap = state.snapshot()
    assert snap.in_conversation and snap.conversation_persona == ALICE
    state.conversation_ended()
    assert not state.snapshot().in_conversation
