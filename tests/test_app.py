from __future__ import annotations


def test_reset_returns_clean_session(app_client):
    response = app_client.post("/reset", json={"seed": 42})
    assert response.status_code == 200

    payload = response.json()
    assert payload["session_id"]

    observation = payload["observation"]
    state = payload["state"]
    assert observation["budget_spent"] == 0
    assert observation["question_idx"] == 0
    assert observation["done"] is False
    assert observation["tools_used_this_question"] == []
    assert state["budget_spent"] == 0
    assert state["current_question_idx"] == 0
    assert state["step_count"] == 0


def test_step_charges_the_right_cost(app_client):
    reset = app_client.post("/reset", json={"seed": 42})
    session_id = reset.json()["session_id"]

    response = app_client.post(
        "/step",
        params={"session_id": session_id},
        json={"tool_id": "calculator", "expression": "2 + 2"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["reward"] == -0.1
    assert payload["state"]["budget_spent"] == 0.1
    assert payload["observation"]["last_tool_result"]["tool_id"] == "calculator"
    assert payload["observation"]["last_tool_result"]["cost"] == 0.1


def test_commit_advances_question_state_correctly(app_client):
    reset = app_client.post("/reset", json={"seed": 42})
    session_id = reset.json()["session_id"]

    response = app_client.post(
        "/step",
        params={"session_id": session_id},
        json={"tool_id": "commit", "answer": "4"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["done"] is False
    assert payload["state"]["questions_answered"] == 1
    assert payload["state"]["current_question_idx"] == 1
    assert payload["observation"]["question_done"] is True


def test_episode_termination_and_cleanup_behave_as_expected(app_client):
    reset = app_client.post("/reset", json={"seed": 42})
    session_id = reset.json()["session_id"]

    first = app_client.post(
        "/step",
        params={"session_id": session_id},
        json={"tool_id": "commit", "answer": "4"},
    )
    assert first.status_code == 200

    second = app_client.post(
        "/step",
        params={"session_id": session_id},
        json={"tool_id": "commit", "answer": "4"},
    )
    assert second.status_code == 200
    assert second.json()["done"] is True

    follow_up = app_client.post(
        "/step",
        params={"session_id": session_id},
        json={"tool_id": "commit", "answer": "4"},
    )
    assert follow_up.status_code == 404
