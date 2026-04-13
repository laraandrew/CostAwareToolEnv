"""FastAPI server for ToolOrchestratorEnv.

Exposes the OpenEnv standard endpoints:
  POST /reset          → OrchestratorObservation + OrchestratorState
  POST /step           → OrchestratorObservation + reward + done + state
  GET  /health         → {"status": "ok"}
  GET  /web            → simple demo UI
  GET  /docs           → OpenAPI (automatic)
"""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from data.loader import load_all
from env.config import EnvConfig
from env.environment import ToolOrchestratorEnvironment
from env.models import OrchestratorAction
from tools import build_tool_registry


# ---------------------------------------------------------------------------
# Request / response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    config_overrides: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    tool_id: str
    query:        Optional[str] = None
    expression:   Optional[str] = None
    code_snippet: Optional[str] = None
    answer:       Optional[str] = None
    metadata:     Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    config  = EnvConfig()
    tools   = build_tool_registry(config)
    dataset = load_all(split=config.data_split, max_per_domain=200)

    # Multi-session state: session_id → ToolOrchestratorEnvironment
    sessions: Dict[str, ToolOrchestratorEnvironment] = {}

    # Default shared environment for single-session usage (no session_id)
    default_env = ToolOrchestratorEnvironment(config=config, tools=tools, dataset=dataset)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(
        title="ToolOrchestratorEnv",
        description="Multi-tool cost-aware RL environment (OpenEnv / AgentX)",
        version="0.1.0",
        lifespan=lifespan,
        root_path=os.environ.get("ROOT_PATH", ""),
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/reset")
    def reset(req: ResetRequest):
        cfg = EnvConfig()
        if req.config_overrides:
            for k, v in req.config_overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        env = ToolOrchestratorEnvironment(config=cfg, tools=tools, dataset=dataset)
        obs, state = env.reset(seed=req.seed)

        session_id = str(uuid.uuid4())
        sessions[session_id] = env

        return {
            "session_id":  session_id,
            "observation": obs.model_dump(),
            "state":       state.model_dump(),
        }

    @app.post("/step")
    def step(req: StepRequest, session_id: Optional[str] = None):
        env = sessions.get(session_id or "", default_env)
        action = OrchestratorAction(
            tool_id=req.tool_id,
            query=req.query or "",
            expression=req.expression or "",
            code_snippet=req.code_snippet or "",
            answer=req.answer or "",
            metadata=req.metadata,
        )
        try:
            obs, reward, done, state = env.step(action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # Clean up finished sessions
        if done and session_id and session_id in sessions:
            del sessions[session_id]

        return {
            "observation": obs.model_dump(),
            "reward":      reward,
            "done":        done,
            "state":       state.model_dump(),
        }

    @app.get("/web", response_class=HTMLResponse)
    def web_ui():
        return _DEMO_HTML

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Demo UI
# ---------------------------------------------------------------------------

_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ToolOrchestratorEnv</title>
<style>
  body { font-family: monospace; max-width: 860px; margin: 40px auto; padding: 0 20px; }
  h1   { color: #333; }
  pre  { background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }
  button { padding: 8px 16px; margin: 4px; cursor: pointer; }
  input, select, textarea { width: 100%; padding: 6px; margin: 4px 0; box-sizing: border-box; }
  label { font-weight: bold; }
  .tool-btn { background: #e8f0fe; border: 1px solid #4a90e2; border-radius: 4px; }
  .tool-btn:hover { background: #cfe1ff; }
  #log { max-height: 480px; overflow-y: auto; }
</style>
</head>
<body>
<h1>ToolOrchestratorEnv</h1>
<p>Multi-tool cost-aware RL environment — AgentX / OpenEnv</p>

<button onclick="doReset()">Reset Episode</button>
<hr>
<label>Tool:</label>
<select id="tool">
  <option value="ceramic_search">ceramic_search (cost 1.0) — Web retrieval</option>
  <option value="wiki_lookup">wiki_lookup (cost 0.5) — Wikipedia</option>
  <option value="calculator">calculator (cost 0.1) — Arithmetic / math</option>
  <option value="code_executor">code_executor (cost 0.3) — Python execution</option>
  <option value="llm_reason">llm_reason (cost 2.0) — LLM chain-of-thought</option>
  <option value="commit">commit (cost 0.0) — Submit answer</option>
</select>
<label>Query / Expression / Code / Answer:</label>
<textarea id="query" rows="3" placeholder="Enter query or answer..."></textarea>
<button class="tool-btn" onclick="doStep()">Step</button>
<hr>
<pre id="log">Click "Reset Episode" to start.</pre>

<script>
const log = document.getElementById('log');
let sessionId = null;

function append(text) { log.textContent += text + '\\n---\\n'; log.scrollTop = log.scrollHeight; }

async function doReset() {
  log.textContent = '';
  const res = await fetch('/reset', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({seed: 42}) });
  const data = await res.json();
  sessionId = data.session_id || null;
  append('RESET session=' + sessionId + '\\n' + JSON.stringify(data, null, 2));
}

async function doStep() {
  const tool_id = document.getElementById('tool').value;
  const input   = document.getElementById('query').value;
  const body    = { tool_id };
  if (tool_id === 'commit')         body.answer = input;
  else if (tool_id === 'calculator') body.expression = input;
  else if (tool_id === 'code_executor') body.code_snippet = input;
  else                              body.query = input;

  const url = sessionId ? '/step?session_id=' + encodeURIComponent(sessionId) : '/step';
  const res = await fetch(url, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const data = await res.json();
  append('STEP tool_id=' + tool_id + '\\n' + JSON.stringify(data, null, 2));
}
</script>
</body>
</html>
"""
