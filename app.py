"""FastAPI server for CostAwareToolEnv.

Exposes the OpenEnv standard endpoints:
  POST /reset          -> OrchestratorObservation + OrchestratorState
  POST /step           -> OrchestratorObservation + reward + done + state
  GET  /health         -> {"status": "ok"}
  GET  /tools          -> canonical tool manifest
  GET  /web            -> simple demo UI
  GET  /docs           -> OpenAPI (automatic)
"""
from __future__ import annotations

import copy
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from data.loader import load_all
from env.config import EnvConfig
from env.environment import CostAwareToolEnvironment
from env.models import OrchestratorAction
from tools import build_tool_catalog, build_tool_registry, catalog_as_dicts, validate_tool_costs


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    config_overrides: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    tool_id: str
    query: Optional[str] = None
    expression: Optional[str] = None
    code_snippet: Optional[str] = None
    answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _merge_config(base: EnvConfig, overrides: Optional[Dict[str, Any]]) -> EnvConfig:
    cfg = copy.deepcopy(base)
    if not overrides:
        return cfg

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config override: {key}")

        current = getattr(cfg, key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged = copy.deepcopy(current)
            merged.update(value)
            setattr(cfg, key, merged)
        else:
            setattr(cfg, key, value)
    return cfg


def _build_demo_html(tool_catalog: List[Any]) -> str:
    tool_options = "\n".join(
        f'  <option value="{spec.tool_id}">{spec.label} (cost {spec.cost}) — {spec.purpose}</option>'
        for spec in tool_catalog
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CostAwareToolEnv</title>
<style>
  body {{ font-family: monospace; max-width: 860px; margin: 40px auto; padding: 0 20px; }}
  h1   {{ color: #333; }}
  pre  {{ background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }}
  button {{ padding: 8px 16px; margin: 4px; cursor: pointer; }}
  input, select, textarea {{ width: 100%; padding: 6px; margin: 4px 0; box-sizing: border-box; }}
  label {{ font-weight: bold; }}
  .tool-btn {{ background: #e8f0fe; border: 1px solid #4a90e2; border-radius: 4px; }}
  .tool-btn:hover {{ background: #cfe1ff; }}
  #log {{ max-height: 480px; overflow-y: auto; }}
</style>
</head>
<body>
<h1>CostAwareToolEnv</h1>
<p>Multi-tool cost-aware RL environment with explicit tool routing and sandboxed execution.</p>

<button onclick="doReset()">Reset Episode</button>
<hr>
<label>Tool:</label>
<select id="tool">
{tool_options}
</select>
<label>Query / Expression / Code / Answer:</label>
<textarea id="query" rows="3" placeholder="Enter query or answer..."></textarea>
<button class="tool-btn" onclick="doStep()">Step</button>
<hr>
<pre id="log">Click "Reset Episode" to start.</pre>

<script>
const log = document.getElementById('log');
let sessionId = null;

function append(text) {{ log.textContent += text + '\\n---\\n'; log.scrollTop = log.scrollHeight; }}

async function doReset() {{
  log.textContent = '';
  const res = await fetch('/reset', {{ method: 'POST', headers: {{'Content-Type':'application/json'}}, body: JSON.stringify({{seed: 42}}) }});
  const data = await res.json();
  sessionId = data.session_id || null;
  append('RESET session=' + sessionId + '\\n' + JSON.stringify(data, null, 2));
}}

async function doStep() {{
  const tool_id = document.getElementById('tool').value;
  const input   = document.getElementById('query').value;
  const body    = {{ tool_id }};
  if (tool_id === 'commit')         body.answer = input;
  else if (tool_id === 'calculator') body.expression = input;
  else if (tool_id === 'code_executor') body.code_snippet = input;
  else                              body.query = input;

  const url = sessionId ? '/step?session_id=' + encodeURIComponent(sessionId) : '/step';
  const res = await fetch(url, {{ method: 'POST', headers: {{'Content-Type':'application/json'}}, body: JSON.stringify(body) }});
  const data = await res.json();
  append('STEP tool_id=' + tool_id + '\\n' + JSON.stringify(data, null, 2));
}}
</script>
</body>
</html>
"""


def create_app(
    config: Optional[EnvConfig] = None,
    tools: Optional[Dict[str, Any]] = None,
    dataset: Optional[List[Dict[str, Any]]] = None,
    load_dataset_fn: Callable[..., List[Dict[str, Any]]] = load_all,
    build_registry_fn: Callable[[EnvConfig | None], Dict[str, Any]] = build_tool_registry,
) -> FastAPI:
    base_config = config or EnvConfig()
    validate_tool_costs(base_config)

    dataset_cache = dataset
    default_env: Optional[CostAwareToolEnvironment] = None
    sessions: Dict[str, CostAwareToolEnvironment] = {}
    tool_catalog = build_tool_catalog(base_config)
    demo_html = _build_demo_html(tool_catalog)

    def get_dataset() -> List[Dict[str, Any]]:
        nonlocal dataset_cache
        if dataset_cache is None:
            dataset_cache = load_dataset_fn(split=base_config.data_split, max_per_domain=200)
        return dataset_cache

    def make_env(effective_config: EnvConfig) -> CostAwareToolEnvironment:
        registry = tools if tools is not None else build_registry_fn(effective_config)
        return CostAwareToolEnvironment(
            config=effective_config,
            tools=registry,
            dataset=get_dataset(),
        )

    def get_default_env() -> CostAwareToolEnvironment:
        nonlocal default_env
        if default_env is None:
            default_env = make_env(base_config)
        return default_env

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(
        title="CostAwareToolEnv",
        description="Multi-tool cost-aware RL environment (OpenEnv / AgentX)",
        version="0.1.0",
        lifespan=lifespan,
        root_path=os.environ.get("ROOT_PATH", ""),
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/tools")
    def tools_manifest():
        return catalog_as_dicts(base_config)

    @app.post("/reset")
    def reset(req: ResetRequest):
        try:
            cfg = _merge_config(base_config, req.config_overrides)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        env = make_env(cfg)
        obs, state = env.reset(seed=req.seed)

        session_id = str(uuid.uuid4())
        sessions[session_id] = env

        return {
            "session_id": session_id,
            "observation": obs.model_dump(),
            "state": state.model_dump(),
        }

    @app.post("/step")
    def step(req: StepRequest, session_id: Optional[str] = None):
        if session_id is None:
            env = get_default_env()
        else:
            env = sessions.get(session_id)
            if env is None:
                raise HTTPException(status_code=404, detail="Unknown session_id")

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

        if done and session_id and session_id in sessions:
            del sessions[session_id]

        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "state": state.model_dump(),
        }

    @app.get("/web", response_class=HTMLResponse)
    def web_ui():
        return demo_html

    return app


app = create_app()
