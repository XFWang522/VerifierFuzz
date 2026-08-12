"""Self-contained HTML reports for local audit triage."""

from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional

from verifierfuzz.engine import AuditFinding
from verifierfuzz.reporting.serialize import json_safe, outcome_to_dict


_SEVERITY = {
    "false_positive": "critical",
    "target_error": "high",
    "reference_error": "high",
    "false_negative": "medium",
    "decision_disagreement": "medium",
    "metamorphic_drift": "medium",
    "inconsistent": "medium",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pretty(value: Any) -> str:
    return _escape(
        json.dumps(
            json_safe(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _finding_card(finding: AuditFinding, index: int) -> str:
    mutation = finding.mutation.name if finding.mutation else "seed"
    severity = _SEVERITY.get(finding.kind, "low")
    completion = (
        finding.minimized_completion
        if finding.minimized_completion is not None
        else finding.case.completion
    )
    searchable = " ".join(
        [
            finding.kind,
            finding.relation,
            finding.case.case_id,
            mutation,
            str(finding.case.prompt),
            str(completion),
        ]
    ).lower()
    return f"""
      <article class="finding" data-kind="{_escape(finding.kind)}"
               data-search="{_escape(searchable)}">
        <header>
          <div>
            <span class="severity {severity}">{severity}</span>
            <strong>{_escape(finding.kind)}</strong>
            <span class="relation">{_escape(finding.relation)}</span>
          </div>
          <code>#{index} · {_escape(finding.case.case_id)}</code>
        </header>
        <div class="outcomes">
          <div><span>Target</span><b>{_escape(finding.target_outcome.decision.value)}</b>
            <small>score={_escape(finding.target_outcome.score)}</small></div>
          <div><span>Reference</span><b>{_escape(finding.reference_outcome.decision.value)}</b>
            <small>score={_escape(finding.reference_outcome.score)}</small></div>
          <div><span>Mutation</span><b>{_escape(mutation)}</b>
            <small>severity={severity}</small></div>
        </div>
        <details open>
          <summary>Counterexample</summary>
          <h4>Prompt</h4><pre>{_pretty(finding.case.prompt)}</pre>
          <h4>Completion</h4><pre>{_pretty(completion)}</pre>
          <h4>Reference</h4><pre>{_pretty(finding.case.reference)}</pre>
        </details>
        <details>
          <summary>Verifier evidence</summary>
          <div class="evidence">
            <div><h4>Target</h4><pre>{_pretty(outcome_to_dict(finding.target_outcome))}</pre></div>
            <div><h4>Reference</h4><pre>{_pretty(outcome_to_dict(finding.reference_outcome))}</pre></div>
          </div>
        </details>
        <details>
          <summary>Metadata and lineage</summary>
          <pre>{_pretty({"metadata": finding.case.metadata, "mutation": finding.mutation})}</pre>
        </details>
      </article>"""


def build_html(
    findings: Iterable[AuditFinding],
    *,
    title: str = "VerifierFuzz Audit Report",
    generated_at: Optional[datetime] = None,
) -> str:
    items: List[AuditFinding] = list(findings)
    counts = Counter(finding.kind for finding in items)
    generated = generated_at or datetime.now(timezone.utc)
    kind_options = "".join(
        f'<option value="{_escape(kind)}">{_escape(kind)} ({count})</option>'
        for kind, count in sorted(counts.items())
    )
    summary_cards = "".join(
        f'<div class="stat"><b>{count}</b><span>{_escape(kind)}</span></div>'
        for kind, count in sorted(counts.items())
    )
    cards = "".join(
        _finding_card(finding, index)
        for index, finding in enumerate(items, start=1)
    )
    if not cards:
        cards = '<div class="empty">No disagreements found.</div>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a2d; --line:#293553;
      --text:#e8edf7; --muted:#91a0bb; --accent:#8b5cf6; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text);
      font:14px/1.5 ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1180px,94vw); margin:36px auto 80px; }}
    h1 {{ margin:0; font-size:32px; }} .subtitle {{ color:var(--muted); margin:6px 0 22px; }}
    .stats {{ display:flex; gap:10px; flex-wrap:wrap; margin:18px 0; }}
    .stat {{ min-width:130px; padding:14px; background:var(--panel);
      border:1px solid var(--line); border-radius:12px; }}
    .stat b {{ display:block; font-size:24px; }} .stat span {{ color:var(--muted); }}
    .controls {{ display:flex; gap:10px; position:sticky; top:0; z-index:2;
      padding:12px 0; background:var(--bg); }}
    input,select {{ width:100%; padding:10px 12px; color:var(--text);
      background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    select {{ max-width:260px; }}
    .finding {{ margin:14px 0; padding:18px; background:var(--panel);
      border:1px solid var(--line); border-radius:14px; }}
    .finding header {{ display:flex; justify-content:space-between; gap:16px;
      align-items:center; }} .relation,small {{ color:var(--muted); }}
    .severity {{ padding:3px 8px; border-radius:999px; margin-right:8px;
      font-size:11px; text-transform:uppercase; }}
    .critical {{ background:#7f1d1d; }} .high {{ background:#9a3412; }}
    .medium {{ background:#854d0e; }} .low {{ background:#334155; }}
    .outcomes {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:16px 0; }}
    .outcomes div {{ padding:10px; border:1px solid var(--line); border-radius:9px; }}
    .outcomes span,.outcomes small {{ display:block; }}
    details {{ border-top:1px solid var(--line); padding-top:10px; margin-top:10px; }}
    summary {{ cursor:pointer; color:#c4b5fd; }} h4 {{ margin:12px 0 4px; color:var(--muted); }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; padding:12px;
      background:#070b16; border-radius:8px; color:#dbeafe; }}
    .evidence {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .empty {{ padding:40px; text-align:center; color:var(--muted); }}
    @media(max-width:700px) {{ .outcomes,.evidence {{ grid-template-columns:1fr; }}
      .finding header {{ align-items:flex-start; flex-direction:column; }} }}
  </style>
</head>
<body>
<main>
  <h1>{_escape(title)}</h1>
  <p class="subtitle">{len(items)} findings · generated {_escape(generated.isoformat())}</p>
  <div class="stats"><div class="stat"><b>{len(items)}</b><span>Total</span></div>{summary_cards}</div>
  <div class="controls">
    <input id="search" type="search" placeholder="Search case, mutation, prompt, or completion">
    <select id="kind"><option value="">All finding kinds</option>{kind_options}</select>
  </div>
  <section id="findings">{cards}</section>
</main>
<script>
  const search = document.getElementById('search');
  const kind = document.getElementById('kind');
  function filterFindings() {{
    const query = search.value.toLowerCase();
    document.querySelectorAll('.finding').forEach((node) => {{
      const visible = (!kind.value || node.dataset.kind === kind.value)
        && (!query || node.dataset.search.includes(query));
      node.hidden = !visible;
    }});
  }}
  search.addEventListener('input', filterFindings);
  kind.addEventListener('change', filterFindings);
</script>
</body>
</html>
"""


def write_html(
    path: str,
    findings: Iterable[AuditFinding],
    *,
    title: str = "VerifierFuzz Audit Report",
) -> None:
    Path(path).write_text(
        build_html(findings, title=title),
        encoding="utf-8",
    )
