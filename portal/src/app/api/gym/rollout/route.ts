import { NextResponse } from "next/server";

/**
 * One episode of one policy, step by step.
 *
 * Proxies the service's `/gym/rollout`. The per-step cost is returned
 * alongside the running total because the budget is spent over the whole
 * episode, so a per-step cost that looks small is not the quantity under
 * constraint.
 *
 * 48 steps of real routing lookups (`FloodEvacuate-v0`) or twelve 64x64
 * solver steps (`FloodOperateFine-v0`) are the slow cases, hence the long
 * timeout.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 120;

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }
  try {
    const r = await fetch(`${SERVICE}/gym/rollout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(110000),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "gym_error", detail: t.slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      {
        error: "service_unavailable",
        detail:
          "The hazard service is not reachable. Start it with: " +
          "python3 -m uvicorn service.app:app --port 8000",
        cause: String((e as Error).message).slice(0, 200),
      },
      { status: 503 },
    );
  }
}
