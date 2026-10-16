import { NextResponse } from "next/server";

/**
 * The four environments, with the ladder of policies measured on each.
 *
 * Proxies the service's `/gym/environments`. Everything prose-shaped in the
 * response comes from `pspe/envs/catalog.py`, which `scripts/score_gym.py
 * --ladder --check` tests, so this page has no second copy of any claim.
 *
 * `probe=1` asks the service to construct each environment and report its real
 * observation and action spaces. That needs the egress table and the fire
 * checkpoint on disk and takes a few seconds, so the page asks for it once on
 * load rather than on every render.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 120;

export async function GET(req: Request) {
  const probe = new URL(req.url).searchParams.get("probe") === "1";
  try {
    const r = await fetch(`${SERVICE}/gym/environments?probe=${probe}`, {
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
