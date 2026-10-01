import { NextResponse } from "next/server";

/**
 * A trip through a flood: origin, destination, and the reasons.
 *
 * Proxies the service's `/route`. Slower than the other endpoints and
 * unavoidably so: the domain is derived from the two ends, so terrain has to be
 * fetched for a box that contains both, the flood solved across it, and the
 * road network pulled from OpenStreetMap before anything can be routed. Two
 * minutes is normal for a 20 km trip.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 600;

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }
  try {
    const r = await fetch(`${SERVICE}/route`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(580_000),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "routing_error", detail: t.slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      {
        error: "service_unavailable",
        detail:
          "The flood service is not reachable. Start it with: " +
          "python3 -m uvicorn service.app:app --port 8000",
        cause: String((e as Error).message).slice(0, 200),
      },
      { status: 503 },
    );
  }
}
