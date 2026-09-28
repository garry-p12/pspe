import { NextResponse } from "next/server";

/**
 * The road network over a box, from OpenStreetMap.
 *
 * Proxies the service's Perceive stage. This is the half of a digital twin that
 * a simulator does not have: the model says what it believes is flooded, and
 * this says what an instrument actually recorded on its last usable overpass,
 * so the two can be put side by side and disagree.
 *
 * Slower than /analyse is fast but faster than it is slow: the scenes are
 * fetched from a public archive as windowed reads, so a minute is normal and
 * two is not alarming.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 300;

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "bad_request" }, { status: 400 });
  }
  try {
    const r = await fetch(`${SERVICE}/roads`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(280_000),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "roads_error", detail: t.slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      {
        error: "service_unavailable",
        detail:
          "The roads service is not reachable. Start it with: " +
          "python3 -m uvicorn service.app:app --port 8000",
        cause: String((e as Error).message).slice(0, 200),
      },
      { status: 503 },
    );
  }
}
