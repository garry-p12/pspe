import { NextResponse } from "next/server";

/**
 * An elevation section across the floodplain, with the solved water on it.
 *
 * Proxies the service's `/section`. It derives the same domain the trip view
 * does and shares its solve cache, so a section drawn over ground already
 * solved for a trip comes back without paying for the flood twice. A cold
 * 7 km line at 80 m cells measured 28 s.
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
    const r = await fetch(`${SERVICE}/section`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(580_000),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "section_error", detail: t.slice(0, 400) },
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
