import { NextResponse } from "next/server";

/**
 * The held-out fire records, ranked by how much burns the next day.
 *
 * Proxies the service's `/fire/records`.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 120;

export async function GET() {
  try {
    const r = await fetch(`${SERVICE}/fire/records`, {
      signal: AbortSignal.timeout(110000),
    });
    if (!r.ok) {
      const t = await r.text();
      return NextResponse.json(
        { error: "fire_error", detail: t.slice(0, 400) },
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
