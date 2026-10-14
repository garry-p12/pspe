import { NextResponse } from "next/server";

/**
 * What the spread model is worth, including where it is behind the field.
 *
 * Proxies the service's `/fire/skill`.
 */
const SERVICE = process.env.FLOOD_SERVICE ?? "http://127.0.0.1:8000";

export const maxDuration = 30;

export async function GET() {
  try {
    const r = await fetch(`${SERVICE}/fire/skill`, {
      signal: AbortSignal.timeout(20000),
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
