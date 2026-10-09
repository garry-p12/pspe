import { NextResponse } from "next/server";

/**
 * Buildings in a block, with the solved water against each floor.
 *
 * Proxies the service's `/buildings`. Slower than it looks: the catchment is
 * solved at its own scale first and the street is downscaled from it, because
 * a street-sized box has no upstream in it and rain falling on one simply
 * drains out of the sides. Two minutes is normal.
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
    const r = await fetch(`${SERVICE}/buildings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(580_000),
    });
    if (!r.ok) {
      return NextResponse.json(
        { error: "street_error", detail: (await r.text()).slice(0, 400) },
        { status: 502 },
      );
    }
    return NextResponse.json(await r.json());
  } catch (e) {
    return NextResponse.json(
      {
        error: "service_unavailable",
        detail: "The flood service is not reachable.",
        cause: String((e as Error).message).slice(0, 200),
      },
      { status: 503 },
    );
  }
}
