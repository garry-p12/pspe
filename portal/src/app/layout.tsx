import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Nav } from "@/components/Nav";
import { RegionProvider } from "@/components/RegionContext";
import "./globals.css";

const sans = Inter({ variable: "--font-sans-stack", subsets: ["latin"], display: "swap" });
const mono = JetBrains_Mono({ variable: "--font-mono-stack", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "Floodplain Planner — Onion Creek, Austin",
  description:
    "Assess flood mitigation options against modelled inundation: what each option protects, what it costs, and where it would move risk rather than remove it.",
};

/**
 * The app is a card on a page, not a full-bleed application.
 *
 * This is the single change that most closes the gap to the design. Full bleed
 * puts the map's cut edges against the browser frame and makes the study box
 * read as the whole world; inset on a warm grey ground, with the card clipping
 * the map's corners, it reads as a document about a place. `overflow-hidden`
 * on the card is what rounds the map.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} h-full antialiased`}>
      <body className="h-full overflow-hidden bg-page p-3 text-ink md:p-4">
        <RegionProvider initial="Austin, Texas">
          <div className="flex h-full flex-col overflow-hidden rounded-[14px] bg-bg
                          shadow-[0_1px_3px_rgba(16,20,28,0.08),0_10px_30px_rgba(16,20,28,0.07)]">
            <Nav />
            <main className="min-h-0 flex-1">{children}</main>
          </div>
        </RegionProvider>
      </body>
    </html>
  );
}
