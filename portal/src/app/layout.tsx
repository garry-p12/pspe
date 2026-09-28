import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Nav } from "@/components/Nav";
import { RegionProvider } from "@/components/RegionContext";
import "./globals.css";

const sans = Inter({ variable: "--font-sans-stack", subsets: ["latin"], display: "swap" });
const mono = JetBrains_Mono({ variable: "--font-mono-stack", subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "Floodplain Planner — Richmond Valley",
  description:
    "Assess flood mitigation options against modelled inundation: what each option protects, what it costs, and where it would move risk rather than remove it.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} h-full antialiased`}>
      <body className="flex h-full flex-col overflow-hidden bg-bg text-ink">
        <RegionProvider initial="Richmond Valley · NSW">
          <Nav />
          <main className="min-h-0 flex-1">{children}</main>
        </RegionProvider>
      </body>
    </html>
  );
}
