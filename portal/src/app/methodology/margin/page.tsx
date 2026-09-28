import type { Metadata } from "next";
import { MarginExplorer } from "@/components/MarginExplorer";

export const metadata: Metadata = {
  title: "Safety margin | PSPE",
  description:
    "When a calibrated conformal margin fits inside the gain an intervention buys, and why threshold amplification makes it hardest exactly where the intervention matters.",
};

export default function MarginPage() {
  return <MarginExplorer />;
}
