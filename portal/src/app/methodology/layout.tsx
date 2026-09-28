/** The research material keeps a scrolling page; the planner does not. */
export default function MethodologyLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <div className="h-full overflow-y-auto">{children}</div>;
}
