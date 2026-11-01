import { Link } from "@tanstack/react-router";
import { HUB_HREF, SINGLE } from "@/lib/mode";

/**
 * "Back to the overview": the in-app matrix on the site, the sibling hub page (`./index.html`)
 * in a single-file export, whose own matrix would only hold one cell.
 */
export function HomeLink({
  className,
  children,
}: {
  className?: string;
  children?: React.ReactNode;
}) {
  if (SINGLE) {
    return (
      <a href={HUB_HREF} className={className}>
        {children ?? "All models"}
      </a>
    );
  }
  return (
    <Link to="/" className={className}>
      {children ?? "Matrix"}
    </Link>
  );
}
