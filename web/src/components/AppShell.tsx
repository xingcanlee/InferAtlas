import { type ErrorComponentProps, Link, Outlet, useRouterState } from "@tanstack/react-router";
import {
  AlertTriangle,
  GitCompareArrows,
  LayoutGrid,
  Loader2,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { applyTheme, useUi } from "@/lib/store";
import { cn } from "@/lib/utils";

function Logo() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <rect
        x="1"
        y="1"
        width="7"
        height="7"
        rx="1.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <rect
        x="10"
        y="10"
        width="7"
        height="7"
        rx="1.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path d="M8 4.5h3.5a1 1 0 0 1 1 1V10" fill="none" stroke="var(--focus)" strokeWidth="1.5" />
    </svg>
  );
}

function NavLink({
  to,
  icon,
  children,
}: {
  to: "/" | "/diff";
  icon: React.ReactNode;
  children: string;
}) {
  const path = useRouterState({ select: (s) => s.location.pathname });
  const active = to === "/" ? path === "/" : path.startsWith(to);
  return (
    <Link
      to={to}
      className={cn(
        "inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[13px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground",
        active && "bg-accent text-foreground",
      )}
    >
      {icon}
      {children}
    </Link>
  );
}

export function AppShell() {
  const theme = useUi((s) => s.theme);
  const setTheme = useUi((s) => s.setTheme);
  const setSearchOpen = useUi((s) => s.setSearchOpen);
  const onModel = useRouterState({ select: (s) => s.location.pathname.startsWith("/m/") });
  useEffect(() => applyTheme(theme), [theme]);

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex h-full min-h-0 flex-col">
        <header className="flex h-11 shrink-0 items-center gap-3 border-b px-3">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <Logo />
            <span className="text-[14px]">InferAtlas</span>
          </Link>
          <span className="h-4 w-px bg-border" />
          <nav className="flex items-center gap-1">
            <NavLink to="/" icon={<LayoutGrid className="size-3.5" />}>
              Matrix
            </NavLink>
            <NavLink to="/diff" icon={<GitCompareArrows className="size-3.5" />}>
              Diff
            </NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-1.5">
            {onModel ? (
              <Button
                variant="outline"
                size="sm"
                className="h-7 w-52 justify-start gap-2 px-2 text-muted-foreground"
                onClick={() => setSearchOpen(true)}
                data-testid="open-search"
              >
                <Search className="size-3.5" />
                <span className="text-[12px]">Find node…</span>
                <Kbd className="ml-auto">/</Kbd>
              </Button>
            ) : null}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="size-7"
                  aria-label="Toggle theme"
                  onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                  data-testid="theme-toggle"
                >
                  {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{theme === "dark" ? "Light theme" : "Dark theme"}</TooltipContent>
            </Tooltip>
          </div>
        </header>
        <main className="min-h-0 flex-1">
          <Outlet />
        </main>
      </div>
    </TooltipProvider>
  );
}

export function LoadingState() {
  return (
    <div className="flex h-full items-center justify-center gap-2 text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      Loading snapshot…
    </div>
  );
}

export function EmptyState({
  title,
  children,
  icon,
}: {
  title: string;
  children?: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 p-6 text-center">
      <div className="text-muted-foreground">{icon ?? <AlertTriangle className="size-5" />}</div>
      <div className="font-medium">{title}</div>
      {children ? (
        <div className="max-w-md text-[12px] text-muted-foreground">{children}</div>
      ) : null}
    </div>
  );
}

export function ErrorState({ error }: ErrorComponentProps) {
  return (
    <EmptyState
      title="Could not load this page"
      icon={<AlertTriangle className="size-5 text-bad" />}
    >
      <p className="font-mono">{error instanceof Error ? error.message : String(error)}</p>
      <p className="mt-3">
        <Link to="/" className="underline underline-offset-2">
          Back to the matrix
        </Link>
      </p>
    </EmptyState>
  );
}

export function NotFoundState() {
  return (
    <EmptyState title="Nothing here">
      <Link to="/" className="underline underline-offset-2">
        Back to the matrix
      </Link>
    </EmptyState>
  );
}
