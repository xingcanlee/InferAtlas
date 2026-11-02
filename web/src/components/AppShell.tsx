import { type ErrorComponentProps, Link, Outlet, useRouterState } from "@tanstack/react-router";
import {
  AlertTriangle,
  GitCompareArrows,
  LayoutGrid,
  LayoutList,
  Loader2,
  Menu,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import { useEffect } from "react";
import { HomeLink } from "@/components/HomeLink";
import { Logo } from "@/components/Logo";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Kbd } from "@/components/ui/kbd";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { DataError } from "@/lib/data";
import { HUB_HREF, SINGLE } from "@/lib/mode";
import { useIsPhone } from "@/lib/responsive";
import { type ModelSearch, modelTab } from "@/lib/search";
import { cellStatusText } from "@/lib/status";
import { applyTheme, useUi } from "@/lib/store";
import { cn } from "@/lib/utils";

const navClass =
  "inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[13px] max-lg:h-10 max-lg:px-3 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground";

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
    <Link to={to} className={cn(navClass, active && "bg-accent text-foreground")}>
      {icon}
      {children}
    </Link>
  );
}

/** Phones: the nav collapses into one menu button (40px target). */
function NavMenu() {
  const path = useRouterState({ select: (s) => s.location.pathname });
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon-lg" aria-label="Navigation" data-testid="nav-menu">
          <Menu className="size-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-44">
        {SINGLE ? (
          <DropdownMenuItem asChild>
            <a href={HUB_HREF} data-testid="nav-all-models">
              <LayoutList /> All models
            </a>
          </DropdownMenuItem>
        ) : (
          <>
            <DropdownMenuItem asChild className={cn(path === "/" && "bg-accent")}>
              <Link to="/">
                <LayoutGrid /> Matrix
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem asChild className={cn(path.startsWith("/diff") && "bg-accent")}>
              <Link to="/diff">
                <GitCompareArrows /> Diff
              </Link>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export function AppShell() {
  const theme = useUi((s) => s.theme);
  const setTheme = useUi((s) => s.setTheme);
  const setSearchOpen = useUi((s) => s.setSearchOpen);
  const phone = useIsPhone();
  // Node search belongs to the graph workspace; hide it on the Architecture tab.
  const onModel = useRouterState({
    select: (s) =>
      s.location.pathname.startsWith("/m/") &&
      modelTab(s.location.search as ModelSearch) === "graph",
  });
  useEffect(() => applyTheme(theme), [theme]);

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex h-full min-h-0 flex-col">
        <header className="flex h-11 shrink-0 items-center gap-3 border-b px-3 pt-[env(safe-area-inset-top)] max-md:safe-px max-md:h-[calc(3rem+env(safe-area-inset-top))] max-md:gap-1.5">
          {phone ? <NavMenu /> : null}
          <HomeLink className="flex items-center gap-2 font-semibold tracking-tight">
            <Logo />
            <span className="text-[14px]">InferAtlas</span>
          </HomeLink>
          {phone ? null : (
            <>
              <span className="h-4 w-px bg-border" />
              <nav className="flex items-center gap-1">
                {SINGLE ? (
                  // One embedded model: the overview of every export is the sibling hub page.
                  <a href={HUB_HREF} className={navClass} data-testid="nav-all-models">
                    <LayoutList className="size-3.5" />
                    All models
                  </a>
                ) : (
                  <>
                    <NavLink to="/" icon={<LayoutGrid className="size-3.5" />}>
                      Matrix
                    </NavLink>
                    <NavLink to="/diff" icon={<GitCompareArrows className="size-3.5" />}>
                      Diff
                    </NavLink>
                  </>
                )}
              </nav>
            </>
          )}
          <div className="ml-auto flex items-center gap-1.5">
            {onModel ? (
              <Button
                variant="outline"
                size="sm"
                className="h-7 w-52 justify-start gap-2 px-2 text-muted-foreground max-md:size-10 max-md:justify-center max-md:px-0"
                onClick={() => setSearchOpen(true)}
                aria-label="Find node"
                data-testid="open-search"
              >
                <Search className="size-3.5" />
                <span className="text-[12px] max-md:hidden">Find node…</span>
                <Kbd className="ml-auto max-md:hidden">/</Kbd>
              </Button>
            ) : null}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="size-7 max-lg:size-10"
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

const backLabel = SINGLE ? "All models" : "Back to the matrix";

export function ErrorState({ error }: ErrorComponentProps) {
  const cell = error instanceof DataError ? error.cell : undefined;
  if (error instanceof DataError && cell) {
    // A deep link to a cell without a numeric bundle: say what its status really is.
    const text = cellStatusText(cell);
    return (
      <EmptyState
        title={`${text.title}: ${cell.model}`}
        icon={<StatusBadge status={cell.status} count={cell.triage_count ?? undefined} />}
      >
        <p data-testid="cell-unavailable">{error.message}</p>
        <p className="mt-3">
          <HomeLink className="underline underline-offset-2">{backLabel}</HomeLink>
        </p>
      </EmptyState>
    );
  }
  return (
    <EmptyState
      title="Could not load this page"
      icon={<AlertTriangle className="size-5 text-bad" />}
    >
      <p className="font-mono">{error instanceof Error ? error.message : String(error)}</p>
      <p className="mt-3">
        <HomeLink className="underline underline-offset-2">{backLabel}</HomeLink>
      </p>
    </EmptyState>
  );
}

export function NotFoundState() {
  return (
    <EmptyState title="Nothing here">
      <HomeLink className="underline underline-offset-2">{backLabel}</HomeLink>
    </EmptyState>
  );
}
