import {
  createHashHistory,
  createRootRoute,
  createRoute,
  createRouter,
  type RouterHistory,
  redirect,
} from "@tanstack/react-router";
import { AppShell, ErrorState, LoadingState, NotFoundState } from "@/components/AppShell";
import { inline, loadCellBundle, loadHardware, loadIndex, singleTarget } from "@/lib/data";
import { SINGLE } from "@/lib/mode";
import { diffSearchSchema, homeSearchSchema, modelSearchSchema } from "@/lib/search";
import { DiffPage } from "@/pages/DiffPage";
import { HomePage } from "@/pages/HomePage";
import { ModelPage } from "@/pages/ModelPage";

const rootRoute = createRootRoute({
  component: AppShell,
  notFoundComponent: NotFoundState,
});

export const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  validateSearch: homeSearchSchema,
  beforeLoad: () => {
    // A single-file export opens on its embedded model, not a one-cell matrix (backlog #2).
    const target = SINGLE ? singleTarget(inline()) : undefined;
    if (target) throw redirect({ to: "/m/$version/$device/$model", params: target, replace: true });
  },
  loader: () => loadIndex(),
  component: HomePage,
  pendingComponent: LoadingState,
  errorComponent: ErrorState,
});

export const modelRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/m/$version/$device/$model",
  validateSearch: modelSearchSchema,
  loader: async ({ params }) => {
    const [data, hardware] = await Promise.all([
      loadCellBundle(params.version, params.model, params.device),
      loadHardware(),
    ]);
    return { ...data, hardware };
  },
  component: ModelPage,
  pendingComponent: LoadingState,
  errorComponent: ErrorState,
});

export const diffRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/diff",
  validateSearch: diffSearchSchema,
  loader: () => loadIndex(),
  component: DiffPage,
  pendingComponent: LoadingState,
  errorComponent: ErrorState,
});

const routeTree = rootRoute.addChildren([homeRoute, modelRoute, diffRoute]);

export function createAppRouter(history: RouterHistory = createHashHistory()) {
  return createRouter({
    routeTree,
    history,
    defaultPreload: "intent",
    defaultPendingMs: 150,
    scrollRestoration: false,
  });
}

export const router = createAppRouter();

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
