import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Theme = "light" | "dark";

interface UiState {
  /** Ephemeral UI only; anything shareable lives in the URL. */
  theme: Theme;
  setTheme: (t: Theme) => void;
  searchOpen: boolean;
  setSearchOpen: (open: boolean) => void;
  hoverNode: string | undefined;
  setHoverNode: (id: string | undefined) => void;
}

const systemTheme = (): Theme =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";

export const useUi = create<UiState>()(
  persist(
    (set) => ({
      theme: systemTheme(),
      setTheme: (theme) => set({ theme }),
      searchOpen: false,
      setSearchOpen: (searchOpen) => set({ searchOpen }),
      hoverNode: undefined,
      setHoverNode: (hoverNode) => set({ hoverNode }),
    }),
    { name: "inferatlas-ui", partialize: (s) => ({ theme: s.theme }) },
  ),
);

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.style.colorScheme = theme;
}
