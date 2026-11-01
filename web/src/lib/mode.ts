/** Build flavour, fixed at build time (`vite build --mode single|hub`). */
export const SINGLE = import.meta.env.MODE === "single";

/** In a release `dist/`, the hub sits beside every single-file export as `index.html`. */
export const HUB_HREF = "./index.html";
