// Use vllm-ascend's declared compatibility; never form a Cartesian product of tags.
const { createHash } = require("node:crypto");

const owner = "vllm-project";
const shaPattern = /^[0-9a-f]{40}$/;

function pairKey(pair) {
  return createHash("sha256")
    .update(JSON.stringify({ ascend: pair.ascend_commit, vllm: pair.vllm_commit }))
    .digest("hex");
}

async function commit(github, repo, ref) {
  const { data } = await github.rest.repos.getCommit({ owner, repo, ref });
  if (!shaPattern.test(data.sha)) throw new Error(`Invalid ${repo} SHA`);
  return data.sha;
}

async function pairing(github, ascendCommit, main) {
  const path = `.github/vllm-${main ? "main-verified" : "release-tag"}.commit`;
  const { data } = await github.rest.repos.getContent({
    owner,
    repo: "vllm-ascend",
    path,
    ref: ascendCommit,
  });
  if (data.encoding !== "base64") throw new Error(`Unexpected encoding for ${path}`);
  const ref = Buffer.from(data.content, "base64").toString("utf8").trim();
  if (!ref || /\s/.test(ref)) throw new Error(`Invalid pairing file ${path}`);
  if (main && !shaPattern.test(ref)) throw new Error("main-verified must be a full SHA");
  return ref;
}

async function candidates(github, core) {
  const releases = {};
  for (const repo of ["vllm", "vllm-ascend"]) {
    releases[repo] = await github.paginate(github.rest.repos.listReleases, {
      owner,
      repo,
      per_page: 100,
    });
  }
  const vllmTags = new Set(releases.vllm.filter((r) => !r.draft).map((r) => r.tag_name));
  const tags = [
    ...new Set(releases["vllm-ascend"].filter((r) => !r.draft).map((r) => r.tag_name)),
  ].sort();
  const pairs = [];
  for (const ascendTag of [...tags, "main"]) {
    const ascendCommit = await commit(github, "vllm-ascend", ascendTag);
    let ref;
    try {
      ref = await pairing(github, ascendCommit, ascendTag === "main");
    } catch (error) {
      // Older releases predate pairing files; other API errors must be retried.
      if (error.status !== 404 || ascendTag === "main") throw error;
      core.warning(`No pairing declaration in vllm-ascend ${ascendTag}; skipping`);
      continue;
    }
    if (ascendTag !== "main" && !vllmTags.has(ref)) {
      core.warning(`vllm-ascend ${ascendTag} declares unreleased vLLM ${ref}; waiting`);
      continue;
    }
    pairs.push({
      ascend_commit: ascendCommit,
      ascend_tag: ascendTag,
      vllm_commit: await commit(github, "vllm", ref),
      vllm_tag: ascendTag === "main" ? "main-verified" : ref,
    });
  }
  const unique = new Map(pairs.map((pair) => [pairKey(pair), pair]));
  return [...unique.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([, pair]) => pair);
}

async function poll({ github, context, core }) {
  const { owner: repoOwner, repo } = context.repo;
  let state = { pairs: {} };
  try {
    const { data } = await github.rest.repos.getContent({
      owner: repoOwner,
      repo,
      path: "state.json",
      ref: "state",
    });
    state = JSON.parse(Buffer.from(data.content, "base64").toString("utf8"));
    if (!state.pairs || typeof state.pairs !== "object" || Array.isArray(state.pairs)) {
      throw new Error("Malformed state.json");
    }
  } catch (error) {
    if (error.status !== 404) throw error;
    // Only a genuinely absent branch is a bootstrap; a damaged state branch is fatal.
    try {
      await github.rest.git.getRef({ owner: repoOwner, repo, ref: "heads/state" });
      throw new Error("state branch exists but state.json is missing");
    } catch (branchError) {
      if (branchError.status !== 404) throw branchError;
    }
  }
  for (const pair of await candidates(github, core)) {
    if (Object.hasOwn(state.pairs, pairKey(pair))) continue;
    await github.rest.actions.createWorkflowDispatch({
      owner: repoOwner,
      repo,
      workflow_id: "build.yml",
      ref: context.payload.repository.default_branch,
      inputs: { ...pair, publish: "true" },
    });
    core.info(`Dispatched ${pair.vllm_tag} + ${pair.ascend_tag} (${pairKey(pair)})`);
  }
}

module.exports = { candidates, pairKey, poll };
