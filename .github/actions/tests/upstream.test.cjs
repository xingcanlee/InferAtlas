const assert = require("node:assert/strict");
const { test } = require("node:test");
const { candidates, meetsFloor, pairKey, poll } = require("../upstream.cjs");

const asc = "a".repeat(40);
const vllm = "b".repeat(40);
const main = "c".repeat(40);
const verified = "d".repeat(40);
const generator = "e".repeat(40);
const previous = "f".repeat(40);
const core = { warning() {}, info() {} };
const context = {
  repo: { owner: "test", repo: "atlas" },
  payload: { repository: { default_branch: "main" } },
};

function fake() {
  const dispatched = [];
  const github = {
    rest: {
      repos: {
        listReleases() {},
        async getCommit({ repo, ref }) {
          const shas = {
            "vllm-ascend:v0.32.0": asc,
            "vllm-ascend:main": main,
            "vllm:v0.32.0": vllm,
            [`vllm:${verified}`]: verified,
            "atlas:main": generator,
            [`atlas:${generator}`]: generator,
            [`atlas:${previous}`]: previous,
          };
          assert.ok(shas[`${repo}:${ref}`], `${repo}:${ref}`);
          return {
            data: {
              sha: shas[`${repo}:${ref}`],
              commit: { tree: { sha: shas[`${repo}:${ref}`] } },
            },
          };
        },
        async getContent({ repo, ref, path }) {
          if (repo === "atlas") throw Object.assign(new Error("Not found"), { status: 404 });
          assert.equal(repo, "vllm-ascend");
          assert.equal(
            path,
            `.github/vllm-${ref === main ? "main-verified" : "release-tag"}.commit`,
          );
          return {
            data: {
              encoding: "base64",
              content: Buffer.from(ref === main ? verified : "v0.32.0\n").toString("base64"),
            },
          };
        },
      },
      git: {
        async getTree() {
          return { data: { tree: [{ path: "rules", sha: "1".repeat(40) }] } };
        },
        async getRef() {
          throw Object.assign(new Error("Not found"), { status: 404 });
        },
      },
      actions: {
        listWorkflowRuns() {},
        async createWorkflowDispatch(request) {
          dispatched.push(request);
        },
      },
    },
    async paginate(method, { repo }) {
      if (method === github.rest.actions.listWorkflowRuns) return [];
      return repo === "vllm"
        ? [{ tag_name: "v0.32.0" }, { tag_name: "v-unpaired" }, { tag_name: "draft", draft: true }]
        : [{ tag_name: "v0.32.0" }, { tag_name: "v0.9.0" }];
    },
  };
  return { github, dispatched };
}

test("candidates use only the declared release and verified-main pairs above the floor", async () => {
  const { github } = fake();
  const pairs = await candidates(github, core);
  assert.deepEqual(new Set(pairs.map((pair) => pair.vllm_commit)), new Set([vllm, verified]));
});

test("candidates put releases in numeric version order before main regardless of hashes", async () => {
  const { github } = fake();
  const tags = ["v0.100.0", "v0.9.0", "v0.10.0", "v0.32.0"];
  const commits = new Map(tags.map((tag, i) => [tag, String(i + 1).repeat(40)]));
  const getCommit = github.rest.repos.getCommit;
  github.rest.repos.getCommit = async (request) =>
    request.repo === "vllm-ascend" && commits.has(request.ref)
      ? { data: { sha: commits.get(request.ref) } }
      : getCommit(request);
  github.paginate = async () => tags.map((tag_name) => ({ tag_name }));

  const pairs = await candidates(github, core, { minimum_ascend_version: "0.9.0" });
  assert.deepEqual(
    pairs.map((pair) => pair.ascend_tag),
    ["v0.9.0", "v0.10.0", "v0.32.0", "v0.100.0", "main"],
  );
});

test("a release keeps priority when main resolves to the same commit pair", async () => {
  const { github } = fake();
  const getCommit = github.rest.repos.getCommit;
  github.rest.repos.getCommit = async (request) => {
    if (request.repo === "vllm-ascend" && request.ref === "main") return { data: { sha: asc } };
    if (request.repo === "vllm" && request.ref === vllm) return { data: { sha: vllm } };
    return getCommit(request);
  };
  const getContent = github.rest.repos.getContent;
  github.rest.repos.getContent = async (request) =>
    request.path.endsWith("main-verified.commit")
      ? { data: { encoding: "base64", content: Buffer.from(vllm).toString("base64") } }
      : getContent(request);
  const pairs = await candidates(github, core);
  assert.equal(pairs.length, 1);
  assert.equal(pairs[0].ascend_tag, "v0.32.0");
});

test("a poll dispatches at most one eligible pair", async () => {
  const { github, dispatched } = fake();
  await poll({ github, context, core });
  assert.equal(dispatched.length, 1);
  assert.equal(dispatched[0].inputs.vllm_commit, vllm);
  for (const request of dispatched) {
    assert.equal(request.ref, "main");
    assert.equal(request.workflow_id, "build.yml");
    assert.equal(request.inputs.publish, "true");
  }
});

test("handled pairs are skipped while unrecorded pairs remain eligible", async () => {
  const { github, dispatched } = fake();
  const pair = { ascend_commit: asc, vllm_commit: vllm };
  const read = github.rest.repos.getContent;
  github.rest.repos.getContent = async (request) =>
    request.repo === "atlas"
      ? {
          data: {
            content: Buffer.from(JSON.stringify({ pairs: { [pairKey(pair)]: {} } })).toString(
              "base64",
            ),
          },
        }
      : read(request);
  await poll({ github, context, core });
  assert.equal(dispatched.length, 1);
  assert.equal(dispatched[0].inputs.ascend_commit, main);
});

test("pair hashes match the Python canonical JSON protocol", () => {
  const { createHash } = require("node:crypto");
  assert.equal(
    pairKey({ ascend_commit: asc, vllm_commit: vllm }),
    createHash("sha256").update(`{"ascend":"${asc}","vllm":"${vllm}"}`).digest("hex"),
  );
});

test("API failures do not silently remove candidates", async () => {
  const { github } = fake();
  github.rest.repos.getContent = async () => {
    throw Object.assign(new Error("rate limit"), { status: 403 });
  };
  await assert.rejects(candidates(github, core), /rate limit/);
});

test("an existing state branch with missing state.json is fatal", async () => {
  const { github } = fake();
  github.rest.git.getRef = async () => ({ data: {} });
  await assert.rejects(poll({ github, context, core }), /state.json is missing/);
});

function ledger(github, state) {
  const read = github.rest.repos.getContent;
  github.rest.repos.getContent = async (request) =>
    request.repo === "atlas"
      ? { data: { content: Buffer.from(JSON.stringify(state)).toString("base64") } }
      : read(request);
}

const releasePair = { ascend_commit: asc, vllm_commit: vllm };
const mainPair = { ascend_commit: main, vllm_commit: verified };

test("hard failures are suppressed for their generator SHA and retried after a new SHA", async () => {
  const { github, dispatched } = fake();
  ledger(github, {
    pairs: {},
    failures: Object.fromEntries(
      [releasePair, mainPair].map((pair) => [pairKey(pair), { [generator]: true }]),
    ),
  });
  await poll({ github, context, core });
  assert.equal(dispatched.length, 0);
  const getCommit = github.rest.repos.getCommit;
  github.rest.repos.getCommit = async (request) =>
    request.repo === "atlas" && request.ref === "main"
      ? { data: { sha: previous } }
      : getCommit(request);
  await poll({ github, context, core });
  assert.equal(dispatched.length, 1);
});

for (const status of ["queued", "in_progress", "waiting", "pending", "requested"]) {
  test(`an existing ${status} build defers polling`, async () => {
    const { github, dispatched } = fake();
    const paginate = github.paginate;
    github.paginate = async (method, request) =>
      method === github.rest.actions.listWorkflowRuns
        ? request.status === status
          ? [{ id: 1 }]
          : []
        : paginate(method, request);
    await poll({ github, context, core });
    assert.equal(dispatched.length, 0);
  });
}

for (const changed of ["docs", "generator", "rules", "annotations"]) {
  test(`blocked models retry only for maintenance trees: ${changed}`, async () => {
    const { github, dispatched } = fake();
    ledger(github, {
      pairs: Object.fromEntries(
        [releasePair, mainPair].map((pair) => [
          pairKey(pair),
          {
            generator: previous,
            blocked_models: ["moe"],
          },
        ]),
      ),
    });
    github.rest.git.getTree = async ({ tree_sha }) => ({
      data: {
        tree: ["docs", "generator", "rules", "annotations"].map((path) => ({
          path,
          sha: path === changed ? tree_sha : "1".repeat(40),
        })),
      },
    });
    await poll({ github, context, core });
    assert.equal(dispatched.length, ["rules", "annotations"].includes(changed) ? 1 : 0);
  });
}

test("a failure at the new revision also suppresses maintenance retries", async () => {
  const { github, dispatched } = fake();
  ledger(github, {
    pairs: Object.fromEntries(
      [releasePair, mainPair].map((pair) => [
        pairKey(pair),
        {
          generator: previous,
          blocked_models: ["moe"],
        },
      ]),
    ),
    failures: Object.fromEntries(
      [releasePair, mainPair].map((pair) => [pairKey(pair), { [generator]: true }]),
    ),
  });
  github.rest.git.getTree = async () => {
    throw new Error("should not compare trees");
  };
  await poll({ github, context, core });
  assert.equal(dispatched.length, 0);
});

test("backfill floor compares numeric versions and excludes a prerelease of the floor", () => {
  assert.equal(meetsFloor("v0.9.9", "0.10.0"), false);
  assert.equal(meetsFloor("v0.10.0rc1", "0.10.0"), false);
  assert.equal(meetsFloor("v0.10.0", "0.10.0"), true);
  assert.equal(meetsFloor("v0.10.0.post1", "0.10.0"), true);
  assert.equal(meetsFloor("v0.11.0rc1", "0.10.0"), true);
  assert.equal(meetsFloor("main", "0.10.0"), false);
  assert.throws(() => meetsFloor("v1.0.0", "bad"), /major.minor.patch/);
});
