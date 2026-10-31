const assert = require("node:assert/strict");
const { test } = require("node:test");
const { candidates, pairKey, poll } = require("../upstream.cjs");

const asc = "a".repeat(40);
const vllm = "b".repeat(40);
const main = "c".repeat(40);
const verified = "d".repeat(40);
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
            "vllm-ascend:v1": asc,
            "vllm-ascend:main": main,
            "vllm:v1": vllm,
            [`vllm:${verified}`]: verified,
          };
          assert.ok(shas[`${repo}:${ref}`], `${repo}:${ref}`);
          return { data: { sha: shas[`${repo}:${ref}`] } };
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
              content: Buffer.from(ref === main ? verified : "v1\n").toString("base64"),
            },
          };
        },
      },
      git: {
        async getRef() {
          throw Object.assign(new Error("Not found"), { status: 404 });
        },
      },
      actions: {
        async createWorkflowDispatch(request) {
          dispatched.push(request);
        },
      },
    },
    async paginate(_method, { repo }) {
      return repo === "vllm"
        ? [{ tag_name: "v1" }, { tag_name: "v-unpaired" }, { tag_name: "draft", draft: true }]
        : [{ tag_name: "v1" }];
    },
  };
  return { github, dispatched };
}

test("poll uses only the declared release and verified-main pairs", async () => {
  const { github, dispatched } = fake();
  await poll({ github, context, core });
  assert.equal(dispatched.length, 2);
  assert.deepEqual(
    new Set(dispatched.map((call) => call.inputs.vllm_commit)),
    new Set([vllm, verified]),
  );
  for (const request of dispatched) {
    assert.equal(request.ref, "main");
    assert.equal(request.workflow_id, "build.yml");
    assert.equal(request.inputs.publish, "true");
  }
});

test("handled pairs are skipped, failed builds without state are retried", async () => {
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
