# CI, automation, and Pages

The workflows live in `.github/workflows/`; their CPU-only helpers and tests live
in `.github/actions/`. Actions are pinned to commit SHAs. GitHub-hosted jobs use
Python 3.12 through uv, Node 22, and the pnpm version in `package.json`.

## Workflows

| File / Actions name | Triggers | Behavior and requirements |
| --- | --- | --- |
| `ci.yml` / CI | Pull requests; pushes except `state` and `gh-pages` | Installs locked workspaces; validates workflows with actionlint 1.7.12 and ShellCheck 0.11.0; runs automation tests, Ruff/Biome, generator tests excluding `extract`, workspace tests, schema drift, and golden checks. No secrets, upstream checkout, weights, or hardware required. |
| `poll.yml` / Poll upstream releases | Every six hours at minute 17 UTC; manual dispatch | Default branch only. Reads both upstream release lists, vllm-ascend's declared vLLM pairing, and the `state` ledger. Dispatches at most one eligible build. The release floor is in `.github/actions/poll-config.json`. Scheduled execution requires `INFERATLAS_AUTOMATION=true`. |
| `nightly.yml` / Nightly main compatibility | Daily at 02:43 UTC; manual dispatch | Default branch only. Resolves both upstream `main` SHAs and calls `build.yml` with `publish=false` to detect compatibility drift. Scheduled execution requires `INFERATLAS_AUTOMATION=true`. |
| `build.yml` / Build snapshots | Manual dispatch; reusable workflow call | Takes full `vllm_commit` and `ascend_commit` SHAs, optional display tags, and `publish` (default `true`). Expands catalog model/device cells, installs CPU dependencies, extracts on meta, verifies, and collects artifacts. Default-branch triage can dispatch maintenance. Publication additionally requires `INFERATLAS_PAGES_DEPLOY=true`. |
| `draft-pr.yml` / Draft maintenance PR | Manual dispatch, including dispatch from a trusted build | Default branch only. Takes a `run_id`, validates its default-branch build/nightly provenance, downloads triage, and runs `skills/inferatlas-maintain/SKILL.md` through Claude. Allows only rule/annotation YAML changes, validates them, then creates a draft PR. Requires both maintenance secrets below. Human review is required before merging. |
| `calibrate.yml` / Calibrate Ascend A2 | Manual dispatch only | Default branch only. Requires a self-hosted runner with labels `self-hosted`, `linux`, `ascend-a2`, the `calibration` environment, a provisioned NPU runtime, and existing local model weights. Inputs: catalog model ID, absolute `model_path`, and TP (1/2/4/8). Starts the runtime and uploads startup evidence. Never runs as part of CI. |

The five non-CI workflows also accept a push that changes their own workflow
file, excluding `state` and `gh-pages`. **Every pipeline job skips on push.**
These discovery events register previously unused workflows in GitHub's Actions
list without executing their pipelines. Manual workflows must exist on the
default branch before dispatch is available; merge this branch before using
the new gates on `main`.

Initially, all six workflow files existed on `main`, but the workflows API
listed only CI and returned 404 for the unused workflows. Actionlint found no
syntax errors. A workflow-file push with the guards above registered all five;
GitHub accepted their definitions and reported all their jobs as skipped.
Poll and nightly were then disabled through the workflows API because `main`
still has their old, unguarded definitions until C1 is merged. They remain
registered with state `disabled_manually`; the other four are active.

## Repository configuration

Configure repository variables under Settings → Secrets and variables → Actions
→ Variables. Values are case-sensitive strings; unset is disabled.

| Variable / secret | Default | Purpose |
| --- | --- | --- |
| Variable `INFERATLAS_AUTOMATION` | Unset / off | Set to `true` to allow scheduled poll and nightly jobs. Manual dispatch bypasses this schedule-only gate, but still requires the default branch. |
| Variable `INFERATLAS_PAGES_DEPLOY` | Unset / off | Set to `true` only after switching Pages to GitHub Actions. Enables the default-branch build publication job, including immutable HTML releases, deployment, and successful-pair ledger updates. `publish=false` still prevents publication. |
| Secret `ANTHROPIC_API_KEY` | Required only for maintenance | Claude API credential for drafting rule and annotation changes. |
| Secret `MAINTENANCE_TOKEN` | Required only for maintenance | A repository-scoped GitHub App or PAT token with Contents, Pull requests, and Issues write access. Used to push the maintenance branch, manage its label, and open its draft PR. App tokens need an external refresh mechanism. |

CI uses read-only `GITHUB_TOKEN` permissions. Poll and build collection request
Actions write access for workflow dispatch. Publication requests Contents write
for releases and the `state` branch, plus Pages write and ID-token write for
deployment. The build failure recorder can update only the failure ledger on
`state`; it never deploys. Repository/organization policy must permit the pinned
actions and the requested permissions. No extra deployment token is needed.

Configure the `github-pages` environment to allow the default branch when
enabling workflow deployment. Protect the `calibration` environment with
maintainer approval and restrict its deployment branches to the default branch
before provisioning an NPU runner. No NPU runner is needed for other workflows.

Builds retain per-cell diagnostics and bundles for 14 days. Calibration logs are
retained for 90 days. Publishing uses the separate `state` branch to retain
verified snapshot JSON and handled/failed pairs. Published HTML release assets
are immutable: reruns verify their bytes rather than replacing them. A triaged
device blocks publication of that model across devices, while other verified
models remain eligible.

To enable automation after merging and checking the configuration:

1. Merge C1 so the default branch has the schedule and publication gates.
   Re-enable the two registered scheduled workflows (this does not dispatch a run):

   ```bash
   gh workflow enable poll.yml --repo xingcanlee/InferAtlas
   gh workflow enable nightly.yml --repo xingcanlee/InferAtlas
   ```

2. Provision the maintenance secrets if automatic draft PRs are wanted. A build
   can dispatch maintenance when triage exists, including a nightly build.
3. Complete the Pages migration below before enabling automatic publication.
4. Set `INFERATLAS_AUTOMATION=true`. The next scheduled poll/nightly can run.

To stop scheduled work, unset `INFERATLAS_AUTOMATION` or set it to `false`.
Already queued/running jobs are not cancelled by changing the variable. Without
the Pages gate, polls can still dispatch extraction builds, but successful pairs
are not recorded as published and can be selected again on later polls.

## Initial Pages publication: `gh-pages`

The initial publication uses the team's locally verified release. A maintainer
publishes the **contents** of the site directory at the root of `gh-pages`, with
offline exports under `offline/`, preserving relative links and filenames:

```text
gh-pages/
  .nojekyll
  index.html
  assets/       # if present in the verified release
  data/         # if present in the verified release
  offline/
    ...html
```

In Settings → Pages → Build and deployment, select **Deploy from a branch**,
branch **gh-pages**, folder **/(root)**, and save. Include `.nojekyll` so GitHub
serves the generated static files without Jekyll processing. The project URL is
`https://xingcanlee.github.io/InferAtlas/`; offline exports are beneath
`https://xingcanlee.github.io/InferAtlas/offline/`.

Leave `INFERATLAS_PAGES_DEPLOY` unset/false while this source is active. The build
workflow cannot enter its publication job in this state, and no workflow writes
to `gh-pages` or changes Pages settings. CI ignores pushes to this static branch.
The discovery push hooks also exclude it. The branch publication itself and
repository Pages settings are managed by the maintainer, not by C1.

## Later migration to GitHub Actions

1. Merge the workflow changes to the default branch. Keep the automation and
   Pages variables disabled during the migration.
2. Verify a build's catalog, upstream pins, extraction results, and static output
   before enabling publication. `build.yml` publishes `dist/site`; its current
   renderer also generates immutable exports in `dist/single/<snapshot>/` for
   release assets. This is a different output assembly from the initial local
   release's `/offline/` directory; verify that route and its links explicitly
   before replacing the initial site.
3. In Settings → Pages → Build and deployment → Source, select **GitHub Actions**.
   Do not add GitHub's suggested starter workflow: `build.yml` already uploads
   `dist/site` using `actions/upload-pages-artifact` and deploys with
   `actions/deploy-pages`.
4. Check the `github-pages` environment's branch rules and required reviewers.
   Set `INFERATLAS_PAGES_DEPLOY=true` only after the source is GitHub Actions.
5. Manually dispatch **Build snapshots** on the default branch with verified
   full upstream SHAs and `publish=true`. Confirm the deployment URL and assets.
   Then enable scheduled automation if desired.

The old `gh-pages` branch can remain as a rollback source. To return to branch
publication, first disable `INFERATLAS_PAGES_DEPLOY`, let active publication jobs
finish (or cancel them), then select `gh-pages` / root again in Pages settings.

GitHub documents [publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
and [workflow trigger behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

## Local checks and registry selection

GitHub-hosted jobs use PyPI and npm's public registry. The committed npm mirror
override was removed; local mirror use is selected through environment variables:

```bash
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export npm_config_registry=https://registry.npmmirror.com
uv sync --project generator --locked
pnpm install --frozen-lockfile
actionlint -config-file .github/actions/actionlint.yaml .github/workflows/*.yml
uv run --project generator ruff check --config generator/pyproject.toml .github/actions
uv run --project generator ruff format --check --config generator/pyproject.toml .github/actions
uv run --project generator pytest .github/actions/tests
node --test .github/actions/tests/*.test.cjs
```

The uv lock records canonical PyPI artifact URLs; `UV_INDEX_URL` does not redirect
those locked downloads. `generator/pyproject.toml` also declares the PyPI index.
Keep the committed lock on PyPI; do not regenerate it against a mirror. For a
mirror-only bootstrap, use `uv pip install --python <dedicated-venv>/bin/python`
with `UV_INDEX_URL` rather than rewriting the project lock. Frozen uv commands
can install/run without resolving a new lock. The root `justfile` still provides
local mirror defaults; hosted jobs explicitly override its Python index with
`just --set UV_INDEX_URL https://pypi.org/simple/ ...`.

Use actionlint 1.7.12 and ShellCheck 0.11.0 for parity with CI. CI downloads
actionlint's Linux amd64 release and checks its pinned SHA-256 before installing
it. The pinned install-action did not recognize that actionlint release and
incorrectly fell back to Cargo; actionlint is a Go program, not a Rust crate.
The pinned installer remains in use for supported tools (`just`, ShellCheck).

Inspect hosted results without dispatching automation:

```bash
gh api repos/xingcanlee/InferAtlas/actions/workflows
gh run list --repo xingcanlee/InferAtlas --branch task/C1 --workflow ci.yml
gh run watch RUN_ID --repo xingcanlee/InferAtlas --exit-status
gh run view RUN_ID --repo xingcanlee/InferAtlas --log-failed
```
