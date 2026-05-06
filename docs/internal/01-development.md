# Setup

Simulac uses `uv` for Python dependencies and `npm` for repository hooks.

So before you start, install [`Node.js`](https://nodejs.org) first.

```bash
uv sync --group dev --all-extras
npm install
```

# Background

This project follows widely adopted software development conventions. (a.k.a. De-Facto)

Before you get started, review the references below:
- [Semantic Versioning](https://semver.org/)
- [Commitlint](https://commitlint.js.org/guides/local-setup.html)
- [Changelog](https://keepachangelog.com/en/1.1.0/)
- [Branch naming strategy](https://conventional-branch.github.io/)
- [RFC 2119: MUST, SHOULD rules](https://datatracker.ietf.org/doc/html/rfc2119)

## Version control

We follow semantic versioning rule.
First of all, to understand version updating rules, it is RECOMMENDED to see the overall structure of [`Simulac`](./architecture.md).

Major and minor version updates depend on whether a change affects the user-surface API, which is exposed through the `/simulac/lib` directory.

- If a release includes incompatible changes to the user-facing API, increment the major version (`1.x.x` -> `2.x.x`).
- If a release includes user-facing API changes that remain backward compatible, increment the minor version (`x.1.x` -> `x.2.x`).
- If a release does not change the user-facing API, increment the patch version (`x.x.1` -> `x.x.2`).

## Commit Message Rules

We follow conventional commit rule to improve development experience and for better productivity.

To force and automate checking commit rules, we get help from [husky](https://typicode.github.io/husky/), [commitlint](https://commitlint.js.org/guides/local-setup.html), and [release-please](https://github.com/googleapis/release-please).


A commit hook, included in `/.husky` directory, enforces Conventional Commits.

Use this format:

```text
<type>(<scope>): <subject>
```
Please follow these rules:

- `<type>` must be lowercase.
- `<scope>` is REQUIRED.
- `<subject>` SHOULD be short, clear, and, most of all,concrete and detailed
- Past tense SHOULD not be used

If you cannot describe concrete and detailed `<subject>`, your commit likely contains too many changes.

### Allowed `<type>` values

| Type | When to use (Use the examples below as contexts, it is not MUST follow) |
| --- | --- |
| `chore` | - Maintenance work that does not fit other categories <br> - When changes don't effect to logic <br> - When there are no changes in `/simulac`|
| `ci` | - Changes to CI workflows or automation <br> - When `/.github` or `/.husky` changed |
| `docs` | - Documentation updates <br> - When `/docs` changed |
| `feat` | A new feature |
| `fix` | A bug fix |
| `refact` | Refactoring that does not add a feature or fix a bug |
| `revert` | Reverting a previous commit |
| `test` | Adding or updating tests |

### Allowed `<scope>` values

| Scope | When to use (It is MUST follow rule)|
| --- | --- |
| `project` | Repository-wide or project-level changes |
| `sdk` | - SDK and runtime service layers <br> - When `/simulac/sdk` changed |
| `lib` | - User-facing library modules <br> - When `/simulac/lib` changed |
| `test` | Test-related code |
| `base` | - Shared base utilities and infrastructure - When `/simulac/base` changed |
| `server` | Server-side modules |
| `cli` | - Command-line interface code <br> - When `/simulac/cli` changed |

If a change clearly spans multiple areas, you may use comma-separated scopes such as `feat(sdk,cli): foo is bar`.

### Good commit message examples

Best commit messages give useful contexts and make the change understandable and predictable before reviewer open the diff. 

- `feat(cli): update cache directory location and improve CLI login/logout messages`
  This is good because a reader can tell both what changed and where the change landed.

- `feat(base,sdk): apply the runtime log level from environment variables`
  This is a strong multi-scope commit because it describes one coherent feature that spans both areas.

- `feat(lib): align gym-style benchmark client with the updated remote protocol`
  This explains the reason for the change and the affected surface area in one line.

- `fix(project): add missed requests package`
  This is short, concrete, and immediately explains why the commit exists.

- `refact(lib): refactor step and reset function of BenchmarkVecEnvironment class`
  The subject is specific enough that a reviewer can predict the refactor before opening the diff.

- `test(sdk): add regression test for runner reconnect after timeout`
  Good test commits say what behavior is protected, not just that tests were added.

- `docs(project): document version bump rules for user-surface changes`
  Good documentation commits still name the exact policy or topic that changed.

### Avoid commit messages like these

Avoid messages that are vague, use unsupported types or scopes, or describe code edits without explaining their effect.

- `First commit - wrote small code and set dev environment`
  This does not follow the required conventional commit format and tells future readers almost nothing useful.

- `todo(sdk): changed Environment to have step -> need seperation`
  `todo` is not an allowed type, and the subject reads like a private note instead of a durable history entry.

- `refactor(sdk): changed id naming convention`
  `refactor` is not an allowed type in this repository. Use `refact` instead. Past tense should also be avoided.


- `feat(error): defined base error class`
  `error` is not an allowed scope, so the message does not match this repository's commit rules.

- `feat(lib): add libero style fields -> Will be change later`
  This is too informal, grammatically unclear, and includes temporary commentary that does not belong in commit history.

- `chore(sdk): seperate interface class`
  Even if the format is close, the subject is too vague. It does not explain what was extracted, renamed, or reorganized.

- `test(sdk): add tests`
  This gives no information about which behavior is covered or why the tests were needed.

## Branch Naming

First of all, branch naming is not MUST-follow rule.
However, [following conventions always recommended](https://nvd.nist.gov/vuln/detail/cve-2014-1266).

A pre-push hook checks branch names.

Use one of the following formats:

- `main`
- `master`
- `develop`
- `<type>/<description>`
- `<type>/<issue>-<description>`

Examples:

- `feat/login-form`
- `bugfix/env-reset`
- `release/v1.2.0`

# Testing & Validation

There are two test cases divided by pytest markers.

1. Unit test
2. Integeration test

Integration tests require external services and credentials, so run them only when your change touches that flow.

## Run test

Use the following commands depending on whether you want to include integration tests:
```bash
uv run pytest -q -s -m "integration" # Run integration tests only
uv run pytest -q -s -m "not integration" # Run all tests except integration tests
```

## Before Opening a Pull Request
Before opening a pull request, run the checks that apply to your change.

```bash
uv run pytest -q -m "not integration"
uv run ruff check ./simulac
uv run pyright ./simulac
```
>We know the codebase currently contains a number of warnings and errors.
>Because the project is still evolving, we ask contributors to resolve as many existing issues as possible before submitting new code.
