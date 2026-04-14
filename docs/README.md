# Documentation Workspace

This directory contains the source files, build configuration, and supporting assets for Simulac documentation.

## What lives here

- `internal/`
  Developer-facing documents such as architecture notes, conventions, and design decisions. Start here if you are contributing to the codebase or trying to understand how the project is structured.

- `publish/`
  Reserved for public-facing documentation content. The long-term plan is to integrate this content with the `tektonian-docs` repository and publish it under `tektonian.com/docs`. (Publishing flow is not implemented yet.)

## Useful starting points

- [internal/architecture.md](internal/architecture.md)
  Project structure, naming rules, and architecture notes for contributors

# For Writers

At Tektonian, we treat documentation as a core part of the product, not as an afterthought. We want every document related to Simulac to remain a [living document](https://en.wikipedia.org/wiki/Living_document) that evolves together with the codebase.

In practice, it is not realistic to apply [literate programming](https://en.wikipedia.org/wiki/Literate_programming) across the entire project. Instead, we plan to keep documentation accurate through a combination of `doctest`-based validation and AI-assisted review.

Scripts can automate part of this work, but they cannot cover every documentation change. For cases where future updates should be checked by AI, please leave an inline comment starts with `<!--ai` in the document like the format below.

```md
<!--ai
When AI review automation is added, update the phrase "planned" in the paragraph above.
-->
