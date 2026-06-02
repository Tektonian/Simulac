# ADR 0003: Hosted Benchmark Protocol

## Status

Pending.

## Context

Simulac uses http + Websocket for running benchmark services.
First decision was GRPC, but our service depends on Cloudflare which makes hard to use grpc protocol.

The protocol must also handle authentication, container cold starts, binary
observation payloads, and vectorized benchmark stepping.

## Decision

Pending. (Hope using gprc in the future)

## Consequences

## Open Work

1. Stablize API spec
2. Use cloud server, instead of serverless
3. Migrate containers to kubernetes
4. Change into GRPC