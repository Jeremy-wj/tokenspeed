# GLM AR+RMSNorm ROCm profiler image

This directory reconstructs the behavior of the qualified
`jeremwan/tokenspeed:rocm7.2.4-torch2.11-profiler` environment for the planned
GLM-5.2-FP8 operator campaign. It does not claim either historical image ID:
the original Dockerfile and complete package inventory were not retained, and
the historical tag identified different images on the GPT and GLM machines.

The replacement pins the published runner manifest, campaign-critical Python
packages, source revision metadata, scheduler build policy, and ROCm loader
repair. The runtime audit is the qualification identity.

## Build

From the TokenSpeed repository root:

```bash
docker/ar-rmsnorm-profiler/build-image.sh
```

The script refuses to overwrite an existing target tag and caps both pull and
build operations at 15 minutes. It never prunes, removes, stops, commits, or
retags shared Docker objects.

## Launch

```bash
docker/ar-rmsnorm-profiler/run-container.sh
```

The launcher refuses an existing container name and accepts only an image with
the expected owner and source-revision labels. It bind-mounts the checkout and
mounts `/data/models/glm-5.2-fp8` read-only.

Export the values printed by the launcher before using the generic benchmark
wrappers. The defaults are:

```text
TOKENSPEED_CONTAINER=jeremwan-ar-rmsnorm-profiler-mi355x-<commit>
CONTAINER_REPO_ROOT=/home/jeremwan/tokenspeed
MODEL_PATH=/data/models/glm-5.2-fp8
```

## Qualify

Run only while all eight GPUs are available:

```bash
docker/ar-rmsnorm-profiler/qualify.sh
```

The qualification script:

- records image, source, package, model, GPU, topology, and loader identity;
- verifies system ROCm 7.2.4 is used for HIP/HSA/RCCL/ROCTX/roctracer;
- runs bounded HIP event-query, Kineto, Proton, communication, graph, and
  transition probes;
- runs one representative M=42 process per campaign arm;
- validates the definitive campaign's 315-process dry-run without collecting
  the campaign.

Every command has an independent 15-minute ceiling. Artifacts are retained
under the Git-ignored GLM environment-qualification result tree. Failures are
evidence; do not delete or reinterpret them as campaign measurements.
