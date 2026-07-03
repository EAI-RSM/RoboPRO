# thirdparty

Heavy dependencies used by this repo live under this subdirectory as **git submodules** (not copied source).

## Submodules

| Path | Upstream |
|------|-----------|
| `DepthAnything` | [ByteDance-Seed/Depth-Anything-V3](https://github.com/ByteDance-Seed/Depth-Anything-3) |

After clone, initialize them:

```bash
git submodule update --init --recursive
```

Or clone once with:

```bash
git clone --recursive <repo-url>
```

## Imports

- **Depth Anything V3**: `thirdparty.DepthAnything.src.depth_anything_3`
