# Proxy perception data

Phase 3 works out of the box with `data.source=synthetic`, which renders PDE
testbed fields through a lossy sensor model (blur, shot noise, vignetting). No
downloads, no accounts, nothing to pay for.

`data.source=real` uses free real-world imagery instead. Expected layout:

```
data/perception_proxy/
├── images/<id>.png      RGB patch (any size; resized to model.image_size)
├── fields/<id>.npy      (C, grid, grid) target field for the same patch
└── captions.json        optional {"<id>": "weak caption"}  (synthesised if absent)
```

`<id>` must match between `images/` and `fields/`. The field is whatever
scalar you want the model to recover - an NDVI/NDWI band, a thermal band, a
turbidity index. That choice defines the inverse problem.

## Free sources

| Source | Resolution | Access |
|---|---|---|
| Sentinel-2 (Copernicus) | 10 m | [Copernicus Browser](https://browser.dataspace.copernicus.eu/) - free account, no card |
| Landsat 8/9 (USGS) | 30 m | [EarthExplorer](https://earthexplorer.usgs.gov/) - free account |
| NAIP aerial (USA) | 0.6-1 m | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) or AWS open data |

None require a paid tier. Commercial high-resolution imagery is the only thing
in this pipeline that costs money, and it is not needed: at the 64x64 patch
scale the testbeds use, Sentinel-2 is more than adequate.

## Preparing patches

Any tiling script works. The pieces that matter:

1. crop to square patches (64x64 pixels is the default testbed scale);
2. save RGB as PNG under `images/`;
3. compute the target index band for the *same* pixels, save as
   `(1, 64, 64)` float32 `.npy` under `fields/`;
4. keep a held-out geographic region rather than a random split - random splits
   over adjacent tiles leak, and the reported reconstruction error stops
   meaning anything.

`captions.json` is optional; without it, `describe_field` synthesises the weak
caption from field statistics, which is the same supervision the synthetic path
uses.
