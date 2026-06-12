# Targeted Data Collection — production implementation (design:
# TARGETED_DATA_COLLECTION.md). Promoted from the PoC in targetted_dataset/poc.
#
# Modules:
#   registry    supported-task list (incremental enablement; unsupported -> skip)
#   sampler     stratified class-by-construction shift sampling (pure)
#   labels      three-layer label derivation (pure)
#   runtime     TargetedRuntime: per-episode perturbation state + triggers,
#               consulted by Bench_base_task hooks
#   hdf5_annot  append the label record to the episode HDF5
