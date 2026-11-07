# Local changes to vendored Aker

Each entry is one commit. Upstream: sjoon-oh/aker `release` e7639a5.

1. CMakeLists.txt: compile extern/xxHash/xxhash.c into libaker (upstream gets it
   from a submodule that is not vendored).
2. CMakeLists.txt: remove the ak_ycsb_gen target (needs the YCSB-C submodule).
3. inc/ak_anns_cache_config.hh, inc/ak_anns_cache_c_wrapper.h, src/ak_anns_cache_c_wrapper.cc,
   src/ak_approx_filter.cc: the query filter's HNSW parameters (m, efSearch, efConstruction)
   were hard-coded as 4 / 8 / 16 (the paper says m 4, efConstruction 8, efSearch 16). They are
   now fields of the config (`filter`) and of the C struct, defaulting to the upstream values;
   the harness sets the paper's values. The file parser is unchanged (the driver sets the
   fields directly on the C struct after importing the config file).
   A non-positive value in the C struct keeps the default, so zero-initialized callers still work.
4. src/core/ak_anns_cache_maintenance.cc: `std::random_shuffle` (removed in C++17, rejected by
   libc++ on macOS) replaced by `std::shuffle`. It is only used by the invalidation stress helper,
   which the harness never calls.
5. inc/utils/ak_stack_alloc.hh: `StackFloatBuffer` called `alloca()` inside its own constructor, so
   the buffer was freed when the constructor returned and the converted query vector was read from
   freed stack memory. Observed here (clang, macOS): garbage query vectors, the approximate filter
   returned no candidate, and approximate hits never happened (only exact-id hits). The buffer now
   owns its storage (std::vector). Upstream's released numbers may depend on how their compiler laid
   out that stack; reported as a finding.
