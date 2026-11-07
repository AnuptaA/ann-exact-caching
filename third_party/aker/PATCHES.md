# Local changes to vendored Aker

Each entry is one commit. Upstream: sjoon-oh/aker `release` e7639a5.

1. CMakeLists.txt: compile extern/xxHash/xxhash.c into libaker (upstream gets it
   from a submodule that is not vendored).
2. CMakeLists.txt: remove the ak_ycsb_gen target (needs the YCSB-C submodule).
