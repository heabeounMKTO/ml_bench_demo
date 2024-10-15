# ml_bench_demo
a small demo for benchmarking some ml frameworks  used in a webserver inference (with rust) <br>
this project depends on 

- tract
- tch-rs
- ort


# installing dependencies
you need to link C++ libraries libonnxruntime and libtorch, and then add it to your LD_LIBRARY_PATH (on linux)

```
export LD_LIBRARY_PATH=/path/to/libonnxruntime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```
refresh your shell by running `source ~/.bashrc` or `exec $SHELL`


# building the project

the webserver demo
```bash
cargo build --bin demo_webserver --release
```

the local demo
```bash
cargo build --bin local --release
```


