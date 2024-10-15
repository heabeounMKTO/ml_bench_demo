# ml_bench_demo
a small demo for benchmarking some ml frameworks  used in a webserver inference (with rust) <br>
this project depends on 

- tract
- tch-rs
- ort


# installing dependencies
you need to link C++ libraries libonnxruntime and libtorch, so download them and then add it to your LD_LIBRARY_PATH (on linux)
you can find libonnxruntime [here](https://github.com/microsoft/onnxruntime/releases) we use version 1.18.1, <br>
and libtorch [here](https://pytorch.org/) we use version 2.4.0


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

# using the project as a library 
todo 


