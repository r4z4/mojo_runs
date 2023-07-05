%%python
def matmul_python(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

%%python
import numpy as np
from timeit import timeit

class Matrix:
    def __init__(self, value, rows, cols):
        self.value = value
        self.rows = rows
        self.cols = cols
        
    def __getitem__(self, idxs):
        return self.value[idxs[0]][idxs[1]]
    
    def __setitem__(self, idxs, value):
        self.value[idxs[0]][idxs[1]] = value

def benchmark_matmul_python(M, N, K):
    A = Matrix(list(np.random.rand(M, K)), M, K)
    B = Matrix(list(np.random.rand(K, N)), K, N)
    C = Matrix(list(np.zeros((M, N))), M, N)
    secs = timeit(lambda: matmul_python(C, A, B), number=2)/2
    gflops = ((2*M*N*K)/secs) / 1e9
    print(gflops, "GFLOP/s")
    return gflops

python_gflops = benchmark_matmul_python(128, 128, 128).to_float64()

#|code-fold: true
#|code-summary: "Import utilities and define `Matrix` (click to show/hide)"

from Benchmark import Benchmark
from DType import DType
from Intrinsics import strided_load
from List import VariadicList
from Math import div_ceil, min
from Memory import memset_zero
from Object import object, Attr
from Pointer import DTypePointer
from Random import rand, random_float64
from TargetInfo import dtype_sizeof, dtype_simd_width

# This exactly the same Python implementation, 
# but is infact Mojo code!
def matmul_untyped(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

fn matrix_getitem(self: object, i: object) raises -> object:
    return self.value[i]


fn matrix_setitem(self: object, i: object, value: object) raises -> object:
    self.value[i] = value
    return None


fn matrix_append(self: object, value: object) raises -> object:
    self.value.append(value)
    return None


fn matrix_init(rows: Int, cols: Int) raises -> object:
    let value = object([])
    return object(
        Attr("value", value), Attr("__getitem__", matrix_getitem), Attr("__setitem__", matrix_setitem), 
        Attr("rows", rows), Attr("cols", cols), Attr("append", matrix_append),
    )

def benchmark_matmul_untyped(M: Int, N: Int, K: Int, python_gflops: Float64):
    C = matrix_init(M, N)
    A = matrix_init(M, K)
    B = matrix_init(K, N)
    for i in range(M):
        c_row = object([])
        b_row = object([])
        a_row = object([])
        for j in range(N):
            c_row.append(0.0)
            b_row.append(random_float64(-5, 5))
            a_row.append(random_float64(-5, 5))
        C.append(c_row)
        B.append(b_row)
        A.append(a_row)

    @parameter
    fn test_fn():
        try:
            _ = matmul_untyped(C, A, B)
        except:
            pass

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = (A, B, C)
    let gflops = ((2*M*N*K)/secs) / 1e9
    let speedup : Float64 = gflops / python_gflops
    print(gflops, "GFLOP/s, a", speedup.value, "x speedup over Python")

benchmark_matmul_untyped(128, 128, 128, python_gflops)

struct Matrix:
    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        rand(self.data, rows*cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    @always_inline
    fn load[nelts:Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    @always_inline
    fn store[nelts:Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        self.data.simd_store[nelts](y * self.cols + x, val)

# Note that C, A, and B have types.
fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

@always_inline
def benchmark[func : fn(Matrix, Matrix, Matrix) -> None]
    (M : Int, N : Int, K : Int, python_gflops: Float64):
    var C = Matrix(M, N)
    C.zero()
    var A = Matrix(M, K)
    var B = Matrix(K, N)

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    # Prevent matrices from being destroyed before we finished benchmarking them.
    _ = A.data
    _ = B.data
    _ = C.data

    let gflops = ((2*M*N*K)/secs) / 1e9
    let speedup : Float64 = gflops / python_gflops
    print(gflops, "GFLOP/s, a", speedup.value, "x speedup over Python")


benchmark[matmul_naive](512, 512, 512, python_gflops)

# Mojo has SIMD vector types, we can vectorize the Matmul code as follows.
alias nelts = dtype_simd_width[DType.float32]() # The SIMD vector width.
fn matmul_vectorized_0(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for nv in range(0, C.cols, nelts):
                C.store[nelts](m,nv, C.load[nelts](m,nv) + A[m,k] * B.load[nelts](k,nv))
        
            # Handle remaining elements with scalars.
            for n in range(nelts*(C.cols//nelts), C.cols):
                C[m,n] += A[m,k] * B[k,n]

benchmark[matmul_vectorized_0](512, 512, 512, python_gflops)

# Simplify the code by using the builtin vectorize function
from Functional import vectorize
fn matmul_vectorized_1(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[nelts, dot](C.cols)

benchmark[matmul_vectorized_1](512, 512, 512, python_gflops)

# Parallelize the code by using the builtin parallelize function
from Functional import parallelize
fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[nelts, dot](C.cols)
        
    parallelize[calc_row](C.rows)

benchmark[matmul_parallelized](512, 512, 512, python_gflops)

from Functional import Static2DTileUnitFunc as Tile2DFunc

# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

# Use the above tile function to perform tiled matmul.
fn matmul_tiled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n + x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))
                vectorize[nelts, dot](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](C.rows)

benchmark[matmul_tiled_parallelized](512, 512, 512, python_gflops)

# Unroll the vectorized loop by a constant factor.
from Functional import vectorize_unroll
fn matmul_tiled_unrolled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n+x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                vectorize_unroll[nelts, tile_x//nelts, dot](tile_x)

        alias tile_size = 4
        tile[calc_tile, nelts*tile_size, tile_size](A.cols, C.cols)
      
    parallelize[calc_row](C.rows)

benchmark[matmul_tiled_unrolled_parallelized](512, 512, 512, python_gflops)

from Autotune import autotune, search
from Time import now
from Pointer import Pointer

alias matmul_fn_sig_type = fn(Matrix, Matrix, Matrix) -> None

# Autotune the tile size used in the matmul.
@adaptive
fn matmul_autotune_impl(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):
                @parameter
                fn dot[nelts : Int,](n : Int):
                    C.store[nelts](m,n+x, C.load[nelts](m,n+x) + A[m,k] * B.load[nelts](k,n+x))
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        # Instead of hardcoding to tile_size = 4, search for the fastest 
        # tile size by evaluting this function as tile size varies.
        alias tile_size = autotune(1, 2, 4, 8, 16, 32)
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)
      
    parallelize[calc_row](C.rows)

fn matmul_evaluator(funcs: Pointer[matmul_fn_sig_type], size: Int) -> Int:
    print("matmul_evaluator, number of candidates: ", size)

    let eval_begin: Int = now()

    # This size is picked at random, in real code we could use a real size
    # distribution here.
    let M = 512
    let N = 512
    let K = 512
    print("Optimizing for size:", M, "x", N, "x", K)

    var best_idx: Int = -1
    var best_time: Int = -1

    alias eval_iterations = 10
    alias eval_samples = 10

    var C = Matrix(M, N)
    var A = Matrix(M, K)
    var B = Matrix(K, N)
    let Cptr = Pointer[Matrix].address_of(C).address
    let Aptr = Pointer[Matrix].address_of(A).address
    let Bptr = Pointer[Matrix].address_of(B).address

    # Find the function that's the fastest on the size we're optimizing for
    for f_idx in range(size):
        let func = funcs.load(f_idx)

        @always_inline
        @parameter
        fn wrapper():
            func(C, A, B)
        let cur_time = Benchmark(1, 100_000, 500_000_000, 1000_000_000).run[wrapper]()

        if best_idx < 0:
            best_idx = f_idx
            best_time = cur_time
        if best_time > cur_time:
            best_idx = f_idx
            best_time = cur_time

    let eval_end: Int = now()
    # Prevent matrices from being destroyed before we finished benchmarking them.
    _ = A.data
    _ = B.data
    _ = C.data
    print("Time spent in matmul_evaluator, ms:", (eval_end - eval_begin) // 1000000)
    print("Best candidate idx:", best_idx)
    return best_idx

fn matmul_autotune(C: Matrix, A: Matrix, B: Matrix):
    alias best_impl: matmul_fn_sig_type
    search[
        matmul_fn_sig_type,
        VariadicList(matmul_autotune_impl.__adaptive_set),
        matmul_evaluator -> best_impl
    ]()
    # Run the best candidate
    return best_impl(C, A, B)

benchmark[matmul_autotune](512, 512, 512, python_gflops)
