#r "nuget: DiffSharp-lite,1.0.7"
(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiffSharp/diffsharp.github.io/blob/master/tensors.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=tensors.ipynb)&emsp;
[![Script](img/badge-script.svg)](tensors.fsx)&emsp;
[![Script](img/badge-notebook.svg)](tensors.ipynb)

* The [dsharp](https://diffsharp.github.io/reference/diffsharp-dsharp.html) API
  

* The [Tensor](https://diffsharp.github.io/reference/diffsharp-tensor.html) type
  

Saving tensors as image and loading images as tensors

## Converting between Tensors and arrays

System.Array and F# arrays

*)
open DiffSharp

// Tensor
let t1 = dsharp.tensor [ 0.0 .. 0.2 .. 1.0 ]

// System.Array
let a1 = t1.toArray()

// []<float32>
let a1b = t1.toArray() :?> float32[]

// Tensor
let t2 = dsharp.randn([3;3;3])

// [,,]<float32>
let a2 = t2.toArray() :?> float32[,,]

