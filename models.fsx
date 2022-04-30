(**
Test

*)
open DiffSharp

dsharp.config(backend=Backend.Reference)

let a = dsharp.tensor([1,2,3])
printfn "%A" a(* output: 
val a: Tensor = tensor([1, 2, 3],dtype=Int32)
val it: unit = ()*)

