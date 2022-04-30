#r "nuget: DiffSharp-lite,1.0.7"
(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiffSharp/diffsharp.github.io/blob/master/extensions.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=extensions.ipynb)&emsp;
[![Script](img/badge-script.svg)](extensions.fsx)&emsp;
[![Script](img/badge-notebook.svg)](extensions.ipynb)

# Extending DiffSharp

DiffSharp provides most of the essential operations found in tensor libraries such as [NumPy](https://numpy.org/), [PyTorch](https://pytorch.org/), and [TensorFlow](https://www.tensorflow.org/). All differentiable operations support the forward, reverse, and nested differentiation modes.

When implementing new operations, you should prefer to implement these as compositions of existing DiffSharp [Tensor](https://diffsharp.github.io/reference/diffsharp-tensor.html) operations, which would give you differentiability out of the box.

In the rare cases where you need to extend DiffSharp with a completely new differentiable operation that cannot be implemented as a composition of existing operations, you can use the provided extension API.

## Simple elementwise functions

If the function you would like to implement is a simple elementwise function, you can use the [UnaryOpElementwise](https://diffsharp.github.io/reference/diffsharp-unaryopelementwise.html) or [BinaryOpElementwise](https://diffsharp.github.io/reference/diffsharp-binaryopelementwise.html) types to define your function and its derivatives. The forward, reverse, and nested differentiation rules for the function are automatically generated by the type. The documentation of these two types detail how they should be instantiated.

Let's see several examples.

$f(a) = \mathrm{sin}(a)$, with derivative $\frac{\partial f(a)}{\partial a} = \mathrm{cos}(a) \;$.

*)
open DiffSharp

type Tensor with
    member a.sin() = 
        Tensor.Op
            { new UnaryOpElementwise("sin") with 
                member _.fRaw(a) = a.SinT()
                member _.dfda(a,f) = a.cos()
            }
            (a)
(**
$f(a) = \mathrm{log}(a)$, with derivative $\frac{\partial f(a)}{\partial a} = 1/a \;$.

*)
type Tensor with
    member a.log() =
        Tensor.Op
            { new UnaryOpElementwise("log") with
                member _.fRaw(a) = a.LogT()
                member _.dfda(a,f) = 1/a
            }
            (a)
(**
$f(a, b) = ab$, with derivatives $\frac{\partial f(a, b)}{\partial a} = b$, $\frac{\partial f(a, b)}{\partial b} = a \;$.

*)
type Tensor with
    member a.mul(b) =
        Tensor.Op
            { new BinaryOpElementwise("mul") with
                member _.fRaw(a,b) = a.MulTT(b)
                member _.dfda(a,b,f) = b
                member _.dfdb(a,b,f) = a
            }
            (a,b)
(**
$f(a, b) = a^b$, with derivatives $\frac{\partial f(a, b)}{\partial a} = b a^{b-1}$, $\frac{\partial f(a, b)}{\partial b} = a^b \mathrm{log}(a) \;$. Note the use of the argument `f` in the derivative definitions that makes use of the pre-computed value of $f(a, b) = a^b$ that is available to the derivative implementation.

*)
type Tensor with
    member a.pow(b) =
        Tensor.Op
            { new BinaryOpElementwise("pow") with
                member _.fRaw(a,b) = a.PowTT(b)
                member _.dfda(a,b,f) = b * f / a  // equivalent to b * a.pow(b-1)
                member _.dfdb(a,b,f) = f * a.log()  // equivalent to a.pow(b) * a.log()
            }
            (a,b)
(**
## General functions

For more complicated functions, you can use the most general way of defining functions using the [UnaryOp](https://diffsharp.github.io/reference/diffsharp-unaryop.html) or [BinaryOp](https://diffsharp.github.io/reference/diffsharp-binaryop.html) types, which allow you to define the full forward and reverse mode differentiation rules. The documentation of these two types detail how they should be instantiated.

Let's see several examples.

$f(A) = A^{\intercal}$, with the forward derivative propagation rule $\frac{\partial f(A)}{\partial X} = \frac{\partial A}{\partial X} \frac{\partial f(A)}{\partial A} = (\frac{\partial A}{\partial X})^{\intercal}$ and the reverse derivative propagation rule $\frac{\partial Y}{\partial A} = \frac{\partial Y}{\partial f(A)} \frac{\partial f(A)}{\partial A} = (\frac{\partial Y}{\partial f(A)})^{\intercal} \;$.

*)
type Tensor with
    member a.transpose() =
        Tensor.Op
            { new UnaryOp("transpose") with
                member _.fRaw(a) = a.TransposeT2()
                member _.ad_dfda(a,ad,f) = ad.transpose()
                member _.fd_dfda(a,f,fd) = fd.transpose()
            }
            (a)
(**
$f(A, B) = AB$, with the forward derivative propagation rule $\frac{\partial(A, B)}{\partial X} = \frac{\partial A}{\partial X} \frac{\partial f(A, B)}{\partial A} + \frac{\partial B}{\partial X} \frac{\partial f(A, B)}{\partial B} = \frac{\partial A}{\partial X} B + A \frac{\partial B}{\partial X}$ and the reverse propagation rule $\frac{\partial Y}{\partial A} = \frac{\partial Y}{\partial f(A, B)} \frac{\partial f(A, B)}{\partial A} = \frac{\partial Y}{\partial f(A, B)} B^{\intercal}$, $\frac{\partial Y}{\partial B} = \frac{\partial Y}{\partial f(A, B)} \frac{\partial f(A, B)}{B} = A^{\intercal} \frac{\partial Y}{\partial f(A, B)} \;$.

*)
type Tensor with
    member a.matmul(b) =
        Tensor.Op
            { new BinaryOp("matmul") with
                member _.fRaw(a,b) = a.MatMulTT(b)
                member _.ad_dfda(a,ad,b,f) = ad.matmul(b)
                member _.bd_dfdb(a,b,bd,f) = a.matmul(bd)
                member _.fd_dfda(a,b,f,fd) = fd.matmul(b.transpose())
                member _.fd_dfdb(a,b,f,fd) = a.transpose().matmul(fd)
            }
            (a,b)
