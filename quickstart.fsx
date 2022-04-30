#r "nuget: DiffSharp-lite,1.0.7"
#r "nuget: SixLabors.ImageSharp,1.0.1"
(**
[![Binder](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiffSharp/diffsharp.github.io/blob/master/quickstart.ipynb)&emsp;
[![Binder](img/badge-binder.svg)](https://mybinder.org/v2/gh/diffsharp/diffsharp.github.io/master?filepath=quickstart.ipynb)&emsp;
[![Script](img/badge-script.svg)](quickstart.fsx)&emsp;
[![Script](img/badge-notebook.svg)](quickstart.ipynb)

# Quickstart

Here we cover some key tasks involved in a typical machine learning pipeline and how these can be implemented with DiffSharp. Note that a significant part of DiffSharp's design has been influenced by [PyTorch](https://pytorch.org/) and you would feel mostly at home if you have familiarity with PyTorch.

## Datasets and Data Loaders

DiffSharp provides the [Dataset](https://diffsharp.github.io/reference/diffsharp-data-dataset.html) type that represents a data source and the [DataLoader](https://diffsharp.github.io/reference/diffsharp-data-dataloader.html) type that handles the loading of data from datasets and iterating over [minibatches](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Iterative_method) of data.

See the [DiffSharp.Data](/reference/diffsharp-data.html) namespace for the full API reference.

### Datasets

DiffSharp has ready-to-use types that cover main datasets typically used in machine learning, such as [MNIST](https://diffsharp.github.io/reference/diffsharp-data-mnist.html), [CIFAR10](https://diffsharp.github.io/reference/diffsharp-data-cifar10.html), [CIFAR100](https://diffsharp.github.io/reference/diffsharp-data-cifar100.html), and also more generic dataset types such as [TensorDataset](https://diffsharp.github.io/reference/diffsharp-data-tensordataset.html) or [ImageDataset](https://diffsharp.github.io/reference/diffsharp-data-imagedataset.html).

The following loads the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and shows one image entry and the corresponding label.

*)
open DiffSharp
open DiffSharp.Data

// First ten images in MNIST training set
let dataset = MNIST("../data", train=true, transform=id, n=10)

// Inspect a single image and label
let data, label = dataset[7]

// Save image to file
data.saveImage("test.png")(* output: 
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcEAYAAAAinQPXAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAB/0lEQVR4nO3XO2sVQRiH8UeJhRaClhYSEWzUxk5QY+EFROwEwbtioShJJSaFWCipAkLAK/oZFEEQQQWxEcGAaS1EC7FU0HghR3hgGMiy7M7urDGH/JqFPfsO+9/ZM+/OAOr16FMD9LnFgAvdYsCmNgiWCXYIbghmVT3OQ8EhwS/Vv49sATcKTggOCpYK1igG66l63AOCW4IRwVdV12cLOC7YJ7I7JrgneKXqumwBn6oY8IvgvmCJijO4VTAksskW8KbggeL534LPKq9fKZhWfKWDMO4b1b+vbAH/CD4qvX6vYJWKv38S/FT9cee9TYTV8YxguYrXXVb6+P884GHBqGC9YjuZa0rxVU+VLeCg4Khgl4rXbVN5mwjL/yXBY8EPpd9X64CbFRvyWjUf76Xgjmgt2wyG5T8cy4TGX/Yls1+x3YQZbKp1wHeCnYIjgieCGZXXnxZcENllm8EPgmuqX3dFCyBgU6H/dSU5YFjO9wieKX2VOyW4LjpTO+B2wZhgt2Cdqr9gVisuHhOCFYrXhQfVtC3MVTvgpGCT4vmLgm8qrw8PZIuKffCF4jftc9Fa6//gWaXXhV3GI8GwqlfdVLUDnhScFxxXdd17wXfFRn5Xsc10pXbAt4JzgteCq4q7gLCtCfvD8IVTtV3qSvIrGrYrtxWP/6t574Nd6/uAfwGuockofAHj/gAAAABJRU5ErkJggg==" style="width: 64px; height: auto"/>*)
// Inspect data as ASCII and show label
printfn "Data: %A\nLabel: %A" (data.toImageString()) label(* output: 
Data: "                            
                            
                            
                            
                            
           ~-}@#####Z       
         -j*W########J'     
         O############i     
         [##Mxxxxo####i     
          ::^    'W##Z      
                 |&##f      
                (o###'      
              (q%###d.      
         "uaaa####8}:       
        _m########O         
        _*####@####?        
         "v<____f##?        
                `##?        
                |##?        
       ?.      1&##?        
     iQ#:    `)8##&!        
     p##txxxxb###o\         
     p#########MC.          
     +J#####wdt_            
       }B#Z}^               
                            
                            
                            
"
Label: tensor(3,dtype=Int32)*)
(**
### Data Loaders

A data loader handles tasks such as constructing minibatches from an underlying dataset on-the-fly, shuffling the data, and moving the data tensors between devices. In the example below we show a single batch of six MNIST images and their corresponding classification labels.

*)
let loader = DataLoader(dataset, shuffle=true, batchSize=6)
let batch, labels = loader.batch()

printfn "%A\nLabels: %A" (batch.toImageString()) labels(* output: 
"                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                           <J#J+                                    
             ```uzO:Y@%u                  +8###8                          u#@!      
        ";)CX#####*Z#&h!                 <W##8M#>                        1&$#!      
       +8########$)\\>~                .i*###b|##v                       u$$#!      
       `m#####k0%&                     U######1o#Y                      lB$p".      
        /C[##d' -C                    <8##ar#W_/@X                     iW$$)        
         ' C#(                       +8##0'?v,  #&+                    C##o         
           j#a                      ~Y#Mp|      ##Y                   ,#$Bl         
           'a#I                     O#8I,:      ##h                  "d#$u          
            ;&*J[                  >##!         ##h                  }$#Q`          
             \8##c^                k#a          @#h                 /8$a^           
              _o##L:              ?%#]          ##t                "###J            
               `)##o              |#M^         z#o'                L$$$~            
                 $#$!             |#*         n#*I                +m$$Z             
              _nQ##d              |#f       +Y#Z                  B$$h'             
            ~tW###$0              |#*      r8#U                   #$$(              
          ^rm####b/               |#$t+:|O*#*Y>                  J@##"              
        ^lq####k\                 |###Ww###hn                   +W#%j.              
      `Xm####h/.                  :k#####Mf                     !$#m                
    >ZW####&x'                     ^n###j~                      !$#m                
    z###qzx`                                                    ^a#m                
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                                                                                    
                    lM~                                                '^{v         
    !\              c0~                                              ;Lp###t        
    uU              Cp~                 >tp##r|t>                  ~L&##*p#M~       
    mU             :#U                 |M##op###X                `L8###w"~##i       
    mU             Q#u                >&#a! '0##r                1####w: ~##i       
   _%U             k#>               1##Q'  )##*,                0##mY"  /##i       
   c#U            ^M#:              x##t'   w##/                 :n>^    {##i       
   J#c            U#w`             u#%O.  ./%#n                          J##i       
   J#l         '|O$#(             `M#O   ;b##X'                      ]vvvb#h        
   J#|   _+rfL&&B0&#~             ^##"^ck&##$/                    .<0##@##W;        
   L#8ddd##$8kf(: M$              `M#####WW#M                    <W#&WX&##Mc        
    cOOOOO1>     }#m               >B#wz-^a#f                   /##ui `p####|       
                 X#z                     !@#[                  z##0` ,b#%nZ##Ql++   
                 X#>                     I##,                 z#&[` <k#w! 'IU&##*   
                 X#>                      ##,                }#&(`?X&#u:     (00~   
                 X@)                     I##,                M#%dw###u              
                 X#1                     [##,                d####Or;               
                 X#C                     _@#,                ')fv^                  
                 X@C                      w#>                                       
                 1#C                      1#o-                                      
                                          'Q#X'                                     
                                           't#-                                     
                                                                                    
"
Labels: tensor([5., 0., 1., 4., 9., 2.])*)
(**
In practice a data loader is typically used to iterate over all minibatches in a given dataset in order to feed each minibatch through a machine learning model. One full iteration over the dataset would be called an "epoch". Typically you would perform multiple such epochs of iterations during the training of a model.

*)
for epoch = 1 to 10 do
    for i, data, labels in loader.epoch() do
        printfn "Epoch %A, minibatch %A" epoch (i+1)
        // Process the minibatch
        // ...
(**
## Models

Many machine learning models are differentiable functions whose parameters can be tuned via [gradient-based optimization](https://en.wikipedia.org/wiki/Gradient_descent), finding an optimum for an objective function that quantifies the fit of the model to a given set of data. These models are typically built as compositions non-linear functions and ready-to-use building blocks such as linear, recurrent, and convolutional layers.

DiffSharp provides the most commonly used model building blocks including convolutions, transposed convolutions, batch normalization, dropout, recurrent and other architectures.

See the [DiffSharp.Model](/reference/diffsharp-model.html) namespace for the full API reference.

### Constructing models, PyTorch style

If you have experience with [PyTorch](https://pytorch.org/), you would find the following way of model definition familiar. Let's look at an example of a [generative adversarial network (GAN)](https://arxiv.org/abs/1406.2661) architecture.

*)
open DiffSharp.Model
open DiffSharp.Compose

// PyTorch style

// Define a model class inheriting the base
type Generator(nz: int) =
    inherit Model()
    let fc1 = Linear(nz, 256)
    let fc2 = Linear(256, 512)
    let fc3 = Linear(512, 1024)
    let fc4 = Linear(1024, 28*28)
    do base.addModel(fc1, fc2, fc3, fc4)
    override self.forward(x) =
        x
        |> dsharp.view([-1;nz])
        |> fc1.forward
        |> dsharp.leakyRelu(0.2)
        |> fc2.forward
        |> dsharp.leakyRelu(0.2)
        |> fc3.forward
        |> dsharp.leakyRelu(0.2)
        |> fc4.forward
        |> dsharp.tanh

// Define a model class inheriting the base
type Discriminator(nz:int) =
    inherit Model()
    let fc1 = Linear(28*28, 1024)
    let fc2 = Linear(1024, 512)
    let fc3 = Linear(512, 256)
    let fc4 = Linear(256, 1)
    do base.addModel(fc1, fc2, fc3, fc4)
    override self.forward(x) =
        x
        |> dsharp.view([-1;28*28])
        |> fc1.forward
        |> dsharp.leakyRelu(0.2)
        |> dsharp.dropout(0.3)
        |> fc2.forward
        |> dsharp.leakyRelu(0.2)
        |> dsharp.dropout(0.3)
        |> fc3.forward
        |> dsharp.leakyRelu(0.2)
        |> dsharp.dropout(0.3)
        |> fc4.forward
        |> dsharp.sigmoid

// Instantiate the defined classes
let nz = 128
let gen = Generator(nz)
let dis = Discriminator(nz)

print gen
print dis(* output: 
Model(Linear(128, 256), Linear(256, 512), Linear(512, 1024), Linear(1024, 784))
Model(Linear(784, 1024), Linear(1024, 512), Linear(512, 256), Linear(256, 1))*)
(**
### Constructing models, DiffSharp style

A key advantage of DiffSharp lies in the [functional programming](https://en.wikipedia.org/wiki/Functional_programming) paradigm enabled by the F# language, where functions are first-class citizens, many algorithms can be constructed by applying and composing functions, and differentiation operations can be expressed as composable [higher-order functions](https://en.wikipedia.org/wiki/Higher-order_function). This allows very succinct (and beautiful) machine learning code to be expressed as a powerful combination of [lambda calculus](https://en.wikipedia.org/wiki/Lambda_calculus) and [differential calculus](https://en.wikipedia.org/wiki/Differential_calculus).

For example, the following constructs the same GAN architecture (that we constructed in PyTorch style in the previous section) using DiffSharp's `-->` composition operator, which allows you to seamlessly compose `Model` instances and differentiable `Tensor->Tensor` functions.

*)
// DiffSharp style

// Model as a composition of models and Tensor->Tensor functions
let generator =
    dsharp.view([-1;nz])
    --> Linear(nz, 256)
    --> dsharp.leakyRelu(0.2)
    --> Linear(256, 512)
    --> dsharp.leakyRelu(0.2)
    --> Linear(512, 1024)
    --> dsharp.leakyRelu(0.2)
    --> Linear(1024, 28*28)
    --> dsharp.tanh

// Model as a composition of models and Tensor->Tensor functions
let discriminator =
    dsharp.view([-1; 28*28])
    --> Linear(28*28, 1024)
    --> dsharp.leakyRelu(0.2)
    --> dsharp.dropout(0.3)
    --> Linear(1024, 512)
    --> dsharp.leakyRelu(0.2)
    --> dsharp.dropout(0.3)
    --> Linear(512, 256)
    --> dsharp.leakyRelu(0.2)
    --> dsharp.dropout(0.3)
    --> Linear(256, 1)
    --> dsharp.sigmoid

print generator
print discriminator(* output: 
Model(Linear(128, 256), Linear(256, 512), Linear(512, 1024), Linear(1024, 784))
Model(Linear(784, 1024), Linear(1024, 512), Linear(512, 256), Linear(256, 1))*)

