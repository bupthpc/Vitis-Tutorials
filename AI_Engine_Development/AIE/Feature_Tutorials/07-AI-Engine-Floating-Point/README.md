<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>AI Engine Development</h1>
    <a href="https://www.xilinx.com/products/design-tools/vitis.html">See Vitis™ Development Environment on xilinx.com</br></a>
    <a href="https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html">See Vitis™ AI Development Environment on xilinx.com</a>
    </td>
 </tr>
</table>

# Using Floating-Point in the AI Engine

***Version: Vitis 2024.2***

## Introduction

The purpose of this set of examples is to understand floating-point vector computations within the AI Engine.

**IMPORTANT**: Before beginning the tutorial, make sure that you have installed the Vitis software.  The AMD Vitis&trade; release includes all the embedded base platforms including the VCK190 base platform that is used in this tutorial. In addition, ensure that you have downloaded the Common Images for Embedded Vitis Platforms from [this link] (https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html).
The ‘common image’ package contains a prebuilt Linux kernel and root file system that can be used with the Versal board for embedded design development using Vitis.
Before starting this tutorial, run the following steps:

1. Go to the directory where you have unzipped the Versal Common Image package.
2. In a Bash shell, run the `/Common Images Dir/xilinx-versal-common-v2024.2/environment-setup-cortexa72-cortexa53-xilinx-linux` script. This script sets up the `SDKTARGETSYSROOT` and `CXX` variables. If the script is not present, run the `/Common Images Dir/xilinx-versal-common-v2024.2/sdk.sh`.
3. Set up your `ROOTFS`, and `IMAGE` to point to the `rootfs.ext4` and `Image` files located in the `/Common Images Dir/xilinx-versal-common-v2024.2` directory.
4. Set up your `PLATFORM_REPO_PATHS` environment variable to `$XILINX_VITIS/base_platforms/`.
This tutorial targets the VCK190 production board for 2024.2 version.



## AI Engine Architecture Details

Versal™ adaptive SoCs combine programmable logic (PL), processing system (PS), and AI Engines with leading-edge memory and interfacing technologies to deliver powerful heterogeneous acceleration for any application. The hardware and software are targeted for programming and optimization by data scientists and software and hardware developers. A host of tools, software, libraries, IP, middleware, and frameworks enable Versal adaptive SoCs to support all industry-standard design flows.

![missing image](./images/Versal.png)


**AI Engine array**

![missing image](./images/AIEngineArray.png)

As seen in the image above, each AI Engine is connected to four memory modules on the four cardinal directions.
The AI Engine and memory modules are both connected to the AXI-Stream interconnect.

The AI Engine is  a VLIW (7-way) processor that contains:

- Instruction Fetch and Decode Unit
- A Scalar Unit
- A Vector Unit (SIMD)
- Three Address Generator Units
- Memory and Stream Interface

**AI Engine Module**

![missing image](./images/AIEngine.png)

Have a look at the fixed-point unit pipeline, as well as floating-point unit pipeline within the vector unit.

### Fixed-Point Pipeline

![missing image](./images/FixedPointPipeline.jpg)

In this pipeline, one can see the data selection and shuffling units, PMXL, PMXR, and PMC. The pre-add (PRA) is just before the multiply block and then two lane reduction blocks (PSA, PSB) allows to perform up to 128 multiplies and get an output on 16 lanes down to two lanes. The accumulator block is fed either by its own output (AM) or by the upshift output. The feedback on the ACC block is only one clock cycle.

### Floating-point Pipeline

![missing image](./images/FloatingPointPipeline.jpg)

In this pipeline, one can see that the selection and shuffling units (PMXL, PMC) are the same as in the fixed-point unit. Unlike the fixed-point pipeline there is no lane reduction unit, so the lanes that you have at the input will also be there at the output. Another difference is that the post-accumulator is on two clock cycles. If the goal is to reuse the same accumulator over and over, only one `fpmac` per two clock cycles can be issued.

## Floating-point intrinsics

There is a limited set of intrinsics with which a multitude of operations can be performed. All of them return either a `vector<float,8>` or  `vector<cfloat,4>`, 256-bit vectors.

The basic addition, subtraction, and negation functions are as follows:

- fpadd
- fpadd_abs
- fpsub
- fpsub_abs
- fpneg
- fpneg_abs
- fpabs

The simple multiplier function is available with the following options:

- fpmul
- fpabs_mul
- fpneg_mul
- fpneg_abs_mul

The multiplication accumulation/subtraction function has the following options:

- fpmac
- fpmac_abs
- fpmsc
- fpmsc_abs

On top of these various intrinsics, you have a fully configurable version multiplier and multiply-accumulate:

- fpmul_conf
- fpmac_conf

#### Start, offset

In all the subsequent intrinsics, the input vector(s) go through a data shuffling function that is controlled by two parameters:

- Start
- Offset

Let us take the `fpmul` function:

_vector<float,8>_ **fpmul**(_vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**, _vector<float,8>_ **zbuf**, _int_ **zstart**, _unsigned int_ **zoffs**)

- **xbuf, xstart, xoffs**: first buffer and shuffling parameters
- **zbuf, zstart, zoffs**: second buffer and shuffling parameters

- **Start**: starting offset for all lanes of the buffer
- **Offset**: additional, lane-dependent offset for the buffer. Definition takes 4 bits per lane.

For example:

_vector<float,8>_ **ret** = **fpmul**(**xbuf**,**2**,**0x210FEDCB**,**zbuf**,**7**,**0x76543210**)

```
for (i = 0 ; i < 8 ; i++)
  ret[i] =  xbuf[xstart + xoffs[i]] * zbuf[zstart + zoffs[i]]
```

All values in hexadecimal:

| ret <br> Index <br> (Lane) | | xbuf <br> Start | xbuf <br> Offset | Final <br> xbuf <br> Index | | zbuf <br> Start | zbuf <br> Offset | Final <br> zbuf <br> Index |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  0  | | 2  | B  | D  | | 7 | 0 | 7 |
| 1   | | 2  | C  | E  | | 7 | 1 | 8 |
| 2   | | 2  | D  | F  | | 7 | 2 | 9 |
| 3   | | 2  | E  | 10  | | 7 | 3 | A |
| 4   | | 2  | F  | 11  | | 7 | 4 | B |
| 5   | | 2  | 0  | 2   | | 7 | 5 | C |
| 6   | | 2  | 1  | 3   | | 7 | 6 | D |
| 7   | | 2  | 2  | 4   | | 7 | 7 | E |


#### fpneg, fpabs, fpadd, fpsub

##### fpneg

Output is the opposite of its input. Input can be either `float` or `cfloat` forming a 512-bit or a 1024-bit buffer (`vector<float,32>, vector<cfloat,16>, vector<cfloat,16>, vector<cfloat,8>`). The output is a 256-bit buffer as all the floating-point operators (`vector<float,8>, vector<cfloat,4>`).

_vector<float,8>_ 	**fpneg** (_vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**)

```
for (i = 0 ; i < 8 ; i++)
  ret[i] = - xbuf[xstart + xoffs[i]]
```

##### fpabs

Output is the absolute value of the input.
It takes only real floating-point input vectors.

##### fpneg_abs

Output is the negation of the absolute value of the input.
It takes only real floating-point input vectors.

##### fpadd, fpsub

Output is the sum (the subtraction) of the input buffers.

_vector<float,8>_ 	**fpadd** (_vector<float,8>_ **acc**, _vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**)

| Parameter | Comment |
| ---: | :--- |
| **acc**  |  First addition input buffer. It has the same type as the output.|
| **xbuf**  |  Second addition input buffer.|
| **xstart**|	Starting offset for all lanes of X.|
| **xoffs**	|  4 bits per lane: Additional lane-dependent offset for X.|

The executed operation is:

```
for (i = 0 ; i < 8 ; i++)
  ret[i] = acc[i] + xbuf[xstart + xoffs[i]]
```

Allowed datatypes:
- **acc**: `vector<float,8>, vector<cfloat,4>`
- **xbuf**: `vector<float,32>, vector<float,16>, vector<cfloat,16>, vector<cfloat,8>`

##### fpadd_abs, fpsub_abs

Adds or subtracts the absolute value of the second buffer to the first one.

```
for (i = 0 ; i < 8 ; i++)
  ret[i] = acc[i] +/- abs(xbuf[xstart + xoffs[i]])
```






#### fpmul

The simple floating-point multiplier comes in many different flavors mixing or not `float` and `cfloat` vector data types. When two `cfloat` are involved, the intrinsic results in two microcode instructions that must be scheduled. The first buffer can be either 512 or 1024-bit long (`vector<float,32>, vector<float,16>, vector<cfloat,16>, vector<cfloat,8>`), the second buffer is always 256-bit long (`vector<float,8>, vector<cfloat,4>`). Any combination is allowed.

_vector<float,8>_ **fpmul**(_vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**, _vector<float,8>_ **zbuf**, _int_ **zstart**, _unsigned int_ **zoffs**)

Returns the multiplication result.

| Parameter | Comment |
| ---: | :--- |
| **xbuf**  |  First multiplication input buffer.|
| **xstart**|	Starting offset for all lanes of X.|
| **xoffs**	|  4 bits per lane, additional lane-dependent offset for X.|
| **zbuf**	|  Second multiplication input buffer.|
| **zstart**|	Starting offset for all lanes of Z. This must be a compile time constant.|
| **zoffs** |  	4 bits per lane, additional lane-dependent offset for Z.|


```
for (i = 0 ; i < 8 ; i++)
  ret[i] =  xbuf[xstart + xoffs[i]] * zbuf[zstart + zoffs[i]]
```


#### fpabs_mul

Only for real arguments. Signature is identical to `fpmul`:

_vector<float,8>_ **fpabs_mul**(_vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**, _vector<float,8>_ **zbuf**, _int_ **zstart**, _unsigned int_ **zoffs**)

 It returns the absolute value of the product:
 ```
 for (i = 0 ; i < 8 ; i++)
   ret[i] =  abs(xbuf[xstart + xoffs[i]] * zbuf[zstart + zoffs[i]])
 ```

#### fpneg_mul

Signature is identical to `fpmul`:

_vector<float,8>_ **fpneg_mul**(_vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**, _vector<float,8>_ **zbuf**, _int_ **zstart**, _unsigned int_ **zoffs**)

It returns the opposite value of the product:

```
for (i = 0 ; i < 8 ; i++)
  ret[i] =  - xbuf[xstart + xoffs[i]] * zbuf[zstart + zoffs[i]]
```

#### fpneg_abs_mul

Only for real arguments. Signature is identical to `fpmul`:

_vector<float,8>_ **fpneg_abs_mul**(_vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**, _vector<float,8>_ **zbuf**, _int_ **zstart**, _unsigned int_ **zoffs**)

It returns the opposite value of the product:

```
for (i = 0 ; i < 8 ; i++)
 ret[i] =  - abs ( xbuf[xstart + xoffs[i]] * zbuf[zstart + zoffs[i]] )
```


#### fpmac, fpmsc, fpmac_abs, fpmsc_abs

For all these functions, there is one more argument compared to the `fpmul` function. This is the previous value of the accumulator.

_vector<float,8>_ **fpmac**(_vector<float,8>_ **acc**, _vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**, _vector<float,8>_ **zbuf**, _int_ **zstart**, _unsigned int_ **zoffs**)

- **fpmac** : multiply operands and add to the accumulator
- **fpmsc** : multiply operands and subtract from the accumulator
- **fpmac_abs** : multiply operands and add the absolute value to the accumulator
- **fpmsc_abs** : multiply operands and subtract the absolute value from the accumulator

 The two "abs" variants are available only for real arguments.

#### fpmul_conf, fpmac_conf

These functions are fully configurable `fpmul` and `fpmac`  functions. The output can be considered to always have eight values because each part of the complex float is treated differently. A `vector<cfloat,4>` will have the loop interating over real0 - complex0 - real1 - complex1 ... This capability is introduced to allow flexibility and implement operations on conjugates.


_vector<float,8>_ **fpmac_conf**(_vector<float,8>_ **acc**, _vector<float,32>_ **xbuf**, _int_ **xstart**, _unsigned int_ **xoffs**, _vector<float,8>_ **zbuf**, _int_ **zstart**, _unsigned int_ **zoffs**, _bool_ **ones**, _bool_ **abs**, _unsigned int_ **addmode**, _unsigned int_ **addmask**, _unsigned int_ **cmpmode**, _unsigned int &_ **cmp**)

Returns the multiplication result.

| Parameter | Comment |
| ---: | :--- |
| **acc**  |  Current accumulator value. This parameter does not exist for **fpmul_conf**.|
| **xbuf**  |  First multiplication input buffer.|
| **xstart**|	Starting offset for all lanes of X.|
| **xoffs**	|  4 bits per lane: Additional lane-dependent offset for X.|
| **zbuf**	|  **Optional** Second multiplication input buffer. If **zbuf** is not specified, **xbuf** is taken as the second buffer.|
| **zstart**|	Starting offset for all lanes of Z. This must be a compile time constant.|
| **zoffs** |  	4 bits per lane: Additional lane-dependent offset for Z.|
| **ones**   |  If true, all lanes from Z are replaced with 1.0. |
| **abs**   | If true, the absolute value is taken before accumulation.  |
| **addmode**   |  Select one of the `fpadd_add` (all add), `fpadd_sub` (all sub), `fpadd_mixadd` or `fpadd_mixsub` (add-sub or sub-add pairs). This must be a compile time constant. |
| **addmask**   | 8 x 1 LSB bits: Corresponding lane is negated if bit is set (depending on addmode).  |
| **cmpmode**   |  Use `fpcmp_lt` to select the minimum between accumulator and result of multiplication per lane, `fpcmp_ge` for the maximum and `fpcmp_nrm` for the usual sum. |
| **cmp**   |  **Optional** 8 x 1 LSB bits: When using `fpcmp_ge` or `fpcmp_lt` in "cmpmode", it sets a bit if accumulator was chosen (per lane). |


## Floating-Point Examples

The purpose of this set of examples is to show how to use floating-point computations within the AI Engines in different schemes:

- FIR filter
- Matrix Multiply

### FIR Filter

As there is no post-add lane reduction hardware in the floating-point pipeline of the AI Engine, all outputs will always be on eight lanes (`float`) or four lanes (`cfloat`). This means that we can compute eight (four) lanes in parallel, each time with a single coefficient, using `fpmul` and then `fpmac` for all the coefficients, one by one.

The floating-point accumulator has a latency of two clock cycles, so two `fpmac` instructions using the same accumulator cannot be used back to back, but only every other cycle. Code can be optimized by using two accumulators, used in turn, that gets added at the end to get the final result.

- Navigate to the `FIRFilter` directory.
- Type `make allaie` in the console and wait for completion of the three following stages:
  1. `aie`
  2. `aiesim`
  3. `aiecmp`
  4. `aieviz`

The last stage is opening `vitis_analyzer` that allows you to visualize the graph of the design and the simulation process timeline.

In this design, you learned:

- How to use real floating-point data and coefficients in FIR filters.
- How to handle complex floating-point data and complex floating-points coefficients in FIR filters.
- How to organize the compute sequence.
- How to use: `fpmul`, `fpmac`, and `fpadd` in the real and complex case.


#### Real Floating-Point Filter

In the example, the filter has 16 coefficients which do not fit within a 256-bit register. The register must be updated in the middle of the computation.

For data storage a small 512-bit register is used. It is decomposed in two 256-bit parts: W0, W1.

- First iteration
  - Part W0 is loaded with first 8 samples  (0...7)
  - Part W1 with the next 8 samples (8...15)
  - Part W0 with the following ones (16...23)
- Second iteration
  - Part W0 : 8...15
  - Part W1 : 16...23
  - Part W0 : 24...31

#### Complex Floating-Point Filter

`cfloat x cfloat` multiplications take two cycles to perform due to the abscence of the post add. These two parts can be interleaved with the two cycle latency of the accumulator.

There are still 16 coefficients but now they are complex; hence, double the size. The coefficients have to be updated four times for a complete iteration. The data transfer is also slightly more complex.

### Matrix Multiply

In this example, a matrix multiply (A*B) example is shown with the simple fpmul and fpmac intrinsics in the real and complex case. In the complex case there are also two other examples using the `fpmul_conf` and `fpmac_conf` intrinsics to compute A*B and A*conj(B).

Intrinsics being lane by lane computation oriented, this feature will be used to compute a number of consecutive columns of the output matrix. The latency of two of the accumulator is absorbed by computing two rows of the output matrix.

All the parameter settings for the `fpmul/mac_conf` intrinsics are explained in the code itself.

- Navigate to the `MatMult` directory.
- Type `make all` in the console and wait for the completions of the 3 stages:
  1. `aie`
  2. `aiesim`
  3. `aieviz`

The last stage is opening `vitis_analyzer` that allows you to visualize the graph of the design and the simulation process timeline.

In this design you learned:

- How to organize matrix multiply compute sequence when using real or complex floating-point numbers.
- How to handle complex floating-point data and complex floating-points coefficients in FIR filters.
- How to use `fpmul_conf` and `fpmac_conf` intrinsics.





## Support

GitHub issues will be used for tracking requests and bugs. For questions, go to [support.xilinx.com](https://support.xilinx.com/).

<p class="sphinxhide" align="center"><sub>Copyright © 2020–2024 Advanced Micro Devices, Inc</sub><br><sup>XD021</sup></br></p>

<p class="sphinxhide" align="center"><sup><a href="https://www.amd.com/en/corporate/copyright">Terms and Conditions</a></sup></p>
