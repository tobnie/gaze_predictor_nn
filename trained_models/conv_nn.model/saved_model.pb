
Ì¢
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.22v2.8.2-0-g2ea19cbb5758ó

Input_Conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameInput_Conv/kernel

%Input_Conv/kernel/Read/ReadVariableOpReadVariableOpInput_Conv/kernel*&
_output_shapes
:*
dtype0
v
Input_Conv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameInput_Conv/bias
o
#Input_Conv/bias/Read/ReadVariableOpReadVariableOpInput_Conv/bias*
_output_shapes
:*
dtype0
w
Dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È(*
shared_nameDense1/kernel
p
!Dense1/kernel/Read/ReadVariableOpReadVariableOpDense1/kernel*
_output_shapes
:	È(*
dtype0
n
Dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*
shared_nameDense1/bias
g
Dense1/bias/Read/ReadVariableOpReadVariableOpDense1/bias*
_output_shapes
:(*
dtype0
~
Dense2_Out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*"
shared_nameDense2_Out/kernel
w
%Dense2_Out/kernel/Read/ReadVariableOpReadVariableOpDense2_Out/kernel*
_output_shapes

:(*
dtype0
v
Dense2_Out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameDense2_Out/bias
o
#Dense2_Out/bias/Read/ReadVariableOpReadVariableOpDense2_Out/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/Input_Conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Input_Conv/kernel/m

,Adam/Input_Conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Input_Conv/kernel/m*&
_output_shapes
:*
dtype0

Adam/Input_Conv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Input_Conv/bias/m
}
*Adam/Input_Conv/bias/m/Read/ReadVariableOpReadVariableOpAdam/Input_Conv/bias/m*
_output_shapes
:*
dtype0

Adam/Dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È(*%
shared_nameAdam/Dense1/kernel/m
~
(Adam/Dense1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense1/kernel/m*
_output_shapes
:	È(*
dtype0
|
Adam/Dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*#
shared_nameAdam/Dense1/bias/m
u
&Adam/Dense1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense1/bias/m*
_output_shapes
:(*
dtype0

Adam/Dense2_Out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*)
shared_nameAdam/Dense2_Out/kernel/m

,Adam/Dense2_Out/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense2_Out/kernel/m*
_output_shapes

:(*
dtype0

Adam/Dense2_Out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Dense2_Out/bias/m
}
*Adam/Dense2_Out/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense2_Out/bias/m*
_output_shapes
:*
dtype0

Adam/Input_Conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Input_Conv/kernel/v

,Adam/Input_Conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Input_Conv/kernel/v*&
_output_shapes
:*
dtype0

Adam/Input_Conv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Input_Conv/bias/v
}
*Adam/Input_Conv/bias/v/Read/ReadVariableOpReadVariableOpAdam/Input_Conv/bias/v*
_output_shapes
:*
dtype0

Adam/Dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È(*%
shared_nameAdam/Dense1/kernel/v
~
(Adam/Dense1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense1/kernel/v*
_output_shapes
:	È(*
dtype0
|
Adam/Dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*#
shared_nameAdam/Dense1/bias/v
u
&Adam/Dense1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense1/bias/v*
_output_shapes
:(*
dtype0

Adam/Dense2_Out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*)
shared_nameAdam/Dense2_Out/kernel/v

,Adam/Dense2_Out/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense2_Out/kernel/v*
_output_shapes

:(*
dtype0

Adam/Dense2_Out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/Dense2_Out/bias/v
}
*Adam/Dense2_Out/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense2_Out/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ê4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*4
valueû3Bø3 Bñ3
Û
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 
¦

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses*
¦

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*
°
3iter

4beta_1

5beta_2
	6decay
7learning_ratemamb#mc$md+me,mfvgvh#vi$vj+vk,vl*
.
0
1
#2
$3
+4
,5*
.
0
1
#2
$3
+4
,5*
* 
°
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

=serving_default* 
a[
VARIABLE_VALUEInput_Conv/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEInput_Conv/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEDense1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEDense1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEDense2_Out/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEDense2_Out/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 

Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

W0
X1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ytotal
	Zcount
[	variables
\	keras_api*
8
	]total
	^count
_	variables
`	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

[	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

]0
^1*

_	variables*
~
VARIABLE_VALUEAdam/Input_Conv/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/Input_Conv/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/Dense1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Dense1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/Dense2_Out/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/Dense2_Out/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/Input_Conv/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/Input_Conv/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/Dense1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Dense1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/Dense2_Out/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/Dense2_Out/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

 serving_default_Input_Conv_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
¬
StatefulPartitionedCallStatefulPartitionedCall serving_default_Input_Conv_inputInput_Conv/kernelInput_Conv/biasDense1/kernelDense1/biasDense2_Out/kernelDense2_Out/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_4480437
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¸

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%Input_Conv/kernel/Read/ReadVariableOp#Input_Conv/bias/Read/ReadVariableOp!Dense1/kernel/Read/ReadVariableOpDense1/bias/Read/ReadVariableOp%Dense2_Out/kernel/Read/ReadVariableOp#Dense2_Out/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/Input_Conv/kernel/m/Read/ReadVariableOp*Adam/Input_Conv/bias/m/Read/ReadVariableOp(Adam/Dense1/kernel/m/Read/ReadVariableOp&Adam/Dense1/bias/m/Read/ReadVariableOp,Adam/Dense2_Out/kernel/m/Read/ReadVariableOp*Adam/Dense2_Out/bias/m/Read/ReadVariableOp,Adam/Input_Conv/kernel/v/Read/ReadVariableOp*Adam/Input_Conv/bias/v/Read/ReadVariableOp(Adam/Dense1/kernel/v/Read/ReadVariableOp&Adam/Dense1/bias/v/Read/ReadVariableOp,Adam/Dense2_Out/kernel/v/Read/ReadVariableOp*Adam/Dense2_Out/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_4480621

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameInput_Conv/kernelInput_Conv/biasDense1/kernelDense1/biasDense2_Out/kernelDense2_Out/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Input_Conv/kernel/mAdam/Input_Conv/bias/mAdam/Dense1/kernel/mAdam/Dense1/bias/mAdam/Dense2_Out/kernel/mAdam/Dense2_Out/bias/mAdam/Input_Conv/kernel/vAdam/Input_Conv/bias/vAdam/Dense1/kernel/vAdam/Dense1/bias/vAdam/Dense2_Out/kernel/vAdam/Dense2_Out/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_4480712â
Ü

I__inference_sequential_1_layer_call_and_return_conditional_losses_4480391

inputsC
)input_conv_conv2d_readvariableop_resource:8
*input_conv_biasadd_readvariableop_resource:8
%dense1_matmul_readvariableop_resource:	È(4
&dense1_biasadd_readvariableop_resource:(;
)dense2_out_matmul_readvariableop_resource:(8
*dense2_out_biasadd_readvariableop_resource:
identity¢Dense1/BiasAdd/ReadVariableOp¢Dense1/MatMul/ReadVariableOp¢!Dense2_Out/BiasAdd/ReadVariableOp¢ Dense2_Out/MatMul/ReadVariableOp¢!Input_Conv/BiasAdd/ReadVariableOp¢ Input_Conv/Conv2D/ReadVariableOp
 Input_Conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¯
Input_Conv/Conv2DConv2Dinputs(Input_Conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

!Input_Conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Input_Conv/BiasAddBiasAddInput_Conv/Conv2D:output:0)Input_Conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Input_Conv/ReluReluInput_Conv/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
max_pooling2d/MaxPoolMaxPoolInput_Conv/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿH  
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	È(*
dtype0
Dense1/MatMulMatMulflatten/Reshape:output:0$Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 Dense2_Out/MatMul/ReadVariableOpReadVariableOp)dense2_out_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
Dense2_Out/MatMulMatMulDense1/Relu:activations:0(Dense2_Out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!Dense2_Out/BiasAdd/ReadVariableOpReadVariableOp*dense2_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Dense2_Out/BiasAddBiasAddDense2_Out/MatMul:product:0)Dense2_Out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityDense2_Out/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp"^Dense2_Out/BiasAdd/ReadVariableOp!^Dense2_Out/MatMul/ReadVariableOp"^Input_Conv/BiasAdd/ReadVariableOp!^Input_Conv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp2F
!Dense2_Out/BiasAdd/ReadVariableOp!Dense2_Out/BiasAdd/ReadVariableOp2D
 Dense2_Out/MatMul/ReadVariableOp Dense2_Out/MatMul/ReadVariableOp2F
!Input_Conv/BiasAdd/ReadVariableOp!Input_Conv/BiasAdd/ReadVariableOp2D
 Input_Conv/Conv2D/ReadVariableOp Input_Conv/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê	
ø
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480152

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Ê	
ø
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480517

inputs0
matmul_readvariableop_resource:(-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs


õ
C__inference_Dense1_layer_call_and_return_conditional_losses_4480498

inputs1
matmul_readvariableop_resource:	È(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	È(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
û

%__inference_signature_wrapper_4480437
input_conv_input!
unknown:
	unknown_0:
	unknown_1:	È(
	unknown_2:(
	unknown_3:(
	unknown_4:
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_4480080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInput_Conv_input
«	

.__inference_sequential_1_layer_call_fn_4480174
input_conv_input!
unknown:
	unknown_0:
	unknown_1:	È(
	unknown_2:(
	unknown_3:(
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480159o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInput_Conv_input
â&

"__inference__wrapped_model_4480080
input_conv_inputP
6sequential_1_input_conv_conv2d_readvariableop_resource:E
7sequential_1_input_conv_biasadd_readvariableop_resource:E
2sequential_1_dense1_matmul_readvariableop_resource:	È(A
3sequential_1_dense1_biasadd_readvariableop_resource:(H
6sequential_1_dense2_out_matmul_readvariableop_resource:(E
7sequential_1_dense2_out_biasadd_readvariableop_resource:
identity¢*sequential_1/Dense1/BiasAdd/ReadVariableOp¢)sequential_1/Dense1/MatMul/ReadVariableOp¢.sequential_1/Dense2_Out/BiasAdd/ReadVariableOp¢-sequential_1/Dense2_Out/MatMul/ReadVariableOp¢.sequential_1/Input_Conv/BiasAdd/ReadVariableOp¢-sequential_1/Input_Conv/Conv2D/ReadVariableOp¬
-sequential_1/Input_Conv/Conv2D/ReadVariableOpReadVariableOp6sequential_1_input_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ó
sequential_1/Input_Conv/Conv2DConv2Dinput_conv_input5sequential_1/Input_Conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
¢
.sequential_1/Input_Conv/BiasAdd/ReadVariableOpReadVariableOp7sequential_1_input_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Å
sequential_1/Input_Conv/BiasAddBiasAdd'sequential_1/Input_Conv/Conv2D:output:06sequential_1/Input_Conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_1/Input_Conv/ReluRelu(sequential_1/Input_Conv/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
"sequential_1/max_pooling2d/MaxPoolMaxPool*sequential_1/Input_Conv/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides
k
sequential_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿH  ¬
sequential_1/flatten/ReshapeReshape+sequential_1/max_pooling2d/MaxPool:output:0#sequential_1/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
)sequential_1/Dense1/MatMul/ReadVariableOpReadVariableOp2sequential_1_dense1_matmul_readvariableop_resource*
_output_shapes
:	È(*
dtype0°
sequential_1/Dense1/MatMulMatMul%sequential_1/flatten/Reshape:output:01sequential_1/Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
*sequential_1/Dense1/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_dense1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0²
sequential_1/Dense1/BiasAddBiasAdd$sequential_1/Dense1/MatMul:product:02sequential_1/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(x
sequential_1/Dense1/ReluRelu$sequential_1/Dense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(¤
-sequential_1/Dense2_Out/MatMul/ReadVariableOpReadVariableOp6sequential_1_dense2_out_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0¹
sequential_1/Dense2_Out/MatMulMatMul&sequential_1/Dense1/Relu:activations:05sequential_1/Dense2_Out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential_1/Dense2_Out/BiasAdd/ReadVariableOpReadVariableOp7sequential_1_dense2_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential_1/Dense2_Out/BiasAddBiasAdd(sequential_1/Dense2_Out/MatMul:product:06sequential_1/Dense2_Out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
IdentityIdentity(sequential_1/Dense2_Out/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿá
NoOpNoOp+^sequential_1/Dense1/BiasAdd/ReadVariableOp*^sequential_1/Dense1/MatMul/ReadVariableOp/^sequential_1/Dense2_Out/BiasAdd/ReadVariableOp.^sequential_1/Dense2_Out/MatMul/ReadVariableOp/^sequential_1/Input_Conv/BiasAdd/ReadVariableOp.^sequential_1/Input_Conv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2X
*sequential_1/Dense1/BiasAdd/ReadVariableOp*sequential_1/Dense1/BiasAdd/ReadVariableOp2V
)sequential_1/Dense1/MatMul/ReadVariableOp)sequential_1/Dense1/MatMul/ReadVariableOp2`
.sequential_1/Dense2_Out/BiasAdd/ReadVariableOp.sequential_1/Dense2_Out/BiasAdd/ReadVariableOp2^
-sequential_1/Dense2_Out/MatMul/ReadVariableOp-sequential_1/Dense2_Out/MatMul/ReadVariableOp2`
.sequential_1/Input_Conv/BiasAdd/ReadVariableOp.sequential_1/Input_Conv/BiasAdd/ReadVariableOp2^
-sequential_1/Input_Conv/Conv2D/ReadVariableOp-sequential_1/Input_Conv/Conv2D/ReadVariableOp:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInput_Conv_input
È

,__inference_Dense2_Out_layer_call_fn_4480507

inputs
unknown:(
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480152o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ(: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 
_user_specified_nameinputs
Æ
`
D__inference_flatten_layer_call_and_return_conditional_losses_4480123

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿH  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¯
E
)__inference_flatten_layer_call_fn_4480472

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4480123a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs

¬
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480159

inputs,
input_conv_4480111: 
input_conv_4480113:!
dense1_4480137:	È(
dense1_4480139:($
dense2_out_4480153:( 
dense2_out_4480155:
identity¢Dense1/StatefulPartitionedCall¢"Dense2_Out/StatefulPartitionedCall¢"Input_Conv/StatefulPartitionedCall
"Input_Conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_4480111input_conv_4480113*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480110ð
max_pooling2d/PartitionedCallPartitionedCall+Input_Conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480089Ø
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4480123
Dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_4480137dense1_4480139*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Dense1_layer_call_and_return_conditional_losses_4480136
"Dense2_Out/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0dense2_out_4480153dense2_out_4480155*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480152z
IdentityIdentity+Dense2_Out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp^Dense1/StatefulPartitionedCall#^Dense2_Out/StatefulPartitionedCall#^Input_Conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2H
"Dense2_Out/StatefulPartitionedCall"Dense2_Out/StatefulPartitionedCall2H
"Input_Conv/StatefulPartitionedCall"Input_Conv/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

I__inference_sequential_1_layer_call_and_return_conditional_losses_4480418

inputsC
)input_conv_conv2d_readvariableop_resource:8
*input_conv_biasadd_readvariableop_resource:8
%dense1_matmul_readvariableop_resource:	È(4
&dense1_biasadd_readvariableop_resource:(;
)dense2_out_matmul_readvariableop_resource:(8
*dense2_out_biasadd_readvariableop_resource:
identity¢Dense1/BiasAdd/ReadVariableOp¢Dense1/MatMul/ReadVariableOp¢!Dense2_Out/BiasAdd/ReadVariableOp¢ Dense2_Out/MatMul/ReadVariableOp¢!Input_Conv/BiasAdd/ReadVariableOp¢ Input_Conv/Conv2D/ReadVariableOp
 Input_Conv/Conv2D/ReadVariableOpReadVariableOp)input_conv_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¯
Input_Conv/Conv2DConv2Dinputs(Input_Conv/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides

!Input_Conv/BiasAdd/ReadVariableOpReadVariableOp*input_conv_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Input_Conv/BiasAddBiasAddInput_Conv/Conv2D:output:0)Input_Conv/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Input_Conv/ReluReluInput_Conv/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
max_pooling2d/MaxPoolMaxPoolInput_Conv/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
ksize
*
paddingVALID*
strides
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿH  
flatten/ReshapeReshapemax_pooling2d/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes
:	È(*
dtype0
Dense1/MatMulMatMulflatten/Reshape:output:0$Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype0
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(^
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(
 Dense2_Out/MatMul/ReadVariableOpReadVariableOp)dense2_out_matmul_readvariableop_resource*
_output_shapes

:(*
dtype0
Dense2_Out/MatMulMatMulDense1/Relu:activations:0(Dense2_Out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!Dense2_Out/BiasAdd/ReadVariableOpReadVariableOp*dense2_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Dense2_Out/BiasAddBiasAddDense2_Out/MatMul:product:0)Dense2_Out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityDense2_Out/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp"^Dense2_Out/BiasAdd/ReadVariableOp!^Dense2_Out/MatMul/ReadVariableOp"^Input_Conv/BiasAdd/ReadVariableOp!^Input_Conv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp2F
!Dense2_Out/BiasAdd/ReadVariableOp!Dense2_Out/BiasAdd/ReadVariableOp2D
 Dense2_Out/MatMul/ReadVariableOp Dense2_Out/MatMul/ReadVariableOp2F
!Input_Conv/BiasAdd/ReadVariableOp!Input_Conv/BiasAdd/ReadVariableOp2D
 Input_Conv/Conv2D/ReadVariableOp Input_Conv/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã

(__inference_Dense1_layer_call_fn_4480487

inputs
unknown:	È(
	unknown_0:(
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Dense1_layer_call_and_return_conditional_losses_4480136o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs


õ
C__inference_Dense1_layer_call_and_return_conditional_losses_4480136

inputs1
matmul_readvariableop_resource:	È(-
biasadd_readvariableop_resource:(
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	È(*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:(*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÈ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
	

.__inference_sequential_1_layer_call_fn_4480364

inputs!
unknown:
	unknown_0:
	unknown_1:	È(
	unknown_2:(
	unknown_3:(
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480250o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
¡
,__inference_Input_Conv_layer_call_fn_4480446

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

.__inference_sequential_1_layer_call_fn_4480347

inputs!
unknown:
	unknown_0:
	unknown_1:	È(
	unknown_2:(
	unknown_3:(
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480159o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ýl

#__inference__traced_restore_4480712
file_prefix<
"assignvariableop_input_conv_kernel:0
"assignvariableop_1_input_conv_bias:3
 assignvariableop_2_dense1_kernel:	È(,
assignvariableop_3_dense1_bias:(6
$assignvariableop_4_dense2_out_kernel:(0
"assignvariableop_5_dense2_out_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: F
,assignvariableop_15_adam_input_conv_kernel_m:8
*assignvariableop_16_adam_input_conv_bias_m:;
(assignvariableop_17_adam_dense1_kernel_m:	È(4
&assignvariableop_18_adam_dense1_bias_m:(>
,assignvariableop_19_adam_dense2_out_kernel_m:(8
*assignvariableop_20_adam_dense2_out_bias_m:F
,assignvariableop_21_adam_input_conv_kernel_v:8
*assignvariableop_22_adam_input_conv_bias_v:;
(assignvariableop_23_adam_dense1_kernel_v:	È(4
&assignvariableop_24_adam_dense1_bias_v:(>
,assignvariableop_25_adam_dense2_out_kernel_v:(8
*assignvariableop_26_adam_dense2_out_bias_v:
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ø
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_input_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_input_conv_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense2_out_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense2_out_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_input_conv_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_input_conv_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_dense1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_dense1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_dense2_out_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense2_out_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_input_conv_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_input_conv_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_dense1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp,assignvariableop_25_adam_dense2_out_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_dense2_out_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¡
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¶
K
/__inference_max_pooling2d_layer_call_fn_4480462

inputs
identityØ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480089
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¬
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480250

inputs,
input_conv_4480232: 
input_conv_4480234:!
dense1_4480239:	È(
dense1_4480241:($
dense2_out_4480244:( 
dense2_out_4480246:
identity¢Dense1/StatefulPartitionedCall¢"Dense2_Out/StatefulPartitionedCall¢"Input_Conv/StatefulPartitionedCall
"Input_Conv/StatefulPartitionedCallStatefulPartitionedCallinputsinput_conv_4480232input_conv_4480234*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480110ð
max_pooling2d/PartitionedCallPartitionedCall+Input_Conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480089Ø
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4480123
Dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_4480239dense1_4480241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Dense1_layer_call_and_return_conditional_losses_4480136
"Dense2_Out/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0dense2_out_4480244dense2_out_4480246*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480152z
IdentityIdentity+Dense2_Out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp^Dense1/StatefulPartitionedCall#^Dense2_Out/StatefulPartitionedCall#^Input_Conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2H
"Dense2_Out/StatefulPartitionedCall"Dense2_Out/StatefulPartitionedCall2H
"Input_Conv/StatefulPartitionedCall"Input_Conv/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480457

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480089

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480467

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«	

.__inference_sequential_1_layer_call_fn_4480282
input_conv_input!
unknown:
	unknown_0:
	unknown_1:	È(
	unknown_2:(
	unknown_3:(
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480250o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInput_Conv_input


G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480110

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
`
D__inference_flatten_layer_call_and_return_conditional_losses_4480478

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿH  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ
:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
¬
¶
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480303
input_conv_input,
input_conv_4480285: 
input_conv_4480287:!
dense1_4480292:	È(
dense1_4480294:($
dense2_out_4480297:( 
dense2_out_4480299:
identity¢Dense1/StatefulPartitionedCall¢"Dense2_Out/StatefulPartitionedCall¢"Input_Conv/StatefulPartitionedCall
"Input_Conv/StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputinput_conv_4480285input_conv_4480287*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480110ð
max_pooling2d/PartitionedCallPartitionedCall+Input_Conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480089Ø
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4480123
Dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_4480292dense1_4480294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Dense1_layer_call_and_return_conditional_losses_4480136
"Dense2_Out/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0dense2_out_4480297dense2_out_4480299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480152z
IdentityIdentity+Dense2_Out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp^Dense1/StatefulPartitionedCall#^Dense2_Out/StatefulPartitionedCall#^Input_Conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2H
"Dense2_Out/StatefulPartitionedCall"Dense2_Out/StatefulPartitionedCall2H
"Input_Conv/StatefulPartitionedCall"Input_Conv/StatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInput_Conv_input
¬
¶
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480324
input_conv_input,
input_conv_4480306: 
input_conv_4480308:!
dense1_4480313:	È(
dense1_4480315:($
dense2_out_4480318:( 
dense2_out_4480320:
identity¢Dense1/StatefulPartitionedCall¢"Dense2_Out/StatefulPartitionedCall¢"Input_Conv/StatefulPartitionedCall
"Input_Conv/StatefulPartitionedCallStatefulPartitionedCallinput_conv_inputinput_conv_4480306input_conv_4480308*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480110ð
max_pooling2d/PartitionedCallPartitionedCall+Input_Conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480089Ø
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_4480123
Dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_4480313dense1_4480315*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_Dense1_layer_call_and_return_conditional_losses_4480136
"Dense2_Out/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0dense2_out_4480318dense2_out_4480320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480152z
IdentityIdentity+Dense2_Out/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
NoOpNoOp^Dense1/StatefulPartitionedCall#^Dense2_Out/StatefulPartitionedCall#^Input_Conv/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2H
"Dense2_Out/StatefulPartitionedCall"Dense2_Out/StatefulPartitionedCall2H
"Input_Conv/StatefulPartitionedCall"Input_Conv/StatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_nameInput_Conv_input
©<

 __inference__traced_save_4480621
file_prefix0
,savev2_input_conv_kernel_read_readvariableop.
*savev2_input_conv_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop0
,savev2_dense2_out_kernel_read_readvariableop.
*savev2_dense2_out_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_input_conv_kernel_m_read_readvariableop5
1savev2_adam_input_conv_bias_m_read_readvariableop3
/savev2_adam_dense1_kernel_m_read_readvariableop1
-savev2_adam_dense1_bias_m_read_readvariableop7
3savev2_adam_dense2_out_kernel_m_read_readvariableop5
1savev2_adam_dense2_out_bias_m_read_readvariableop7
3savev2_adam_input_conv_kernel_v_read_readvariableop5
1savev2_adam_input_conv_bias_v_read_readvariableop3
/savev2_adam_dense1_kernel_v_read_readvariableop1
-savev2_adam_dense1_bias_v_read_readvariableop7
3savev2_adam_dense2_out_kernel_v_read_readvariableop5
1savev2_adam_dense2_out_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: õ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¥
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_input_conv_kernel_read_readvariableop*savev2_input_conv_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop,savev2_dense2_out_kernel_read_readvariableop*savev2_dense2_out_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_input_conv_kernel_m_read_readvariableop1savev2_adam_input_conv_bias_m_read_readvariableop/savev2_adam_dense1_kernel_m_read_readvariableop-savev2_adam_dense1_bias_m_read_readvariableop3savev2_adam_dense2_out_kernel_m_read_readvariableop1savev2_adam_dense2_out_bias_m_read_readvariableop3savev2_adam_input_conv_kernel_v_read_readvariableop1savev2_adam_input_conv_bias_v_read_readvariableop/savev2_adam_dense1_kernel_v_read_readvariableop-savev2_adam_dense1_bias_v_read_readvariableop3savev2_adam_dense2_out_kernel_v_read_readvariableop1savev2_adam_dense2_out_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ö
_input_shapesÄ
Á: :::	È(:(:(:: : : : : : : : : :::	È(:(:(::::	È(:(:(:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	È(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	È(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	È(: 

_output_shapes
:(:$ 

_output_shapes

:(: 

_output_shapes
::

_output_shapes
: "ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
U
Input_Conv_inputA
"serving_default_Input_Conv_input:0ÿÿÿÿÿÿÿÿÿ>

Dense2_Out0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ói
õ
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
»

#kernel
$bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
»

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
¿
3iter

4beta_1

5beta_2
	6decay
7learning_ratemamb#mc$md+me,mfvgvh#vi$vj+vk,vl"
	optimizer
J
0
1
#2
$3
+4
,5"
trackable_list_wrapper
J
0
1
#2
$3
+4
,5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
.__inference_sequential_1_layer_call_fn_4480174
.__inference_sequential_1_layer_call_fn_4480347
.__inference_sequential_1_layer_call_fn_4480364
.__inference_sequential_1_layer_call_fn_4480282À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480391
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480418
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480303
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480324À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÖBÓ
"__inference__wrapped_model_4480080Input_Conv_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
=serving_default"
signature_map
+:)2Input_Conv/kernel
:2Input_Conv/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_Input_Conv_layer_call_fn_4480446¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480457¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ù2Ö
/__inference_max_pooling2d_layer_call_fn_4480462¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480467¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ó2Ð
)__inference_flatten_layer_call_fn_4480472¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_flatten_layer_call_and_return_conditional_losses_4480478¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 :	È(2Dense1/kernel
:(2Dense1/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_Dense1_layer_call_fn_4480487¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_Dense1_layer_call_and_return_conditional_losses_4480498¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
#:!(2Dense2_Out/kernel
:2Dense2_Out/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
Ö2Ó
,__inference_Dense2_Out_layer_call_fn_4480507¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480517¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÕBÒ
%__inference_signature_wrapper_4480437Input_Conv_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ytotal
	Zcount
[	variables
\	keras_api"
_tf_keras_metric
N
	]total
	^count
_	variables
`	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
Y0
Z1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
:  (2total
:  (2count
.
]0
^1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
0:.2Adam/Input_Conv/kernel/m
": 2Adam/Input_Conv/bias/m
%:#	È(2Adam/Dense1/kernel/m
:(2Adam/Dense1/bias/m
(:&(2Adam/Dense2_Out/kernel/m
": 2Adam/Dense2_Out/bias/m
0:.2Adam/Input_Conv/kernel/v
": 2Adam/Input_Conv/bias/v
%:#	È(2Adam/Dense1/kernel/v
:(2Adam/Dense1/bias/v
(:&(2Adam/Dense2_Out/kernel/v
": 2Adam/Dense2_Out/bias/v¤
C__inference_Dense1_layer_call_and_return_conditional_losses_4480498]#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 |
(__inference_Dense1_layer_call_fn_4480487P#$0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÈ
ª "ÿÿÿÿÿÿÿÿÿ(§
G__inference_Dense2_Out_layer_call_and_return_conditional_losses_4480517\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_Dense2_Out_layer_call_fn_4480507O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ(
ª "ÿÿÿÿÿÿÿÿÿ·
G__inference_Input_Conv_layer_call_and_return_conditional_losses_4480457l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_Input_Conv_layer_call_fn_4480446_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ«
"__inference__wrapped_model_4480080#$+,A¢>
7¢4
2/
Input_Conv_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

Dense2_Out$!

Dense2_Outÿÿÿÿÿÿÿÿÿ©
D__inference_flatten_layer_call_and_return_conditional_losses_4480478a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª "&¢#

0ÿÿÿÿÿÿÿÿÿÈ
 
)__inference_flatten_layer_call_fn_4480472T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ

ª "ÿÿÿÿÿÿÿÿÿÈí
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_4480467R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_layer_call_fn_4480462R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480303z#$+,I¢F
?¢<
2/
Input_Conv_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480324z#$+,I¢F
?¢<
2/
Input_Conv_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480391p#$+,?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
I__inference_sequential_1_layer_call_and_return_conditional_losses_4480418p#$+,?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_sequential_1_layer_call_fn_4480174m#$+,I¢F
?¢<
2/
Input_Conv_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_1_layer_call_fn_4480282m#$+,I¢F
?¢<
2/
Input_Conv_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_1_layer_call_fn_4480347c#$+,?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_1_layer_call_fn_4480364c#$+,?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ
%__inference_signature_wrapper_4480437#$+,U¢R
¢ 
KªH
F
Input_Conv_input2/
Input_Conv_inputÿÿÿÿÿÿÿÿÿ"7ª4
2

Dense2_Out$!

Dense2_Outÿÿÿÿÿÿÿÿÿ