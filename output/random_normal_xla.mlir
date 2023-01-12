ENTRY %a_inference_random_normal_494__.23 () -> f32[3,4] {
  %constant.1 = s32[2]{0} constant({3, 4}), metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %constant.5 = s32[2]{0} constant({3, 4}), metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %slice.6 = s32[1]{0} slice(s32[2]{0} %constant.5), slice={[0:1]}, metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %reshape.7 = s32[] reshape(s32[1]{0} %slice.6), metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %convert.8 = s32[] convert(s32[] %reshape.7), metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %slice.9 = s32[1]{0} slice(s32[2]{0} %constant.5), slice={[1:2]}, metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %reshape.10 = s32[] reshape(s32[1]{0} %slice.9), metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %convert.11 = s32[] convert(s32[] %reshape.10), metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %constant.12 = f32[] constant(1), metadata={op_type="Mul" op_name="random_normal/mul" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %constant.16 = f32[] constant(0), metadata={op_type="AddV2" op_name="random_normal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %constant.3 = f32[] constant(0), metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %constant.2 = f32[] constant(1), metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %rng.4 = f32[3,4]{1,0} rng(f32[] %constant.3, f32[] %constant.2), distribution=rng_normal, metadata={op_type="RandomStandardNormal" op_name="random_normal/RandomStandardNormal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %constant.13 = f32[] constant(1), metadata={op_type="Mul" op_name="random_normal/mul" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %broadcast.14 = f32[3,4]{1,0} broadcast(f32[] %constant.13), dimensions={}, metadata={op_type="Mul" op_name="random_normal/mul" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %multiply.15 = f32[3,4]{1,0} multiply(f32[3,4]{1,0} %rng.4, f32[3,4]{1,0} %broadcast.14), metadata={op_type="Mul" op_name="random_normal/mul" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %constant.17 = f32[] constant(0), metadata={op_type="AddV2" op_name="random_normal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %broadcast.18 = f32[3,4]{1,0} broadcast(f32[] %constant.17), dimensions={}, metadata={op_type="AddV2" op_name="random_normal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %add.19 = f32[3,4]{1,0} add(f32[3,4]{1,0} %multiply.15, f32[3,4]{1,0} %broadcast.18), metadata={op_type="AddV2" op_name="random_normal" source_file="<ipython-input-28-7efce820a03a>" source_line=8}
  %reshape.20 = f32[3,4]{1,0} reshape(f32[3,4]{1,0} %add.19), metadata={op_name="XLA_Retvals"}
  %tuple.21 = (f32[3,4]{1,0}) tuple(f32[3,4]{1,0} %reshape.20), metadata={op_name="XLA_Retvals"}
  ROOT %get-tuple-element.22 = f32[3,4]{1,0} get-tuple-element((f32[3,4]{1,0}) %tuple.21), index=0, metadata={op_name="XLA_Retvals"}
}