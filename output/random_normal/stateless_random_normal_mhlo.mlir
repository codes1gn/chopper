module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1286 : i32}}  {
  func @main() -> tensor<2x3xf32> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "identity_RetVal"}} {
    %0 = mhlo.constant dense<1.000000e+00> : tensor<2x3xf32>
    %1 = mhlo.constant dense<2.000000e+00> : tensor<2x3xf32>
    %2 = mhlo.constant dense<1234> : tensor<ui64>
    %3 = mhlo.constant dense<5678> : tensor<ui64>
    %4 = mhlo.constant dense<1053357856> : tensor<ui32>
    %5 = mhlo.constant dense<38149673> : tensor<ui32>
    %6 = mhlo.constant dense<3468443297> : tensor<ui32>
    %7 = mhlo.constant dense<2454539055> : tensor<ui32>
    %8 = mhlo.constant dense<4095070582> : tensor<ui32>
    %9 = mhlo.constant dense<1440634813> : tensor<ui32>
    %10 = mhlo.constant dense<3081166340> : tensor<ui32>
    %11 = mhlo.constant dense<426730571> : tensor<ui32>
    %12 = mhlo.constant dense<2067262098> : tensor<ui32>
    %13 = mhlo.constant dense<3707793625> : tensor<ui32>
    %14 = mhlo.constant dense<3716387409> : tensor<ui32>
    %15 = mhlo.constant dense<572253132> : tensor<ui32>
    %16 = mhlo.constant dense<1723086151> : tensor<ui32>
    %17 = mhlo.constant dense<2873919170> : tensor<ui32>
    %18 = mhlo.constant dense<4024752189> : tensor<ui32>
    %19 = mhlo.constant dense<880617912> : tensor<ui32>
    %20 = mhlo.constant dense<2031450931> : tensor<ui32>
    %21 = mhlo.constant dense<3182283950> : tensor<ui32>
    %22 = mhlo.constant dense<3528531795> : tensor<ui64>
    %23 = mhlo.constant dense<3449720151> : tensor<ui64>
    %24 = mhlo.constant dense<32> : tensor<ui64>
    %25 = mhlo.constant dense<0> : tensor<ui64>
    %26 = mhlo.constant dense<9> : tensor<6xui32>
    %27 = mhlo.constant dense<1.1920929E-7> : tensor<6xf32>
    %28 = mhlo.constant dense<1.000000e+00> : tensor<6xf32>
    %29 = mhlo.constant dense<0.000000e+00> : tensor<6xf32>
    %30 = mhlo.constant dense<1.000000e-07> : tensor<3xf32>
    %31 = mhlo.constant dense<6.28318548> : tensor<3xf32>
    %32 = mhlo.constant dense<-2.000000e+00> : tensor<3xf32>
    %33 = mhlo.shift_left %3, %24 : tensor<ui64>
    %34 = mhlo.or %33, %2 : tensor<ui64>
    %35 = "mhlo.convert"(%34) : (tensor<ui64>) -> tensor<ui32>
    %36 = mhlo.shift_right_logical %34, %24 : tensor<ui64>
    %37 = "mhlo.convert"(%36) : (tensor<ui64>) -> tensor<ui32>
    %38 = "mhlo.convert"(%35) : (tensor<ui32>) -> tensor<ui64>
    %39 = "mhlo.convert"(%37) : (tensor<ui32>) -> tensor<ui64>
    %40 = "mhlo.convert"(%38) : (tensor<ui64>) -> tensor<ui32>
    %41 = mhlo.shift_right_logical %38, %24 : tensor<ui64>
    %42 = "mhlo.convert"(%41) : (tensor<ui64>) -> tensor<ui32>
    %43 = "mhlo.convert"(%39) : (tensor<ui64>) -> tensor<ui32>
    %44 = mhlo.shift_right_logical %39, %24 : tensor<ui64>
    %45 = "mhlo.convert"(%44) : (tensor<ui64>) -> tensor<ui32>
    %46 = "mhlo.convert"(%40) : (tensor<ui32>) -> tensor<ui64>
    %47 = mhlo.multiply %46, %22 : tensor<ui64>
    %48 = "mhlo.convert"(%47) : (tensor<ui64>) -> tensor<ui32>
    %49 = mhlo.shift_right_logical %47, %24 : tensor<ui64>
    %50 = "mhlo.convert"(%49) : (tensor<ui64>) -> tensor<ui32>
    %51 = "mhlo.convert"(%43) : (tensor<ui32>) -> tensor<ui64>
    %52 = mhlo.multiply %51, %23 : tensor<ui64>
    %53 = "mhlo.convert"(%52) : (tensor<ui64>) -> tensor<ui32>
    %54 = mhlo.shift_right_logical %52, %24 : tensor<ui64>
    %55 = "mhlo.convert"(%54) : (tensor<ui64>) -> tensor<ui32>
    %56 = mhlo.xor %55, %42 : tensor<ui32>
    %57 = mhlo.xor %56, %4 : tensor<ui32>
    %58 = mhlo.xor %50, %45 : tensor<ui32>
    %59 = mhlo.xor %58, %5 : tensor<ui32>
    %60 = "mhlo.convert"(%57) : (tensor<ui32>) -> tensor<ui64>
    %61 = mhlo.multiply %60, %22 : tensor<ui64>
    %62 = "mhlo.convert"(%61) : (tensor<ui64>) -> tensor<ui32>
    %63 = mhlo.shift_right_logical %61, %24 : tensor<ui64>
    %64 = "mhlo.convert"(%63) : (tensor<ui64>) -> tensor<ui32>
    %65 = "mhlo.convert"(%59) : (tensor<ui32>) -> tensor<ui64>
    %66 = mhlo.multiply %65, %23 : tensor<ui64>
    %67 = "mhlo.convert"(%66) : (tensor<ui64>) -> tensor<ui32>
    %68 = mhlo.shift_right_logical %66, %24 : tensor<ui64>
    %69 = "mhlo.convert"(%68) : (tensor<ui64>) -> tensor<ui32>
    %70 = mhlo.xor %69, %53 : tensor<ui32>
    %71 = mhlo.xor %70, %13 : tensor<ui32>
    %72 = mhlo.xor %64, %48 : tensor<ui32>
    %73 = mhlo.xor %72, %21 : tensor<ui32>
    %74 = "mhlo.convert"(%71) : (tensor<ui32>) -> tensor<ui64>
    %75 = mhlo.multiply %74, %22 : tensor<ui64>
    %76 = "mhlo.convert"(%75) : (tensor<ui64>) -> tensor<ui32>
    %77 = mhlo.shift_right_logical %75, %24 : tensor<ui64>
    %78 = "mhlo.convert"(%77) : (tensor<ui64>) -> tensor<ui32>
    %79 = "mhlo.convert"(%73) : (tensor<ui32>) -> tensor<ui64>
    %80 = mhlo.multiply %79, %23 : tensor<ui64>
    %81 = "mhlo.convert"(%80) : (tensor<ui64>) -> tensor<ui32>
    %82 = mhlo.shift_right_logical %80, %24 : tensor<ui64>
    %83 = "mhlo.convert"(%82) : (tensor<ui64>) -> tensor<ui32>
    %84 = mhlo.xor %83, %67 : tensor<ui32>
    %85 = mhlo.xor %84, %12 : tensor<ui32>
    %86 = mhlo.xor %78, %62 : tensor<ui32>
    %87 = mhlo.xor %86, %20 : tensor<ui32>
    %88 = "mhlo.convert"(%85) : (tensor<ui32>) -> tensor<ui64>
    %89 = mhlo.multiply %88, %22 : tensor<ui64>
    %90 = "mhlo.convert"(%89) : (tensor<ui64>) -> tensor<ui32>
    %91 = mhlo.shift_right_logical %89, %24 : tensor<ui64>
    %92 = "mhlo.convert"(%91) : (tensor<ui64>) -> tensor<ui32>
    %93 = "mhlo.convert"(%87) : (tensor<ui32>) -> tensor<ui64>
    %94 = mhlo.multiply %93, %23 : tensor<ui64>
    %95 = "mhlo.convert"(%94) : (tensor<ui64>) -> tensor<ui32>
    %96 = mhlo.shift_right_logical %94, %24 : tensor<ui64>
    %97 = "mhlo.convert"(%96) : (tensor<ui64>) -> tensor<ui32>
    %98 = mhlo.xor %97, %81 : tensor<ui32>
    %99 = mhlo.xor %98, %11 : tensor<ui32>
    %100 = mhlo.xor %92, %76 : tensor<ui32>
    %101 = mhlo.xor %100, %19 : tensor<ui32>
    %102 = "mhlo.convert"(%99) : (tensor<ui32>) -> tensor<ui64>
    %103 = mhlo.multiply %102, %22 : tensor<ui64>
    %104 = "mhlo.convert"(%103) : (tensor<ui64>) -> tensor<ui32>
    %105 = mhlo.shift_right_logical %103, %24 : tensor<ui64>
    %106 = "mhlo.convert"(%105) : (tensor<ui64>) -> tensor<ui32>
    %107 = "mhlo.convert"(%101) : (tensor<ui32>) -> tensor<ui64>
    %108 = mhlo.multiply %107, %23 : tensor<ui64>
    %109 = "mhlo.convert"(%108) : (tensor<ui64>) -> tensor<ui32>
    %110 = mhlo.shift_right_logical %108, %24 : tensor<ui64>
    %111 = "mhlo.convert"(%110) : (tensor<ui64>) -> tensor<ui32>
    %112 = mhlo.xor %111, %95 : tensor<ui32>
    %113 = mhlo.xor %112, %10 : tensor<ui32>
    %114 = mhlo.xor %106, %90 : tensor<ui32>
    %115 = mhlo.xor %114, %18 : tensor<ui32>
    %116 = "mhlo.convert"(%113) : (tensor<ui32>) -> tensor<ui64>
    %117 = mhlo.multiply %116, %22 : tensor<ui64>
    %118 = "mhlo.convert"(%117) : (tensor<ui64>) -> tensor<ui32>
    %119 = mhlo.shift_right_logical %117, %24 : tensor<ui64>
    %120 = "mhlo.convert"(%119) : (tensor<ui64>) -> tensor<ui32>
    %121 = "mhlo.convert"(%115) : (tensor<ui32>) -> tensor<ui64>
    %122 = mhlo.multiply %121, %23 : tensor<ui64>
    %123 = "mhlo.convert"(%122) : (tensor<ui64>) -> tensor<ui32>
    %124 = mhlo.shift_right_logical %122, %24 : tensor<ui64>
    %125 = "mhlo.convert"(%124) : (tensor<ui64>) -> tensor<ui32>
    %126 = mhlo.xor %125, %109 : tensor<ui32>
    %127 = mhlo.xor %126, %9 : tensor<ui32>
    %128 = mhlo.xor %120, %104 : tensor<ui32>
    %129 = mhlo.xor %128, %17 : tensor<ui32>
    %130 = "mhlo.convert"(%127) : (tensor<ui32>) -> tensor<ui64>
    %131 = mhlo.multiply %130, %22 : tensor<ui64>
    %132 = "mhlo.convert"(%131) : (tensor<ui64>) -> tensor<ui32>
    %133 = mhlo.shift_right_logical %131, %24 : tensor<ui64>
    %134 = "mhlo.convert"(%133) : (tensor<ui64>) -> tensor<ui32>
    %135 = "mhlo.convert"(%129) : (tensor<ui32>) -> tensor<ui64>
    %136 = mhlo.multiply %135, %23 : tensor<ui64>
    %137 = "mhlo.convert"(%136) : (tensor<ui64>) -> tensor<ui32>
    %138 = mhlo.shift_right_logical %136, %24 : tensor<ui64>
    %139 = "mhlo.convert"(%138) : (tensor<ui64>) -> tensor<ui32>
    %140 = mhlo.xor %139, %123 : tensor<ui32>
    %141 = mhlo.xor %140, %8 : tensor<ui32>
    %142 = mhlo.xor %134, %118 : tensor<ui32>
    %143 = mhlo.xor %142, %16 : tensor<ui32>
    %144 = "mhlo.convert"(%141) : (tensor<ui32>) -> tensor<ui64>
    %145 = mhlo.multiply %144, %22 : tensor<ui64>
    %146 = "mhlo.convert"(%145) : (tensor<ui64>) -> tensor<ui32>
    %147 = mhlo.shift_right_logical %145, %24 : tensor<ui64>
    %148 = "mhlo.convert"(%147) : (tensor<ui64>) -> tensor<ui32>
    %149 = "mhlo.convert"(%143) : (tensor<ui32>) -> tensor<ui64>
    %150 = mhlo.multiply %149, %23 : tensor<ui64>
    %151 = mhlo.shift_right_logical %150, %24 : tensor<ui64>
    %152 = "mhlo.convert"(%151) : (tensor<ui64>) -> tensor<ui32>
    %153 = mhlo.xor %152, %137 : tensor<ui32>
    %154 = mhlo.xor %153, %7 : tensor<ui32>
    %155 = mhlo.xor %148, %132 : tensor<ui32>
    %156 = mhlo.xor %155, %15 : tensor<ui32>
    %157 = "mhlo.convert"(%154) : (tensor<ui32>) -> tensor<ui64>
    %158 = mhlo.multiply %157, %22 : tensor<ui64>
    %159 = mhlo.shift_right_logical %158, %24 : tensor<ui64>
    %160 = "mhlo.convert"(%159) : (tensor<ui64>) -> tensor<ui32>
    %161 = "mhlo.convert"(%156) : (tensor<ui32>) -> tensor<ui64>
    %162 = mhlo.multiply %161, %23 : tensor<ui64>
    %163 = "mhlo.convert"(%162) : (tensor<ui64>) -> tensor<ui32>
    %164 = mhlo.xor %160, %146 : tensor<ui32>
    %165 = mhlo.xor %164, %14 : tensor<ui32>
    %166 = "mhlo.convert"(%165) : (tensor<ui32>) -> tensor<ui64>
    %167 = mhlo.multiply %166, %23 : tensor<ui64>
    %168 = "mhlo.convert"(%167) : (tensor<ui64>) -> tensor<ui32>
    %169 = mhlo.shift_right_logical %167, %24 : tensor<ui64>
    %170 = "mhlo.convert"(%169) : (tensor<ui64>) -> tensor<ui32>
    %171 = mhlo.xor %170, %163 : tensor<ui32>
    %172 = mhlo.xor %171, %6 : tensor<ui32>
    %173 = "mhlo.convert"(%168) : (tensor<ui32>) -> tensor<ui64>
    %174 = mhlo.shift_left %173, %24 : tensor<ui64>
    %175 = "mhlo.convert"(%172) : (tensor<ui32>) -> tensor<ui64>
    %176 = mhlo.or %175, %174 : tensor<ui64>
    %177 = mhlo.shift_left %25, %24 : tensor<ui64>
    %178 = "mhlo.reshape"(%177) : (tensor<ui64>) -> tensor<1xui64>
    %179 = "mhlo.reshape"(%176) : (tensor<ui64>) -> tensor<1xui64>
    %180 = "mhlo.bitcast_convert"(%179) : (tensor<1xui64>) -> tensor<1xui64>
    %181 = "mhlo.bitcast_convert"(%178) : (tensor<1xui64>) -> tensor<1xui64>
    %182 = "mhlo.concatenate"(%180, %181) {dimension = 0 : i64} : (tensor<1xui64>, tensor<1xui64>) -> tensor<2xui64>
    %183 = "mhlo.rng_bit_generator"(%182) {rng_algorithm = 0 : i32} : (tensor<2xui64>) -> tuple<tensor<2xui64>, tensor<6xui32>>
    %184 = "mhlo.get_tuple_element"(%183) {index = 1 : i32} : (tuple<tensor<2xui64>, tensor<6xui32>>) -> tensor<6xui32>
    %185 = mhlo.shift_right_logical %184, %26 : tensor<6xui32>
    %186 = "mhlo.convert"(%185) : (tensor<6xui32>) -> tensor<6xf32>
    %187 = mhlo.multiply %186, %27 : tensor<6xf32>
    %188 = mhlo.multiply %187, %28 : tensor<6xf32>
    %189 = mhlo.add %188, %29 : tensor<6xf32>
    %190 = "mhlo.slice"(%189) {limit_indices = dense<3> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xf32>) -> tensor<3xf32>
    %191 = "mhlo.slice"(%189) {limit_indices = dense<6> : tensor<1xi64>, start_indices = dense<3> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<6xf32>) -> tensor<3xf32>
    %192 = mhlo.maximum %190, %30 : tensor<3xf32>
    %193 = mhlo.multiply %191, %31 : tensor<3xf32>
    %194 = "mhlo.log"(%192) : (tensor<3xf32>) -> tensor<3xf32>
    %195 = mhlo.multiply %194, %32 : tensor<3xf32>
    %196 = "mhlo.sqrt"(%195) : (tensor<3xf32>) -> tensor<3xf32>
    %197 = "mhlo.sine"(%193) : (tensor<3xf32>) -> tensor<3xf32>
    %198 = mhlo.multiply %197, %196 : tensor<3xf32>
    %199 = "mhlo.cosine"(%193) : (tensor<3xf32>) -> tensor<3xf32>
    %200 = mhlo.multiply %199, %196 : tensor<3xf32>
    %201 = "mhlo.concatenate"(%198, %200) {dimension = 0 : i64} : (tensor<3xf32>, tensor<3xf32>) -> tensor<6xf32>
    %202 = "mhlo.reshape"(%201) : (tensor<6xf32>) -> tensor<2x3xf32>
    %203 = mhlo.multiply %202, %1 : tensor<2x3xf32>
    %204 = mhlo.add %203, %0 : tensor<2x3xf32>
    return %204 : tensor<2x3xf32>
  }
}